"""trace_store.py：审查轨迹唯一写入口 + 断点续跑游标。

为什么所有写操作必须从这一个类走（唯一写入口）：
  幂等约束只在 SQL 键上有效，如果任何代码绕过它自己 INSERT，唯一键就成了
  摆设；集中一处后，"重放同一步"在数据层必然撞 (review_id, step_seq) 键
  被跳过——评测六"无重复结论"不靠测试碰运气，靠约束保证。

断点续跑协议（评测六的实现基础）：
  1. 每个节点跑完 → record_step 存全量 state_snapshot，并推进 reviews.current_node；
  2. 进程被 kill -9 后重启 → load_progress 读回 last_node + snapshot；
  3. 执行器只从 last_node 的下一个节点继续，已完成的节点连函数都不进。
"""
import json
import time
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TraceStore:
    """审查轨迹与结论的唯一写入口。每个方法开短连接短事务（单机 demo 足够，
    生产换连接池只改这一处）。"""

    def __init__(self, dsn: str):
        self.dsn = dsn

    def _conn(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn, row_factory=dict_row)

    # ---- 审查单生命周期 -------------------------------------------------

    def create_review(self, review_id: str, document_id: str, mode: str = "rule") -> None:
        """建审查单。ON CONFLICT DO NOTHING：同 review_id 重提直接复用旧单（幂等）。"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO reviews (review_id, document_id, status, mode, started_at)
                   VALUES (%s, %s, 'running', %s, %s)
                   ON CONFLICT (review_id) DO NOTHING""",
                (review_id, document_id, mode, _now()),
            )

    def finish_review(self, review_id: str, status: str, error: str = "") -> None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """UPDATE reviews
                      SET status = %s, error = %s, finished_at = %s
                    WHERE review_id = %s""",
                (status, error, _now(), review_id),
            )

    def get_review(self, review_id: str) -> dict | None:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM reviews WHERE review_id = %s", (review_id,))
            return cur.fetchone()

    # ---- 轨迹（幂等写 + 恢复游标）---------------------------------------

    def record_step(
        self,
        review_id: str,
        step_seq: int,
        agent: str,
        state_snapshot: dict,
        input_summary: str = "",
        output_summary: str = "",
        latency_ms: int = 0,
        token_est: int = 0,
        status: str = "ok",
    ) -> bool:
        """落一步轨迹 + 推进恢复游标。返回是否真正写入（False=重放被跳过）。

        撞唯一键 = 这一步早已完成 → DO NOTHING 且不推进游标，续跑时执行器
        会据此跳过整个节点。"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO review_steps
                       (review_id, step_seq, agent, input_summary, output_summary,
                        state_snapshot, latency_ms, token_est, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (review_id, step_seq) DO NOTHING""",
                (review_id, step_seq, agent, input_summary, output_summary,
                 json.dumps(state_snapshot, ensure_ascii=False),
                 latency_ms, token_est, status),
            )
            inserted = cur.rowcount == 1
            if inserted:  # 只有真写入才推进游标，重放不动它
                cur.execute(
                    """UPDATE reviews SET current_node = %s
                        WHERE review_id = %s""",
                    (agent, review_id),
                )
            return inserted

    def load_progress(self, review_id: str) -> tuple[str, dict, int]:
        """读恢复游标：(最后完成的节点名, 全量 state 快照, 最后步号)。
        从未跑过 → ("", {}, 0)，执行器从第一个节点开始。
        步号一并返回：续跑的 step_seq 接着编号，轨迹才是一条连续序列。"""
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT agent, step_seq, state_snapshot FROM review_steps
                    WHERE review_id = %s
                    ORDER BY step_seq DESC LIMIT 1""",
                (review_id,),
            )
            row = cur.fetchone()
        if row is None:
            return "", {}, 0
        return row["agent"], row["state_snapshot"], row["step_seq"]

    def list_steps(self, review_id: str) -> list[dict]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT step_seq, agent, input_summary, output_summary,
                          latency_ms, token_est, status, created_at
                     FROM review_steps WHERE review_id = %s
                    ORDER BY step_seq""",
                (review_id,),
            )
            return cur.fetchall()

    # ---- 结论（断点重放不重复）------------------------------------------

    def save_finding(self, review_id: str, finding: dict) -> bool:
        """写一条结论，(review_id, item_id) 撞键即跳过。"""
        ev = finding.get("evidence", {}) or {}
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO findings
                       (finding_id, review_id, item_id, item_seq, item_text,
                        verdict, confidence, chunk_id, clause_no, quote,
                        rationale, verify_status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (review_id, item_id) DO NOTHING""",
                (
                    f"{review_id}:{finding['item_id']}",
                    review_id,
                    finding["item_id"],
                    finding.get("item_seq", 0),
                    finding.get("item_text", ""),
                    finding["verdict"],
                    finding.get("confidence", 0.0),
                    ev.get("chunk_id", ""),
                    ev.get("clause_no", ""),
                    ev.get("quote", ""),
                    finding.get("rationale", ""),
                    finding.get("verify", "n/a"),
                ),
            )
            return cur.rowcount == 1

    def list_findings(self, review_id: str) -> list[dict]:
        with self._conn() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT item_id, item_seq, item_text, verdict, confidence,
                          chunk_id, clause_no, quote, rationale, verify_status
                     FROM findings WHERE review_id = %s ORDER BY item_seq""",
                (review_id,),
            )
            return cur.fetchall()