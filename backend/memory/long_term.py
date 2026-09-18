"""long_term.py：长期记忆——历史审查结论沉淀为经验库（L3）。

设计：不符合项沉淀为 (domain, attribute, pattern) 三元组，pattern 是
"这类文档特征 → 常见不符合项"的一句话模式。新审查时按领域+属性召回，
注入 reviewer 的判定上下文（提示"这类属性历史上常见的坑"）。

幂等：三元组唯一键，重复经验 ON CONFLICT 只涨 hit_count——
经验库越用越准靠计数，不靠行数膨胀。
评测四（记忆消融）用 enabled=False 屏蔽召回，其余配置完全一致。
"""
import psycopg
from psycopg.rows import dict_row


class MemoryStore:
    def __init__(self, dsn: str):
        self.dsn = dsn

    def upsert(self, domain: str, attribute: str, pattern: str,
               clause_no: str = "") -> None:
        """沉淀一条经验；同三元组重复出现只涨计数。"""
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO memory_entries (memory_id, domain, attribute, pattern, clause_no)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (domain, attribute, pattern)
                   DO UPDATE SET hit_count = memory_entries.hit_count + 1,
                                 updated_at = now()""",
                (f"mem-{domain}-{attribute}-{abs(hash(pattern)) & 0xffffff:x}",
                 domain, attribute, pattern[:200], clause_no),
            )

    def search(self, domain: str, attribute: str = "", limit: int = 3) -> list[dict]:
        """召回经验：同领域优先、同属性加权（属性完全一致的排前面）。"""
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT attribute, pattern, clause_no, hit_count
                     FROM memory_entries
                    WHERE domain = %s AND (%s = '' OR attribute = %s)
                    ORDER BY (attribute = %s) DESC, hit_count DESC
                    LIMIT %s""",
                (domain, attribute, attribute, attribute, limit),
            )
            return cur.fetchall()

    def count(self) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) AS n FROM memory_entries")
            return cur.fetchone()["n"]

    def clear(self) -> None:
        """评测四消融用：清空经验库重建基线。"""
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM memory_entries")
