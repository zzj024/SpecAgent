"""评测六：断点恢复测试——kill -9 注入 n 次，重启续跑，校验结果一致性。

方法：固定一份 6 检查项文档，先跑一遍 uninterrupted 基线；
然后 10 轮注入：子进程 SPECAGENT_CRASH_AFTER=<node> 在该节点完成落库后
os._exit(137)（等价 kill -9：无清理、无 finally），父进程重启同 review_id
续跑至完成。校验四条：退出码 137 / 续跑完成 / 无重复无丢失检查项 /
findings 与基线逐项一致。恢复成功率 = 通过轮数 / 10。
"""
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.graph import run_review                     # noqa: E402
from core.config import DSN                             # noqa: E402
from reliability.trace_store import TraceStore          # noqa: E402

HERE = Path(__file__).parent

DOC = """# 断点续跑测试文档（评测六）

[粗糙度] 配合表面粗糙度 Ra 3.2 μm
[余高] 对接焊缝余高 4.0 mm
[接地] 电气柜保护接地电阻 5.0 Ω
[比例] 主视图绘图比例 1:3
[咬边] 咬边深度 0.9 mm
[通道] 柜前维修通道宽度 600 mm
"""
N_ROUNDS = 10
CRASH_NODES = ["router", "retriever", "reviewer", "verifier", "writer"]
CODE = (
    "import sys; sys.path.insert(0, 'backend');"
    "from agents.graph import run_review;"
    "run_review({doc!r}, document_name='eval6.md', review_id={rid!r},"
    " mode='rule', use_memory=False)"
)


def _purge(rid: str) -> None:
    s = TraceStore(DSN)
    with s._conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM findings WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM review_steps WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM reviews WHERE review_id = %s", (rid,))


def _key(report: dict) -> list:
    return sorted((f["item_seq"], f["verdict"]) for f in report.get("findings", []))


def main() -> None:
    base_rid = "rev-eval6-base"
    _purge(base_rid)
    baseline = run_review(DOC, document_name="eval6.md", review_id=base_rid,
                          mode="rule", use_memory=False)
    base_key = _key(baseline)

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    ok = 0
    rounds = []
    for i in range(N_ROUNDS):
        node = CRASH_NODES[i % len(CRASH_NODES)]
        rid = f"rev-eval6-{i:02d}"
        _purge(rid)
        crashed = subprocess.run(
            [sys.executable, "-c", CODE.format(doc=DOC, rid=rid)],
            env={**env, "SPECAGENT_CRASH_AFTER": node},
            capture_output=True, text=True, timeout=600)
        exit_ok = crashed.returncode == 137

        resumed = run_review(DOC, document_name="eval6.md", review_id=rid,
                             mode="rule", use_memory=False)      # “重启续跑”
        no_dup = len({s for s, _ in _key(resumed)}) == 6
        consistent = _key(resumed) == base_key and "error" not in resumed
        passed = exit_ok and no_dup and consistent
        ok += passed
        rounds.append({"round": i, "crash_at": node, "exit137": exit_ok,
                       "no_dup": no_dup, "consistent": consistent, "pass": passed})
        _purge(rid)
    _purge(base_rid)

    result = {
        "date": date.today().isoformat(), "n_rounds": N_ROUNDS,
        "resume_success_rate": round(ok / N_ROUNDS, 4), "rounds": rounds,
        "baseline_items": len(base_key),
    }
    out = HERE / "results" / f"checkpoint_results_{date.today().isoformat()}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["rounds"], ensure_ascii=False))
    print(f"断点续跑成功率：{result['resume_success_rate']}（{ok}/{N_ROUNDS}）")
    print(f"结果已存 {out}")


if __name__ == "__main__":
    main()
