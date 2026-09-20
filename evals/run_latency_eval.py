"""延迟基准：级联前后端到端耗时对比（数字面板"交互延迟"行的来源）。

三组配置跑同一份文档：
  rule        ：纯比较器轨（无 KEY 环境的默认路径）
  llm-cascade ：级联（快路比较器 + 慢路 LLM，需 KEY）
  llm-all     ：全项慢路（级联前的旧架构，作为对照；如耗时 >15 分钟则跳过大文档）

指标：整单墙钟时间（SSE 场景下用户看到的首条结论远早于整单完成，
但整单时间是最保守的口径，先量它）。
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.graph import run_review                     # noqa: E402
from agents.llm import get_llm                          # noqa: E402
from core.config import DSN                             # noqa: E402
from reliability.trace_store import TraceStore          # noqa: E402

HERE = Path(__file__).parent

DEMO = """# 延迟基准：6 项演示文档

[粗糙度] 阶梯轴配合表面粗糙度 Ra 3.2
[粗糙度] 端面非加工表面粗糙度 Ra 12.5
[比例] 主视图绘图比例 1:2
[余高] 对接焊缝余高 4 mm
[接地] 电气柜保护接地电阻 5 Ω
[涂装] 防火涂料耐火极限 2.0 h
"""

_TEMPLATES = [
    "[粗糙度] 配合表面粗糙度 Ra {v} μm", "[余高] 对接焊缝余高 {v} mm",
    "[接地] 电气柜保护接地电阻 {v} Ω", "[比例] 主视图绘图比例 1:{v}",
    "[咬边] 咬边深度 {v} mm", "[通道] 柜前维修通道宽度 {v} mm",
    "[返修] 同一焊缝部位返修 {v} 次", "[烘干] 低氢型焊条烘干温度 {v} ℃",
    "[探伤] 二级焊缝抽检探伤比例 {v}%", "[标识] PE 保护导体采用绿黄双色",
]


def big_doc(n: int = 100) -> str:
    lines = [f"# 延迟基准：{n} 项批量文档", ""]
    for i in range(n):
        lines.append(_TEMPLATES[i % len(_TEMPLATES)].format(
            v=[1.0, 2.0, 3.0, 2, 0.3, 900, 1, 370, 30, 0][i % 10]))
    return "\n".join(lines)


def _purge(rid: str) -> None:
    s = TraceStore(DSN)
    with s._conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM findings WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM review_steps WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM reviews WHERE review_id = %s", (rid,))


def bench(doc: str, rid: str, mode: str, strategy: str = "cascade") -> float:
    _purge(rid)
    t0 = time.perf_counter()
    report = run_review(doc, document_name="latency.md", review_id=rid,
                        mode=mode, strategy=strategy, use_memory=False)
    dt = time.perf_counter() - t0
    assert "error" not in report, report.get("error")
    return round(dt, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", type=int, default=100, help="批量文档检查项数")
    args = ap.parse_args()
    llm_on = get_llm().available

    results = {}
    results["rule_demo_6"] = bench(DEMO, "rev-lat-rule-demo", "rule")
    results["rule_big"] = bench(big_doc(args.big), "rev-lat-rule-big", "rule")
    if llm_on:
        results["llm_cascade_demo_6"] = bench(DEMO, "rev-lat-llm-demo", "llm")
        results["llm_cascade_big"] = bench(big_doc(args.big), "rev-lat-llm-big", "llm")
        # llm-all 仅跑小文档做对照（大文档全慢路 = 数十分钟，量级已由评测二测得）
        results["llm_all_demo_6"] = bench(DEMO, "rev-lat-llm-all-demo", "llm",
                                          strategy="all")
    out = {
        "date": date.today().isoformat(), "n_items": {"demo": 6, "big": args.big},
        "llm": llm_on, "elapsed_s": results,
    }
    (HERE / "results").mkdir(exist_ok=True)
    path = HERE / "results" / f"latency_results_{date.today().isoformat()}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"结果已存 {path}")


if __name__ == "__main__":
    main()
