"""评测五：上下文压缩对比——full / window / compact 三策略的 token 开销。

方法：先跑一次 20 检查项的真实审查拿到判定记录（rule 轨），把同一批记录
喂给三种策略的 WorkingMemory（与 reviewer 节点同一调用方式），比较：
  - 累计 context token 估算（每项判定后 context 的 token_est 求和
    = 全程喂给 prompt 的真实开销口径）
  - 结论一致性：rule 轨判定由确定性比较器产生、与工作记忆无关，
    三策略下结论必然一致（100%，构造使然，如实标注）

诚实口径：本评测量的是"机制省多少 token"；"压缩是否损失判定质量"
需要 llm 轨（判定真正消费上下文）才有意义，note 里写明。
"""
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.graph import run_review                     # noqa: E402
from core.config import DSN                             # noqa: E402
from memory.working import WorkingMemory                # noqa: E402
from reliability.trace_store import TraceStore          # noqa: E402

HERE = Path(__file__).parent

DOC = """# 长单审查（评测五 · 20 检查项）

[粗糙度] 配合表面粗糙度 Ra 3.2 μm
[余高] 对接焊缝余高 4.0 mm
[接地] 电气柜保护接地电阻 5.0 Ω
[比例] 主视图绘图比例 1:3
[咬边] 咬边深度 0.9 mm
[通道] 柜前维修通道宽度 600 mm
[返修] 同一焊缝部位返修 4 次
[烘干] 低氢型焊条烘干温度 320 ℃
[探伤] 二级焊缝抽检探伤比例 10%
[标识] PE 保护导体采用红色
[错边] 板厚 20 mm 对接接头错边量 2.8 mm
[安全] 高度 2000 mm 的裸露带电部分未加装护罩
[PE] 相线 25 mm²，PE 导体 10 mm²
[通道] 柜后维修通道宽度 450 mm
[咬边] 咬边连续长度 160 mm
[粗糙度] 非加工表面粗糙度 Ra 40 μm
[余高] 对接焊缝余高 2.0 mm
[接地] 电气柜保护接地电阻 3.0 Ω
[比例] 主视图绘图比例 1:2
[咬边] 咬边深度 0.3 mm
"""


def main() -> None:
    rid = "rev-eval5"
    s = TraceStore(DSN)
    with s._conn() as conn, conn.cursor() as cur:      # 清旧单
        cur.execute("DELETE FROM findings WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM review_steps WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM reviews WHERE review_id = %s", (rid,))

    report = run_review(DOC, document_name="eval5.md", review_id=rid,
                        mode="rule", use_memory=False)
    records = [(f["item_seq"], f"{f['item_text']}→{f['verdict']}", f["verdict"])
               for f in report.get("findings", [])]

    tokens = {}
    for strategy in ("full", "window", "compact"):
        wm = WorkingMemory(strategy=strategy, window_n=5)
        cumulative = 0
        for seq, text, kind in records:
            wm.add(seq, text, kind)
            cumulative += wm.token_est               # 每步判定时的真实上下文开销
        tokens[strategy] = cumulative

    full = tokens["full"]
    result = {
        "date": date.today().isoformat(), "mode": "rule", "n_items": len(records),
        "cumulative_tokens": tokens,
        "window_saving": round(1 - tokens["window"] / full, 4),
        "compact_saving": round(1 - tokens["compact"] / full, 4),
        "verdict_consistency": 1.0,
        "note": "rule 轨判定不消费工作记忆，三策略结论一致是构造使然；"
                "token 差值为机制收益，质量影响需 llm 轨评测",
    }
    out = HERE / "results" / f"compression_results_{date.today().isoformat()}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已存 {out}")


if __name__ == "__main__":
    main()
