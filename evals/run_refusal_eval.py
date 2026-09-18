"""评测三：多级拒答正确率——应拒样本拒对等级 / 不应拒样本误拒。

评测集（refusal_cases.jsonl，先于跑分存在）：
  A 级应拒 ×10：超出语料领域的问题（防火/装修/绩效…），期望 refusal_A；
  B 级应拒 ×10：命中文库但抽不出可核对约束的提问（术语解释类），期望 refusal_B；
  C 级应拒 ×10：低解析置信度文档（scan-confidence 标记），期望 refusal_C；
  不应拒 ×30：正常可判定检查项（取自评测二合规池），期望 compliant。

指标：各级正确拒答率（拒对等级才算对）、不应拒样本误拒率。
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.graph import run_review                     # noqa: E402
from core.config import DSN                             # noqa: E402
from reliability.trace_store import TraceStore          # noqa: E402

HERE = Path(__file__).parent

A_QUERIES = [
    "[待核] 防火涂料耐火极限 2.0 h", "[待核] 办公室装修风格采用后现代",
    "[待核] 员工绩效考核分五档", "[待核] 食品添加剂苯甲酸钠限量 1 g/kg",
    "[待核] 软件界面主色调用深蓝", "[待核] 消防通道宽度 1.2 m",
    "[待核] 包装盒印刷色差 ΔE≤3", "[待核] 物流三天达承诺",
    "[待核] 广告投放 ROI 3.5", "[待核] 工位照度 500 lx",
]
B_QUERIES = [
    "[术语] 什么是轮廓算术平均偏差 Ra", "[术语] 解释公差等级的定义",
    "[术语] 什么是焊缝余高", "[术语] 咬边的定义是什么", "[术语] 解释保护导体的含义",
    "[术语] 什么是错边量", "[术语] 解释安全距离的概念", "[术语] 什么是无损检测",
    "[术语] 基本尺寸的定义", "[术语] 什么是过盈配合",
]
C_DOC = "<!-- scan-confidence: 0.2 -->\n[粗糙度] 配合表面粗糙度 Ra 3.2\n[接地] 接地电阻 5 Ω"
NOT_REFUSE = [
    "[粗糙度] 配合表面粗糙度 Ra 1.2 μm", "[余高] 对接焊缝余高 2.0 mm",
    "[接地] 电气柜保护接地电阻 3.0 Ω", "[比例] 主视图绘图比例 1:2",
    "[咬边] 咬边深度 0.3 mm", "[通道] 柜前维修通道宽度 900 mm",
    "[返修] 同一焊缝部位返修 1 次", "[烘干] 低氢型焊条烘干温度 370 ℃",
    "[探伤] 二级焊缝抽检探伤比例 30%", "[标识] PE 保护导体采用绿黄双色",
    "[粗糙度] 配合表面粗糙度 Ra 3.2 μm", "[余高] 对接焊缝余高 4.5 mm",
    "[接地] 电气柜保护接地电阻 6.0 Ω", "[比例] 主视图绘图比例 1:7",
    "[咬边] 咬边深度 0.9 mm", "[通道] 柜前维修通道宽度 600 mm",
    "[返修] 同一焊缝部位返修 4 次", "[烘干] 低氢型焊条烘干温度 320 ℃",
    "[探伤] 二级焊缝抽检探伤比例 10%", "[标识] PE 保护导体采用红色",
    "[错边] 板厚 20 mm 对接接头错边量 1.0 mm", "[安全] 高度 2500 mm 的裸露带电部分未加装护罩",
    "[PE] 相线 25 mm²，PE 导体 20 mm²", "[PE] 相线 25 mm²，PE 导体 10 mm²",
    "[安全] 高度 2000 mm 的裸露带电部分未加装护罩", "[通道] 柜后维修通道宽度 700 mm",
    "[咬边] 咬边连续长度 80 mm", "[粗糙度] 非加工表面粗糙度 Ra 20 μm",
    "[错边] 板厚 20 mm 对接接头错边量 2.8 mm", "[咬边] 咬边连续长度 160 mm",
]


def _purge(rid: str) -> None:
    s = TraceStore(DSN)
    with s._conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM findings WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM review_steps WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM reviews WHERE review_id = %s", (rid,))


def _run(items_text: str, rid: str) -> list[str]:
    _purge(rid)
    report = run_review(items_text, document_name=f"{rid}.md", review_id=rid,
                        mode="rule", use_memory=False)
    return [f["verdict"] for f in report.get("findings", [])]


def main() -> None:
    cases = []                                     # (期望等级, 检查项文本)
    cases += [("refusal_A", q) for q in A_QUERIES]
    cases += [("refusal_B", q) for q in B_QUERIES]
    cases += [("not_refuse", q) for q in NOT_REFUSE]

    t0 = time.perf_counter()
    per_case = []
    for i, (expect, text) in enumerate(cases):
        verdicts = _run(text, f"rev-eval3-item{i:02d}")
        verdict = verdicts[0] if verdicts else "missing"
        per_case.append({"expect": expect, "text": text[:40], "verdict": verdict})

    c_verdicts = _run(C_DOC, "rev-eval3-cdoc")     # C 级：一份低置信文档 ×2 项 ×5 次
    for _ in range(5):                             # 10 条 C 级样本
        per_case += [{"expect": "refusal_C", "text": "低置信扫描件项",
                      "verdict": v} for v in c_verdicts]

    def rate(expect: str, want: str | None = None) -> tuple[float, int]:
        sel = [c for c in per_case if c["expect"] == expect]
        ok = [c for c in sel if c["verdict"] == (want or expect)]
        return (round(len(ok) / len(sel), 4) if sel else 0.0), len(sel)

    a_rate, a_n = rate("refusal_A")
    b_rate, b_n = rate("refusal_B")
    c_rate, c_n = rate("refusal_C")
    # 误拒率：不应拒样本被判成任何 refusal_* 的比例
    nr = [c for c in per_case if c["expect"] == "not_refuse"]
    wrongly = [c for c in nr if c["verdict"].startswith("refusal")]
    fr_rate = round(len(wrongly) / len(nr), 4) if nr else 0.0

    result = {
        "date": date.today().isoformat(), "mode": "rule",
        "A_rate": a_rate, "B_rate": b_rate, "C_rate": c_rate,
        "false_refusal_rate": fr_rate,
        "n": {"A": a_n, "B": b_n, "C": c_n, "not_refuse": len(nr)},
        "wrongly_refused": [c["text"] for c in wrongly],
        "B_details": [c for c in per_case if c["expect"] == "refusal_B"],
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    out = HERE / "results" / f"refusal_results_{date.today().isoformat()}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("B_details", "wrongly_refused")},
                     ensure_ascii=False, indent=2))
    if wrongly:
        print("误拒样本：", result["wrongly_refused"])
    print(f"结果已存 {out}")


if __name__ == "__main__":
    main()
