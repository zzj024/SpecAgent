"""gen_docs.py：评测二素材——人工埋错法批量生成待审文档。

设计（03-评测方案 · 评测二）：
  - 30 份文档 × 8~12 检查项，每份埋 3~6 处已知不符合（约 1/4 文档完全合规）；
  - 每个检查项从"属性模板池"生成：compliant 值从合规区间随机采样，
    violation 值从违规区间采样——违规一定可由治理条款判定（ground truth 可信）；
  - 标注文件 evals/review_docs/annotations.jsonl 记录 (doc, seq, violated)，
    评测脚本只认标注文件，不重新推导。

种子固定 → 文档集合可复现；生成先于评测存在（评测纪律：先建集后跑分）。
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "review_docs"

# 属性模板池：(tag, 文本模板, 合规值采样, 违规值采样, 治理条款)
# 文本模板里的 {v} 由采样值填充；黑白名单型直接给候选集。
POOL = [
    ("ra",          "[粗糙度] 阶梯轴配合表面粗糙度 Ra {v} μm",
     lambda r: f"{r.uniform(0.4, 1.6):.1f}", lambda r: f"{r.uniform(1.8, 6.4):.1f}"),
    ("ra_raw",      "[粗糙度] 非加工表面粗糙度 Ra {v} μm",
     lambda r: f"{r.uniform(6.4, 25):.1f}", lambda r: f"{r.uniform(26, 60):.1f}"),
    ("scale_ratio", "[比例] 主视图绘图比例 1:{v}",
     lambda r: r.choice([1, 2, 5]), lambda r: r.choice([3, 7])),
    ("reinforce",   "[余高] 对接焊缝余高 {v} mm",
     lambda r: f"{r.uniform(0.5, 3.0):.1f}", lambda r: f"{r.uniform(3.5, 6.0):.1f}"),
    ("undercut_d",  "[咬边] 咬边深度 {v} mm",
     lambda r: f"{r.uniform(0.1, 0.5):.2f}", lambda r: f"{r.uniform(0.6, 1.2):.2f}"),
    ("undercut_l",  "[咬边] 咬边连续长度 {v} mm",
     lambda r: r.randint(10, 100), lambda r: r.randint(110, 200)),
    ("misalign",    "[错边] 板厚 20 mm 对接接头错边量 {v} mm",
     lambda r: f"{r.uniform(0.2, 1.8):.2f}", lambda r: f"{r.uniform(2.3, 3.5):.2f}"),
    ("rt_pct",      "[探伤] 二级焊缝抽检探伤比例 {v}%",
     lambda r: r.randint(20, 100), lambda r: r.randint(5, 19)),
    ("dry_temp",    "[烘干] 低氢型焊条烘干温度 {v} ℃",
     lambda r: r.randint(350, 400), lambda r: r.choice([r.randint(300, 340), r.randint(410, 450)])),
    ("repair",      "[返修] 同一焊缝部位返修 {v} 次",
     lambda r: r.randint(0, 2), lambda r: r.choice([3, 4])),
    ("ground_res",  "[接地] 电气柜保护接地电阻 {v} Ω",
     lambda r: f"{r.uniform(0.5, 4.0):.1f}", lambda r: f"{r.uniform(4.5, 10):.1f}"),
    ("aisle_front", "[通道] 电气柜前维修通道宽度 {v} mm",
     lambda r: r.randint(800, 1400), lambda r: r.randint(500, 790)),
    ("aisle_back",  "[通道] 电气柜后维修通道宽度 {v} mm",
     lambda r: r.randint(600, 1000), lambda r: r.randint(400, 590)),
    ("pe_section",  "[PE] 相线 25 mm²，PE 导体 {v} mm²",
     lambda r: r.choice([16, 20, 25]), lambda r: r.choice([10, 12])),
    ("color_pe",    "[标识] PE 保护导体采用{v}",
     lambda r: "绿黄双色", lambda r: r.choice(["红色", "黄色"])),
    ("live_h",      "[安全] 高度 {v} mm 的裸露带电部分未加装护罩",
     lambda r: r.randint(2300, 2600), lambda r: r.randint(1900, 2290)),
]

N_DOCS = 30
CLEAN_DOCS = 7          # 完全合规文档（测误报率的纯样本）


def gen(seed: int = 42):
    rng = random.Random(seed)
    OUT.mkdir(exist_ok=True)
    annotations = []
    for d in range(N_DOCS):
        doc_id = f"eval2-doc{d:02d}"
        clean = d < CLEAN_DOCS
        n_items = rng.randint(8, 12)
        picks = rng.choices(POOL, k=n_items)
        n_viol = 0 if clean else rng.randint(3, min(6, n_items))
        viol_pos = set(rng.sample(range(n_items), n_viol)) if n_viol else set()

        lines = [f"# 待审技术文档 {doc_id}（评测二语料，生成种子 {seed}）", ""]
        for seq, (attr, tpl, ok_fn, bad_fn) in enumerate(picks):
            violated = seq in viol_pos
            v = bad_fn(rng) if violated else ok_fn(rng)
            lines.append(tpl.format(v=v))
            annotations.append({
                "doc_id": doc_id, "seq": seq, "attr": attr,
                "violated": violated, "text": tpl.format(v=v),
            })
        (OUT / f"{doc_id}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "annotations.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in annotations) + "\n",
        encoding="utf-8")
    n_v = sum(1 for a in annotations if a["violated"])
    print(f"生成 {N_DOCS} 份文档 / {len(annotations)} 检查项 / 埋错 {n_v} 处"
          f"（合规纯样本 {CLEAN_DOCS} 份）→ {OUT}")


if __name__ == "__main__":
    gen()
