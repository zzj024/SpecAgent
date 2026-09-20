"""评测七：回归评测——改 Prompt/换模型/调检索参数后防止效果回退。

为什么需要它：前六套评测各自独立跑全套太重（评测二全量 ~5 分钟、
评测三 ~8 分钟），日常改代码需要一套"3 分钟内出结果"的固定子集 +
与基线的自动对比。子集构成（固定，不随改动调整——回归集先于改动存在）：
  retrieval : 评测一 QA 前 20 条，hybrid+rerank 单档 recall@5 / MRR
  review    : 评测二埋错文档前 5 份（rule 轨），漏报 / 误报
  refusal   : A/B/C 各前 5 条 + 不应拒前 10 条，各级正确率 / 误拒率
用法：
  建基线：python evals/run_regression.py --update-baseline
  回归：  python evals/run_regression.py   （与 evals/results/regression_baseline.json 对比）
"""
import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from agents.graph import run_review                     # noqa: E402
from core.config import DSN                             # noqa: E402
from rag.retrieve import hybrid_search, load_model, rerank  # noqa: E402
from reliability.trace_store import TraceStore          # noqa: E402
from run_refusal_eval import (A_QUERIES, B_QUERIES,     # noqa: E402
                              C_DOC, NOT_REFUSE)

HERE = Path(__file__).parent
BASELINE = HERE / "results" / "regression_baseline.json"
K = 5


def _purge(rid: str) -> None:
    s = TraceStore(DSN)
    with s._conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM findings WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM review_steps WHERE review_id = %s", (rid,))
        cur.execute("DELETE FROM reviews WHERE review_id = %s", (rid,))


def _run_items(texts: list[str], tag: str, mode: str = "rule") -> list[list[str]]:
    out = []
    for i, text in enumerate(texts):
        rid = f"rev-reg-{tag}{i:02d}"
        _purge(rid)
        report = run_review(text, document_name=f"reg-{tag}.md", review_id=rid,
                            mode=mode, use_memory=False)
        out.append([f["verdict"] for f in report.get("findings", [])])
    return out


def suite_retrieval(model) -> dict:
    qa = [json.loads(x) for x in
          (HERE / "retrieval_qa.jsonl").read_text(encoding="utf-8").splitlines()
          if x.strip()][:20]
    hits_n, rr = 0, 0.0
    for item in qa:
        hits = rerank(item["query"], hybrid_search(item["query"], k=20, model=model), k=K)
        for rank, h in enumerate(hits, 1):
            if h["chunk_id"].startswith(item["gold_standard"]) and \
               h["clause_no"] == item["gold_clause"]:
                hits_n += 1
                rr += 1.0 / rank
                break
    return {"recall@5": round(hits_n / len(qa), 4), "mrr": round(rr / len(qa), 4)}


def suite_review() -> dict:
    ann = [json.loads(x) for x in
           (HERE / "review_docs" / "annotations.jsonl").read_text(encoding="utf-8")
           .splitlines() if x.strip()]
    by_doc: dict[str, dict[int, bool]] = {}
    for a in ann:
        if a["doc_id"] < "eval2-doc05":
            by_doc.setdefault(a["doc_id"], {})[a["seq"]] = a["violated"]
    tp = fn = fp = tn = 0
    for doc_id, truth in sorted(by_doc.items()):
        rid = f"rev-reg-rev-{doc_id}"
        _purge(rid)
        text = (HERE / "review_docs" / f"{doc_id}.md").read_text(encoding="utf-8")
        report = run_review(text, document_name=f"{doc_id}.md", review_id=rid,
                            mode="rule", use_memory=False)
        findings = {f["item_seq"]: f["verdict"] for f in report.get("findings", [])}
        for seq, violated in truth.items():
            nc = findings.get(seq) == "non_compliant"
            tp, fn, fp, tn = (tp + (violated and nc), fn + (violated and not nc),
                              fp + (not violated and nc), tn + (not violated and not nc))
    return {"miss_rate": round(fn / max(tp + fn, 1), 4),
            "false_alarm_rate": round(fp / max(fp + tn, 1), 4)}


def suite_refusal() -> dict:
    c_verdicts = [v for vs in _run_items([C_DOC] * 1, "c") for v in vs]
    results = {}
    for tag, texts, want in (("A_rate", A_QUERIES[:5], "refusal_A"),
                             ("B_rate", B_QUERIES[:5], "refusal_B")):
        vs = _run_items(texts, tag[0].lower())     # A_QUERIES 已带 [待核] 前缀
        results[tag] = round(sum(v == want for v, in vs) / len(texts), 4)
    c = sum(1 for v in c_verdicts if v == "refusal_C")
    results["C_rate"] = round(c / max(len(c_verdicts), 1), 4)
    nr = _run_items(NOT_REFUSE[:10], "n")
    wrongly = sum(1 for vs in nr for v in vs if v.startswith("refusal"))
    total = sum(len(vs) for vs in nr)
    results["false_refusal_rate"] = round(wrongly / max(total, 1), 4)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = load_model()
    current = {
        "date": date.today().isoformat(),
        "retrieval": suite_retrieval(model),
        "review": suite_review(),
        "refusal": suite_refusal(),
    }
    current["elapsed_min"] = round((time.perf_counter() - t0) / 60, 1)

    if args.update_baseline or not BASELINE.exists():
        BASELINE.write_text(json.dumps(current, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"基线已写入 {BASELINE}")
        print(json.dumps(current, ensure_ascii=False, indent=2))
        return

    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    print(f"回归对比（基线 {base['date']} → 当前 {current['date']}，"
          f"{current['elapsed_min']} 分钟）\n")
    regressions = []
    for section in ("retrieval", "review", "refusal"):
        print(f"[{section}]")
        for key in base[section]:
            old, new = base[section][key], current[section][key]
            delta = round(new - old, 4)
            mark = ""
            if delta < -0.02:                       # 容忍 ±0.02 的子集噪声
                mark = "  ← 回退!"
                regressions.append(f"{section}.{key}: {old} → {new}")
            print(f"  {key:22} {old:8} → {new:8} ({delta:+}){mark}")
    if regressions:
        print("\n检出回退，阻止合并：")
        for r in regressions:
            print("  -", r)
        sys.exit(1)
    print("\n无回退，回归通过。")


if __name__ == "__main__":
    main()
