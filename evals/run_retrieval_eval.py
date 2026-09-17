"""评测一：检索三档对比（recall@5 / MRR / 平均延迟）。

用法：HF_HUB_OFFLINE=1 python evals/run_retrieval_eval.py
纪律：评测集（retrieval_qa.jsonl）先于本脚本的任何调参存在；
结果写入 evals/results/ 并注明日期，README 数字面板与本文件数字一致。

三档（单变量递进，归因干净）：
  tier1 vector : 纯向量单路
  tier2 hybrid : 向量∥BM25 → RRF（= hybrid_search）
  tier3 +rerank: 档2输出原样进 rerank
"""
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

from rag.retrieve import hybrid_search, rerank, vector_search  # noqa: E402
from rag.store import DSN, load_model  # noqa: E402

HERE = Path(__file__).parent
K = 5


def is_hit(item: dict, hits: list[dict]) -> bool:
    return any(
        h["chunk_id"].startswith(item["gold_standard"]) and h["clause_no"] == item["gold_clause"]
        for h in hits
    )


def reciprocal_rank(item: dict, hits: list[dict]) -> float:
    for rank, h in enumerate(hits, 1):
        if h["chunk_id"].startswith(item["gold_standard"]) and h["clause_no"] == item["gold_clause"]:
            return 1.0 / rank
    return 0.0


def run_tier(tier: str, qa: list[dict], model) -> dict:
    rec_count, rr_sum, total_lat = 0, 0.0, 0.0
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for item in qa:
                t0 = time.perf_counter()
                if tier == "vector":
                    hits = vector_search(cur, model, item["query"], K)
                elif tier == "hybrid":
                    hits = hybrid_search(item["query"], k=K, model=model)
                else:  # rerank
                    merged = hybrid_search(item["query"], k=20, model=model)
                    hits = rerank(item["query"], merged, k=K)
                total_lat += time.perf_counter() - t0
                rec_count += is_hit(item, hits)
                rr_sum += reciprocal_rank(item, hits)
    n = len(qa)
    return {
        "recall@5": round(rec_count / n, 4),
        "mrr": round(rr_sum / n, 4),
        "avg_latency_s": round(total_lat / n, 3),
    }


def main() -> None:
    qa = [json.loads(line) for line in
          (HERE / "retrieval_qa.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"评测集：{len(qa)} 条 QA，指标口径 recall@{K} / MRR / 平均延迟")
    model = load_model()

    results = {}
    for tier in ("vector", "hybrid", "rerank"):
        results[tier] = run_tier(tier, qa, model)
        print(f"{tier:>8}: {results[tier]}")

    out = HERE / "results" / f"retrieval_results_{date.today().isoformat()}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"date": date.today().isoformat(), "n_qa": len(qa), "k": K, "results": results},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已存 {out}")


if __name__ == "__main__":
    main()
