"""retrieve.py：混合检索——向量∥BM25 → RRF → rerank。

按协作模式 AI 逐段代写+用户走读确认：
  第 1 段：检索双通路（向量/BM25）——已确认写入
  第 2 段：RRF 融合（待确认）
  第 3 段：rerank 接入（待确认）
  第 4 段：测试

通路约定：入库和查询必须同套加工——向量路两端都是 BGE-M3 整句编码；
BM25 路两端都是 jieba 分词 + 'simple' 配置。分词器/配置两端不一致，
词就对不上（入库切"上限值"、查询切"上限 值"→ 倒排索引查空）。
"""
import numpy as np
import psycopg

from rag.store import DSN, embed_texts, load_model, tokenize


def vector_search(cur, model, query: str, k: int = 20) -> list[dict]:
    """向量通路：整句编码，按余弦距离排队取前 k。

    不取相似度分数——RRF 只认名次不认分数（两路分数量纲不同不能加，
    "体温36.5度+3个苹果"）。名次由返回顺序隐含表达。"""
    q = np.asarray(embed_texts(model, [query]), dtype=np.float32)[0]
    rows = cur.execute(
        """SELECT chunk_id, clause_no, content
           FROM chunks ORDER BY embedding <=> %s LIMIT %s""",
        (q, k),
    ).fetchall()
    return [dict(chunk_id=r[0], clause_no=r[1], content=r[2]) for r in rows]


def keyword_search(cur, query: str, k: int = 20) -> list[dict]:
    """BM25 通路：查询先 jieba 分词（与入库同切法），词间用 OR 连接。

    OR 不用 AND：AND 要求所有词都在卡里，口语查询极易 0 结果；
    OR 宽进保召回，ts_rank 按命中质量排名。倒排索引（GIN）直接给出
    候选名单——没命中的卡连比都不比。"""
    tsquery = " | ".join(tokenize(query).split())
    rows = cur.execute(
        """SELECT chunk_id, clause_no, content,
                  ts_rank(tsv, q) AS score
           FROM chunks, to_tsquery('simple', %s) q
           WHERE tsv @@ q
           ORDER BY score DESC LIMIT %s""",
        (tsquery, k),
    ).fetchall()
    return [dict(chunk_id=r[0], clause_no=r[1], content=r[2], score=float(r[3]))
            for r in rows]


# ---- 第 2 段：RRF 融合（2026-09-17 用户走读确认后写入）----


def rrf_merge(rankings: list[list[dict]], k: int = 5, smooth: int = 60) -> list[dict]:
    """多路名次倒数求和：贡献 = 1/(smooth+名次)，同卡多路相加。

    两路写同一本账：外层循环轮到第几路，就按该路名次往 scores 打钱；
    同一 chunk_id 被几路打到就自动加几笔（get(id,0) + 贡献）。
    纯函数不碰库，行为由单测钉死。"""
    scores: dict[str, float] = {}
    seen: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):     # 名次从 1 数起
            scores[hit["chunk_id"]] = (
                scores.get(hit["chunk_id"], 0.0) + 1.0 / (smooth + rank)
            )
            seen.setdefault(hit["chunk_id"], hit)          # 记住卡片本体，输出还原
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [{**seen[cid], "rrf": score} for cid, score in ordered]
