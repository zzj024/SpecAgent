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
import re

import numpy as np
import psycopg

from pgvector.psycopg import register_vector  # noqa: E402

from rag.store import DSN, embed_texts, load_model, tokenize

_TSQUERY_SAFE = re.compile(r"^[\w\u4e00-\u9fff]+$")  # 字母/数字/下划线/中文，tsquery 语法外一律剔除


def _to_tsquery_text(query: str) -> str:
    """查询词 → 合法 tsquery 串。剔除 ':' '&' 等保留符号（如 '1:3' 的冒号
    会让 to_tsquery 直接语法报错），全被剔除时返回空串由调用方短路。"""
    words = [w for w in tokenize(query).split() if _TSQUERY_SAFE.match(w)]
    return " | ".join(words)


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
    tsquery = _to_tsquery_text(query)
    if not tsquery:
        return []
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


# ---- 第 3 段：hybrid_search 总装（2026-09-17 用户走读确认后写入）----


def hybrid_search(
    query: str,
    k: int = 5,
    candidate_k: int = 20,
    model=None,
) -> list[dict]:
    """混合检索总装：双通路海选 → RRF 融合 → 切前 k。

    candidate_k=20：宽入围——单路名次靠后但另一路认可的卡要有进场机会；
    每路只取 1 张则融合名存实亡（宽进严出是漏斗的第一原则）。
    model 可外部传入复用：2GB 权重一个进程只读一次。"""
    model = model or load_model()
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            vec = vector_search(cur, model, query, candidate_k)
            kw = keyword_search(cur, query, candidate_k)
    return rrf_merge([vec, kw], k=k)


def hybrid_search_batch(queries: list[str], k: int = 5, candidate_k: int = 20,
                        model=None, dsn: str | None = None,
                        encode_batch: int = 32) -> list[list[dict]]:
    """批量版总装：N 个查询 → N 组候选（级联判定的检索层，2026-09-20）。

    与逐个调 hybrid_search 的三点差异（都是 100 项文档的延迟地板）：
      1. embedding 一次批量算（encode 有固定调度开销，逐条调用白付 N 次）；
      2. 全程一条数据库连接（逐次 connect 的握手成本 ×N 完全可省）；
      3. 只做 RRF 排序、不 rerank——rerank 属于慢路（级联快路不需要精排，
         比较器自己在候选序列里找治理条款）。
    """
    import numpy as np

    from core.config import DSN as _DEFAULT_DSN

    model = model or load_model()
    out: list[list[dict]] = []
    with psycopg.connect(dsn or _DEFAULT_DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            for start in range(0, len(queries), encode_batch):
                chunk = queries[start:start + encode_batch]
                qvecs = np.asarray(embed_texts(model, chunk), dtype=np.float32)
                for q, qv in zip(chunk, qvecs):
                    vec = _vector_search_with_vec(cur, qv, candidate_k)
                    kw = keyword_search(cur, q, candidate_k)
                    out.append(rrf_merge([vec, kw], k=k))
    return out


def _vector_search_with_vec(cur, qv, k: int) -> list[dict]:
    """向量通路（查询向量已算好）：与 vector_search 同构，免重复编码。"""
    rows = cur.execute(
        """SELECT chunk_id, clause_no, content
           FROM chunks ORDER BY embedding <=> %s LIMIT %s""",
        (qv, k),
    ).fetchall()
    return [dict(chunk_id=r[0], clause_no=r[1], content=r[2]) for r in rows]


# ---- 第 4 段：rerank 接入（2026-09-17 用户走读确认后写入）----
from sentence_transformers import CrossEncoder  # noqa: E402

_RERANKER = None


def get_reranker() -> CrossEncoder:
    """懒加载：第一次调用才载入 2.2GB 权重，之后全局复用同一份。"""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _RERANKER


def rerank(query: str, candidates: list[dict], k: int = 5) -> list[dict]:
    """面试：[问题, 卡片正文] 成对批量送入 cross-encoder，按相关分重排。

    predict 一次收全部候选（批处理）；分数存 rerank 字段仅供观察；
    cross-encoder 输入只有文本对，看不到海选的 rrf——RRF 的权力
    止于决定谁入围。独立成函数是为评测一三档对比留单变量归因。"""
    pairs = [(query, c["content"]) for c in candidates]
    scores = get_reranker().predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank"] = float(s)
    return sorted(candidates, key=lambda c: c["rerank"], reverse=True)[:k]
