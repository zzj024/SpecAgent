"""store.py：把卡片灌入 PostgreSQL——parse → chunk → embed → INSERT。

按协作模式 AI 逐段代写+用户走读确认：
  第 2 段：embedding 与分词两把工具——已确认写入
  第 3 段：ingest 主函数 + 幂等插入（待确认）
"""
import jieba
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-m3"


def load_model() -> SentenceTransformer:
    """加载本地模型（首次读 2GB 权重后有缓存，之后秒级）。"""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """一批文字 → 一批 1024 维向量。

    1. 一次传整批不循环单条：每次计算调用都有固定开销（数据搬运/算子调度），
       批处理把开销摊到所有条目上，快一个数量级；
    2. normalize=True：归一化不改方向 → 余弦结果不变；真正价值是让
       内积=余弦，流水线里任何一处用内积计算都不会被长向量带偏。"""
    return model.encode(texts, normalize_embeddings=True).tolist()


def tokenize(text: str) -> str:
    """正文 → jieba 词串（空格分隔），喂给 chunks.tokens 列。

    PG 全文检索不认识中文（无分词器，整句变一个词），先切词 BM25 通路才生效。"""
    return " ".join(jieba.lcut(text))


# ---- 第 3 段：ingest 主函数（2026-09-17 用户走读确认后写入）----

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

from rag.chunking import chunk_standard  # noqa: E402
from rag.parsing import parse_markdown  # noqa: E402

DSN = "postgresql://specagent:specagent@localhost:5432/specagent"


def ingest(md_paths: list[str], dsn: str = DSN) -> int:
    """语料 md 列表 → 库。返回处理的卡片总数。四步：
    ①解析分块 ②登记 standards ③批量向量化 ④幂等插入。

    幂等：chunk_id / standard_id 都是业务主键，重跑全撞键 DO NOTHING，
    库里永远一份。外键要求先插 standards 再插 chunks（顺序反了全拒）。"""
    docs = [parse_markdown(open(p, encoding="utf-8").read()) for p in md_paths]
    all_chunks = [c for d in docs for c in chunk_standard(d)]
    if not all_chunks:
        return 0

    model = load_model()
    embeddings = np.asarray(
        embed_texts(model, [c.text for c in all_chunks]), dtype=np.float32
    )

    with psycopg.connect(dsn) as conn:
        register_vector(conn)          # 教 psycopg 认识 vector 类型
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO standards (standard_id, title) VALUES (%s, %s)"
                " ON CONFLICT (standard_id) DO NOTHING",
                [(d.standard_id, d.title) for d in docs],
            )
            cur.executemany(
                """INSERT INTO chunks
                       (chunk_id, standard_id, clause_no, section_path,
                        content, tokens, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (chunk_id) DO NOTHING""",
                [
                    (c.chunk_id, c.standard_id, c.clause_no, c.section_path,
                     c.text, tokenize(c.text), e)
                    for c, e in zip(all_chunks, embeddings)
                ],
            )
    return len(all_chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description="语料入库：md → chunks 表")
    ap.add_argument("paths", nargs="+", help="语料 md 路径")
    n = ingest(ap.parse_args().paths)
    print(f"入库处理完成：{n} 张卡片")


if __name__ == "__main__":
    main()
