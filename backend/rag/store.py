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
