"""容器内初始化：建表 + 语料入库（docker-compose 的 ingest 服务入口）。

为什么不用 psql < schema.sql：API 镜像基于 python-slim 不带 psql 客户端，
为一个初始化步骤多装一层不划算；psycopg3 在 autocommit 下单次 execute
可跑多语句，schema.sql 全是 CREATE EXTENSION/TABLE/INDEX（无函数/事务块），
直接整体执行即可。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import psycopg  # noqa: E402

from core.config import DSN  # noqa: E402
from rag.store import ingest  # noqa: E402

SCHEMA = Path(__file__).resolve().parents[1] / "backend" / "rag" / "schema.sql"
CORPUS = Path(__file__).resolve().parents[1] / "corpus"


def main() -> None:
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute(SCHEMA.read_text(encoding="utf-8"))
    print("schema 就绪")

    md_files = sorted(CORPUS.glob("SPGT*.md"))
    n = ingest([str(p) for p in md_files], dsn=DSN)
    print(f"入库完成：{n} 张卡片（{len(md_files)} 份标准）")


if __name__ == "__main__":
    main()
