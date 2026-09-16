-- backend/rag/schema.sql：标准库两张核心表（M1 范围）
-- 执行：cat backend/rag/schema.sql | docker exec -i spec-agent-pg psql -U specagent -d specagent
--
-- 设计要点（对应已能讲清单）：
-- 1. chunks 主键用业务 ID 而非自增序号：重灌同卡撞主键被拒 = 幂等的数据层落点；
--    自增主键挡不住重复数据，top-5 会被同一张卡占两席、挤出真正相关的条款。
-- 2. vector(1024) 定长才能建 HNSW；1024 = BGE-M3 输出维度（已实测验证）。
-- 3. tokens 列：PG 全文检索无中文分词器，jieba 先切词再建 tsv 生成列，
--    BM25 通路才对中文生效。
-- 4. 三个索引各服务一路查询：HNSW=向量 / GIN=全文(BM25) / B-tree=条款号溯源。

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS standards (
    standard_id TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'simulated'   -- simulated | public_catalog
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    standard_id  TEXT NOT NULL REFERENCES standards(standard_id),
    clause_no    TEXT NOT NULL,
    section_path TEXT[] NOT NULL DEFAULT '{}',
    content      TEXT NOT NULL,
    tokens       TEXT NOT NULL DEFAULT '',
    kind         TEXT NOT NULL DEFAULT 'clause',
    embedding    vector(1024) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('simple', tokens)) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunks_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_chunks_clause ON chunks (standard_id, clause_no);
