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

-- =====================================================================
-- M2/M3 表：待审文档 → 审查单 → 轨迹/结论 → 长期记忆
-- 幂等设计三条（对应 L4）：
--   1. review_steps (review_id, step_seq) 唯一键——重放同一步撞键跳过，
--      轨迹写入口唯一（reliability/trace_store.py），幂等不写在 if 里写在键里；
--   2. findings (review_id, item_id) 唯一键——断点续跑重放不产生重复结论
--      （评测六"无重复结论"的数据层保证）；
--   3. memory_entries (domain, attribute, pattern) 唯一键——经验库天然去重，
--      重复经验只涨 hit_count 不涨行数。
-- =====================================================================

CREATE TABLE IF NOT EXISTS documents (
    document_id      TEXT PRIMARY KEY,
    filename         TEXT NOT NULL,
    doc_type         TEXT NOT NULL DEFAULT 'tech_doc',
    parse_confidence REAL NOT NULL DEFAULT 1.0,   -- C 级拒答判据（扫描件/低置信）
    content          TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS check_items (
    item_id          TEXT PRIMARY KEY,            -- "{document_id}:{seq}"
    document_id      TEXT NOT NULL REFERENCES documents(document_id),
    seq              INT  NOT NULL,
    text             TEXT NOT NULL,
    tag              TEXT NOT NULL DEFAULT '',
    attribute        TEXT NOT NULL DEFAULT '',
    value_text       TEXT NOT NULL DEFAULT '',
    parse_confidence REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id     TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(document_id),
    status        TEXT NOT NULL DEFAULT 'pending', -- pending|running|completed|failed
    domain        TEXT NOT NULL DEFAULT '',
    mode          TEXT NOT NULL DEFAULT 'rule',    -- rule|llm
    budget_steps  INT  NOT NULL DEFAULT 200,       -- 终止条件：步数上限
    current_node  TEXT NOT NULL DEFAULT '',        -- 断点续跑游标：最后完成的节点
    error         TEXT NOT NULL DEFAULT '',
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS review_steps (
    id             BIGSERIAL PRIMARY KEY,
    review_id      TEXT NOT NULL REFERENCES reviews(review_id),
    step_seq       INT  NOT NULL,
    agent          TEXT NOT NULL,
    input_summary  TEXT NOT NULL DEFAULT '',
    output_summary TEXT NOT NULL DEFAULT '',
    state_snapshot JSONB NOT NULL DEFAULT '{}',    -- 节点完成后的全量 state（恢复用）
    latency_ms     INT  NOT NULL DEFAULT 0,
    token_est      INT  NOT NULL DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'ok',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (review_id, step_seq)
);
CREATE INDEX IF NOT EXISTS idx_steps_review ON review_steps (review_id, step_seq);

CREATE TABLE IF NOT EXISTS findings (
    finding_id    TEXT PRIMARY KEY,                -- "{review_id}:{item_id}"
    review_id     TEXT NOT NULL REFERENCES reviews(review_id),
    item_id       TEXT NOT NULL REFERENCES check_items(item_id),
    item_seq      INT  NOT NULL DEFAULT 0,
    item_text     TEXT NOT NULL DEFAULT '',
    verdict       TEXT NOT NULL,                   -- core/schemas.py Verdict
    confidence    REAL NOT NULL DEFAULT 0.0,
    chunk_id      TEXT NOT NULL DEFAULT '',
    clause_no     TEXT NOT NULL DEFAULT '',
    quote         TEXT NOT NULL DEFAULT '',
    rationale     TEXT NOT NULL DEFAULT '',
    verify_status TEXT NOT NULL DEFAULT 'n/a',
    UNIQUE (review_id, item_id)
);

CREATE TABLE IF NOT EXISTS memory_entries (
    memory_id  TEXT PRIMARY KEY,
    domain     TEXT NOT NULL,
    attribute  TEXT NOT NULL,
    pattern    TEXT NOT NULL,                      -- 常见不符合项模式描述
    clause_no  TEXT NOT NULL DEFAULT '',
    hit_count  INT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (domain, attribute, pattern)
);
