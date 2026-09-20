# SpecAgent — 工程标准符合性审查多智能体系统

> 给一份技术文档，五个 Agent 自动完成：识别领域 → 检索适用标准条文 → 逐项核对 →
> 独立校验证据链 → 产出结构化审查报告。**每个结论强制挂「条文原文 + 条款号 + 置信度」
> 证据三元组，证据不足分级拒答而不是硬答**；全程轨迹落库可审计，进程被 kill 后
> 从断点续跑；所有效果数字由评测脚本离线复现。

[![tests](https://img.shields.io/badge/pytest-59%20passed-brightgreen)]()[![python](https://img.shields.io/badge/python-3.11-blue)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()

## 为什么做这个

工程文档审查的现状：标准条文分散在几百页文档里，人工逐条核对慢、易漏，
而"直接拿 LLM 问答"给不出可追责的结论——审查要的是"依据 §4.3.1，Ra 3.2 超过
上限 1.6"，不是一段通顺的废话。**本项目的核心命题：让 Agent 只在有证据时下结论**：

- **证据门控是代码不是提示词**：条款号必须在库里真实存在、引用原文必须逐字是
  条款正文的子串，两道程序化硬校验任一不过，结论强制降级为拒答；
- **审查与校验分离**：初判（reviewer）与证据门控+交叉复核（verifier）是两个
  独立 Agent，自己判自己不算数；
- **拒答分三级**：A 标准库未覆盖 / B 相关但依据不足 / C 输入解析存疑——
  "拒答正确率"是被评测的指标，不是托词。

## 架构

```
┌────────────────────────────────────────────────────┐
│ L0 接入   FastAPI（REST + SSE 流式）  Vue3 免构建前端 │
├────────────────────────────────────────────────────┤
│ L1 编排   LangGraph 状态机（五节点线性图 + 执行器）    │
│   ①路由 → ②检索 → ③审查(Plan-Execute) → ④校验 → ⑤报告 │
├────────────────────────────────────────────────────┤
│ L2 RAG    结构化分块(条款号=主键) → pgvector(HNSW)     │
│           + BM25(jieba+GIN) → RRF → bge-reranker      │
├────────────────────────────────────────────────────┤
│ L3 记忆   工作记忆(滚动窗口+Compact压缩)               │
│           长期记忆(历史不符合项经验库，消融可测)          │
├────────────────────────────────────────────────────┤
│ L4 可靠性  唯一写入口幂等 / PG快照断点续跑 / 重试+熔断   │
├────────────────────────────────────────────────────┤
│ L5 安全   证据门控(程序化) / 拒答分级 / 工具白名单       │
└────────────────────────────────────────────────────┘
  L6 评测（横切）：检索三档 / 埋错法 / 拒答 / 消融 / 压缩 / 断点恢复
```

**双轨判定 + 级联架构**：审查判定是"快路规则 + 慢路 LLM"的级联——

| | 快路（rule 比较器） | 慢路（LLM 升级） |
|---|---|---|
| 触发 | 检测出已知属性且抽得出约束（覆盖域内 ~85-90% 项） | 快路判不动的项：语义模糊 / 无可核对约束 |
| 耗时 | ~1ms（免精排免 LLM） | 精排 + 拒答阈值 + LLM 判定（8 路并发） |
| 证据门控 | 条款存在 + 引用子串 + 交叉复核 | 相同 |
| 离线可用 | 全程无需 API Key | 需 `.env` 配任意 OpenAI 兼容端点 |

- **级联的效果**（`evals/run_latency_eval.py` 可复现）：100 项模板文档 16s（0 次 LLM
  调用）；全项走 LLM 的旧架构同规模要数十分钟——延迟优化的第一问不是"怎么并行"，
  而是"这件事该不该走慢路"（设计复盘见 `docs/问题与解决记录.md` P25）；
- **A/B 拒答只发生在慢路**：判不动的项才有"没覆盖（A）还是依据不足（B）"的定级
  问题——C 级（解析存疑）在检索前，A/B 在慢路，每级在自己掌握证据的那一层拍板；
- **llm 模式**即级联（`strategy=cascade`）；`strategy=all`（全项慢路）保留为延迟
  消融的对照组，评测脚本 `--strategy` 可切换。

## 评测数字（全部脚本可复现，rule 轨、无 LLM API 依赖）

| 指标 | 数字 | 复现命令 |
|------|------|---------|
| 检索 recall@5：纯向量 → +BM25 → +rerank | **1.00 → 0.98 → 1.00**（MRR 0.965/0.957/0.965，n=50） | `evals/run_retrieval_eval.py` |
| 审查漏报率 / 误报率（rule 轨，30 份埋错文档 295 项） | **0% / 0%**（首轮 6.2%/8.2%，两个系统性 bug 修复后清零，归因见 `docs/问题与解决记录.md` P20/P21） | `evals/gen_docs.py && evals/run_review_eval.py` |
| 审查漏报率 / 误报率（llm 级联轨，前 10 份 102 项，MiMo） | **0% / 0%**——级联不仅快，覆盖域内快路比较器比 LLM 更稳 | `run_review_eval.py --mode llm --limit 10` |
| 证据条款命中（不符合结论引用的条款号正确率） | **1.00**（两轨一致） | 同上 |
| 拒答：A 级 / B 级 / C 级正确率 | **100% / 100% / 100%**，不应拒误拒率 **0%**（A/B 边界 = 特征词交集，与快路领域守卫同一口径） | `evals/run_refusal_eval.py` |
| **交互延迟（稳态，级联架构）**：100 项批量 / 6 项演示 | llm 级联 **8.8s / 12.1s**（模板覆盖内 0 次 LLM 调用）；对照：全项走 LLM 慢路 6 项 38~124s | `evals/run_latency_eval.py` |
| 记忆消融（10 文档 × 有/无经验库） | rule 轨两组判定一致（设计使然：记忆注入 rationale 不进判定）；经验库沉淀 14 条 | `evals/run_memory_eval.py` |
| 工作记忆压缩（20 项长单累计 token） | 滑动窗口 **省 57%** / Compact **省 51%**，结论一致率 100% | `evals/run_compression_eval.py` |
| kill -9 断点续跑成功率（5 节点轮转注入 10 次） | **100%**（无重复结论、与基线逐项一致） | `evals/run_checkpoint_eval.py` |
| **评测七·回归**：固定子集（检索 20 QA + 审查 5 文档 + 拒答 25 条） | 基线全绿，改动后 `run_regression.py` 5 分钟出对比，降幅 >2% 退出码 1 | `evals/run_regression.py` |
| pytest | **59 passed**（44 快速单测 + 15 集成，含 kill 续跑端到端） | `pytest && pytest -m integration` |

> **评测口径（诚实声明）**：埋错评测集由固定种子从规则比较器覆盖的属性模板生成，
> 满分说明"rule 轨在其声明的覆盖范围内判定可靠 + 修复后无残留"，**不等于**任意
> 文档满分——超出属性模板的真实文档由级联慢路（LLM）接手，LLM 边界判定的
> 稳定性低于比较器（全项 LLM 对照组漏报 14.3%，其中含缓存混杂，见
> `evals/results/review_results_llm_*.json`）。延迟为稳态口径（模型已加载），
> 冷启动首次请求另付 ~10-20s 模型加载。

> 埋错法评测集由固定种子生成（`evals/gen_docs.py`，seed=42）：
> 30 份文档 × 8~12 检查项，埋 3~6 处已知违规，含 7 份完全合规纯样本。
> 标注文件 `evals/review_docs/annotations.jsonl` 先于跑分存在（先建集后调参）。

## 快速开始

```bash
# 0) 依赖：Python 3.11 + Docker（跑 PostgreSQL）
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"   # Windows
# Linux/macOS: python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# 1) 起数据库 + 建表
docker compose -f docker/docker-compose.dev.yml up -d
cat backend/rag/schema.sql | docker exec -i spec-agent-pg psql -U specagent -d specagent

# 2) 语料入库（三份自写模拟标准，首次会下载 BGE-M3 约 2GB）
HF_HUB_OFFLINE=0 .venv/Scripts/python -m rag.store corpus/SPGT21001-2025_机械制图与尺寸公差.md corpus/SPGT21002-2025_焊接质量验收规范.md corpus/SPGT21003-2025_电气设备标识与安全要求.md

# 3) 起服务（含前端）
.venv/Scripts/uvicorn api.main:app --port 8000
# 打开 http://localhost:8000 —— 粘贴文档 → 实时看五 agent 流转 → 带证据链报告
```

启用 LLM 判定轨（可选）：`cp .env.example .env` 并填入任意 OpenAI 兼容端点的
KEY（DeepSeek / 小米 MiMo 等均已验证），重启服务后 `/reviews` 返回的 `mode`
字段从 `rule` 变为 `llm`——证据门控与拒答纪律两条轨完全一致。

MCP 接入（Claude Code / 任何 MCP 宿主）：

```bash
pip install -e ".[mcp]"
# .mcp.json: {"mcpServers": {"specagent": {"command": ".venv/Scripts/python.exe", "args": ["mcp_server/server.py"]}}}
# 提供三个工具：standard_search / document_review / citation_lookup
```

全容器化：`docker compose -f docker/docker-compose.yml up -d --build`（pg + 语料入库 + API）。

## 待审文档格式

`[标签]` 开头的行是一个检查项，其余是背景说明：

```
# 支架焊接件 + 电气柜 技术要求

[粗糙度] 阶梯轴配合表面粗糙度 Ra 3.2
[余高] 对接焊缝余高 4 mm
[接地] 电气柜保护接地电阻 5 Ω
[涂装] 防火涂料耐火极限 2.0 h        ← 语料未覆盖 → A 级拒答
```

## 关键设计取舍（为什么不用 X）

| 选型 | 为什么 |
|------|--------|
| pgvector（不用 Chroma/FAISS） | 一套 PostgreSQL 同时解决向量 + 全文检索 + 业务表；HNSW 参数可讲调优；两者被视为 demo 级：没处理过 ANN 索引与持久化运维 |
| LangGraph（不用 Chain） | 显式状态机，节点/边/条件路由全可审计；线性图 + 自建 PG 快照，审计轨迹与恢复数据同源 |
| 自建断点续跑（不只靠 checkpointer） | 每节点完成后把全量 state 快照写进 `review_steps`（(review_id, step_seq) 唯一键幂等），kill -9 后从最后完成节点的下一个继续；`SPECAGENT_CRASH_AFTER` 环境变量即崩溃注入点，评测六注入 10 次验证 |
| jieba+GIN（不用 zhparser） | 中文全文检索需分词器，zhparser 要编译扩展，Windows/容器装不了；jieba 纯 Python 零编译 |
| httpx 裸调 LLM（不用 openai SDK） | 只用 /chat/completions + json_object 两个能力，SDK 传递依赖十几个；换 DeepSeek/Qwen 只改环境变量 |
| RRF 融合（不加权分数） | 向量余弦与 BM25 ts_rank 量纲不同不能相加；RRF 只认名次不认分数 |

## 语料版权说明

`corpus/SPGT*.md` 均为**自写模拟标准**（编号虚构前缀 SPG/T），内容不复制任何
真实标准条文；`corpus/metadata/` 只收录真实国标的目录信息（标准号+名称，
公开事实）。待审文档与埋错评测集同样全部生成/自写。

## 项目结构

```
backend/
├── api/          FastAPI：检索 / 审查 / SSE / 静态托管
├── agents/       五节点 + LangGraph 执行器 + 比较器 + LLM 客户端 + 文档解析
├── core/         配置中心 + Pydantic Schema 单一事实源
├── rag/          解析/分块/入库/混合检索/rerank + 全量建表 SQL
├── memory/       长期经验库 + 工作记忆压缩
├── reliability/  幂等轨迹写入口 + 熔断器
├── safety/       程序化证据门控
└── tests/        43 单测 + 15 集成（-m integration）
evals/            七套评测脚本 + 固定种子评测集 + 结果 JSON
web/index.html    Vue3 免构建前端（SSE 实时流转 + 证据溯源）
mcp_server/       MCP 三工具（standard_search / document_review / citation_lookup）
docker/           开发编排（pg）+ 全栈编排 + Dockerfile
docs/             设计决策、防守笔记、问题档案
```

## License

MIT
