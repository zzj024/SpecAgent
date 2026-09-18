"""core/config.py：全局配置——环境变量优先，默认值本地裸机可跑。

配置集中一处（此前 DSN 散落在 rag/store.py）：换环境（CI / 容器 / 评委机器）
只改环境变量不动代码。所有阈值都挂环境变量，评测七回归时改参数即改配置。

LLM 约定（重要）：DEEPSEEK_API_KEY 未设置时系统整体进入**确定性规则模式**
（见 agents/llm.py 的 LLMClient.available）——审查流水线、评测全流程照常跑通，
只是"支持性判定"由程序化比较器完成而不是 LLM。这保证：
  1) pytest / 评测数字离线可复现（不依赖外部 API 的波动与计费）；
  2) 拿到 KEY 即自动升级为 LLM 判定，代码路径不变。
"""
import os

# 数据库
DSN = os.getenv(
    "SPECAGENT_DSN",
    "postgresql://specagent:specagent@localhost:5432/specagent",
)

# LLM 主模型（DeepSeek，OpenAI 兼容 /chat/completions 协议，httpx 直连不引 SDK）
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
LLM_BASE_URL = os.getenv("SPECAGENT_LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("SPECAGENT_LLM_MODEL", "deepseek-chat")
LLM_TIMEOUT_S = float(os.getenv("SPECAGENT_LLM_TIMEOUT_S", "30"))
LLM_MAX_RETRY = int(os.getenv("SPECAGENT_LLM_MAX_RETRY", "3"))

# 检索
RERANK_ON = os.getenv("SPECAGENT_RERANK_ON", "1") == "1"
RETRIEVAL_K = int(os.getenv("SPECAGENT_RETRIEVAL_K", "5"))
CANDIDATE_K = int(os.getenv("SPECAGENT_CANDIDATE_K", "20"))

# 多级拒答阈值
# A 级（标准库未覆盖）：top1 候选的 rerank 相关分（sigmoid 归一到 0~1）低于此线
# → "标准库未覆盖该项"。校准实测（bge-reranker-v2-m3，2026-09-18）：
# 域内查询 top1 ∈ [0.70, 0.73]，超语料查询 top1 ∈ [0.50, 0.52]，
# 0.55~0.70 是空带——取空带中点 0.60，两侧各留 ~0.1 余量
REFUSE_A_SCORE = float(os.getenv("SPECAGENT_REFUSE_A_SCORE", "0.60"))
# C 级（输入解析存疑）：文档或检查项解析置信度低于此线
REFUSE_C_CONF = float(os.getenv("SPECAGENT_REFUSE_C_CONF", "0.5"))

# 可靠性
BREAKER_FAIL_THRESHOLD = int(os.getenv("SPECAGENT_BREAKER_THRESHOLD", "3"))
BREAKER_WINDOW_S = float(os.getenv("SPECAGENT_BREAKER_WINDOW_S", "60"))
