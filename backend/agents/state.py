"""state.py：LangGraph 全局状态——五个 agent 节点共享的唯一黑板。

写入权约定（线性图，每字段单一写入者，覆盖语义）：
  findings  router（C 级整单拒答）或 reviewer（正常初判）写——二者互斥，
            verifier 不改 findings、只产 verified，避免结论列表双份。
  verified  verifier 独占写：门控放行的、降级拒答的、透传的，全在这。
  events    多节点都写 → append reducer：LangGraph 在 super-step 边界合并，
            谁发的过程事件都不丢（SSE / 审计消费）。
"""
from typing import Annotated, TypedDict

from core.schemas import CheckItem, Finding


def append_list(a: list | None, b: list | None) -> list:
    return list(a or []) + list(b or [])


class ReviewState(TypedDict, total=False):
    """total=False：断点续跑从 DB 快照恢复时，字段可以不全。"""

    # 输入（router 之前就有）
    review_id: str
    document_id: str
    document_name: str
    doc_text: str
    doc_parse_confidence: float          # 文档级解析置信度（C 级拒答判据）
    mode: str                            # rule | llm
    strategy: str                        # cascade（级联，默认）| all（llm 轨全项慢路，消融用）
    use_memory: bool                     # 评测四消融开关
    wm_strategy: str                     # full | window | compact（评测五）

    # router 产出
    domain: str                          # mechanical | welding | electrical | general
    items: list[CheckItem]

    # retriever 产出
    candidates: list[dict]               # [{"seq", "hits": [卡...]}] 按 items 序

    # reviewer 产出（C 级单由 router 直接写 findings）
    findings: list[Finding]
    wm_token_est: int                    # 工作记忆 token 估算（评测五观察值）

    # verifier 产出
    verified: list[Finding]

    # 横切
    events: Annotated[list[dict], append_list]   # 节点过程事件（SSE / 审计）
    error: str
    report: dict                         # writer 产出
