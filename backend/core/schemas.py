"""core/schemas.py：全项目 Pydantic Schema 单一事实源。

报告 Schema 在这里约束（需求 P0-8"结构化报告"），agents 产出、API 返回、
评测脚本消费的是同一套模型——三处手写字段名迟早对不齐，一处定义永不跑偏。

verdict 取值即多级拒答的落点：
  compliant / non_compliant : 有证据链的结论（挂 Evidence 三元组）
  refusal_A                 : 标准库未覆盖（检索零相关）
  refusal_B                 : 有相关条文但证据不支持结论
  refusal_C                 : 输入解析存疑（置信度低，请人工复核）
  degraded                  : 依赖服务熔断降级，该检查项跳过（主流程不中断）
"""
from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal[
    "compliant", "non_compliant", "refusal_A", "refusal_B", "refusal_C", "degraded",
]
VerifyStatus = Literal["passed", "rejected", "gate_blocked", "n/a"]


class Evidence(BaseModel):
    """证据三元组：条文原文 + 出处条款号 + 检索相关分。结论必须挂在它上面。"""

    chunk_id: str = ""
    standard_id: str = ""
    clause_no: str = ""
    quote: str = ""
    score: float = 0.0


class CheckItem(BaseModel):
    """检查项：从待审文档解析出的一个可核对断言（一个带值的属性）。"""

    seq: int
    text: str
    tag: str = ""                      # 文档里方括号标签，如 [粗糙度]
    attribute: str = ""                # 归一化属性名（ra / weld_reinforce / ...）
    value_text: str = ""               # 原文值表达，如 "3.2" / "绿黄双色"
    parse_confidence: float = 1.0


class Finding(BaseModel):
    """一条结论：检查项 × 判定 × 证据。校验 Agent 未放行前 verify 不是 passed。"""

    item: CheckItem
    verdict: Verdict
    confidence: float = 0.0
    evidence: Evidence = Field(default_factory=Evidence)
    rationale: str = ""
    verify: VerifyStatus = "n/a"


class ReviewReport(BaseModel):
    """结构化审查报告（Pydantic 约束，writer 节点唯一产出形态）。"""

    review_id: str
    document_id: str
    document_name: str = ""
    domain: str = ""
    mode: Literal["llm", "rule"] = "rule"
    findings: list[Finding] = Field(default_factory=list)
    stats: dict = Field(default_factory=dict)
    memory_used: int = 0
    elapsed_s: float = 0.0
    created_at: str = ""
