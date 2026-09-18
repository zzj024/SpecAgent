"""working.py：工作记忆——滚动窗口 + Compact 摘要压缩（L3，评测五的机制）。

问题：逐项审查到第 N 项时，把前 N-1 项的完整判定历史塞进 prompt，
token 随项数线性膨胀，长单既贵又触发上下文上限；但完全丢历史又会
丢掉"前一项刚判过同一属性"的上下文。

三种策略（评测五的实验组）：
  full    ：全量历史（基线，质量上限对照）
  window  ：只保留最近 n 条（省 token，丢早期上下文）
  compact ：窗口外的不丢——按结论类别聚成统计摘要
            （"历史第1-7项：non_compliant×2，refusal_A×1"），
            窗口内保明细。压的是"每条的全文复述"，保的是
            "到目前为止什么结论出现过多少次"这一统计事实。

add() 带 kind（结论类别）：compact 摘要靠它计数，不需要解析文本。
token 估算用 chars/2（中文约 2 字符 ≈ 1 token，无分词器的轻量近似，
绝对值不精确但组间可比——评测五比的是差值不是绝对值）。
"""
from dataclasses import dataclass, field


def est_tokens(text: str) -> int:
    return max(1, len(text) // 2)


@dataclass
class WorkingMemory:
    strategy: str = "full"          # full | window | compact
    window_n: int = 5
    records: list[dict] = field(default_factory=list)   # {"seq","text","kind"}
    _counts: dict = field(default_factory=dict)         # 被逐出记录的结论计数
    _span: tuple[int, int] = (0, 0)                     # 被逐出记录的 seq 范围

    def add(self, seq: int, text: str, kind: str = "") -> None:
        self.records.append({"seq": seq, "text": text, "kind": kind})
        if self.strategy in ("window", "compact") and len(self.records) > self.window_n:
            evicted = self.records.pop(0)
            if self.strategy == "compact":              # window 逐出即丢
                lo, hi = self._span
                self._span = (evicted["seq"] if lo == 0 else lo, evicted["seq"])
                self._counts[evicted["kind"] or "unknown"] = \
                    self._counts.get(evicted["kind"] or "unknown", 0) + 1

    def context(self) -> str:
        """拼给 prompt 的工作记忆。compact = 统计摘要 + 近窗明细。"""
        if self.strategy == "full":
            return "\n".join(f"第{r['seq']}项：{r['text']}" for r in self.records)
        recent = self.records[-self.window_n:]
        tail = "\n".join(f"第{r['seq']}项：{r['text']}" for r in recent)
        if self.strategy == "window":
            return tail
        head = ""
        if self._counts:
            stats = "，".join(f"{k}×{v}" for k, v in sorted(self._counts.items()))
            head = f"[历史摘要] 第{self._span[0]}~{self._span[1]}项：{stats}"
        return "\n".join(x for x in (head, tail) if x)

    @property
    def token_est(self) -> int:
        return est_tokens(self.context())

    @property
    def dropped(self) -> int:
        """被滑出窗口丢掉明细的记录数（window 丢事实，compact 只丢细节）。"""
        if self.strategy == "full":
            return 0
        if self.strategy == "compact":
            return sum(self._counts.values())
        # window：连续 seq 场景下，逐出数 = 最大 seq - 窗口内条数
        return max(0, self.records[-1]["seq"] - len(self.records)) if self.records else 0
