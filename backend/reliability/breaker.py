"""breaker.py：滑动窗口熔断器（L4）。

为什么熔断在 Agent 系统里必须有：审查单动辄几十个检查项，每项都可能调
LLM；上游故障时不熔断 = 每项都干等重试耗尽，整单卡死。熔断后该检查项
标 degraded 跳过，主流程继续——"局部失败不放大成全局失败"。

三态语义：CLOSED（正常）→ 窗口内连续 threshold 次失败 → OPEN（拒绝）→
冷却期过 → HALF_OPEN（放一个试探请求，成功关闸，失败重新开闸）。
Redis 计数器的进程内版（demo 单机足够；接口不变，换 Redis 只改实现）。
"""
import time
from collections import deque


class CircuitBreaker:
    def __init__(self, fail_threshold: int = 3, window_s: float = 60.0,
                 cooldown_s: float = 30.0):
        self.fail_threshold = fail_threshold
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._events: deque[tuple[float, bool]] = deque()   # (ts, ok)
        self._opened_at: float | None = None

    def allow(self) -> bool:
        """是否放行。OPEN 且未过冷却 → False；过冷却 → True（半开试探）。"""
        if self._opened_at is None:
            return True
        return (time.monotonic() - self._opened_at) >= self.cooldown_s

    def record(self, ok: bool) -> None:
        now = time.monotonic()
        self._events.append((now, ok))
        while self._events and now - self._events[0][0] > self.window_s:
            self._events.popleft()                          # 滑出窗口
        if ok:
            self._events.clear()                            # 成功打断连续失败计数
            if self._opened_at is not None:
                self._opened_at = None                      # 半开试探成功 → 关闸
        else:
            recent_fails = [e for e in self._events if not e[1]]
            if len(recent_fails) >= self.fail_threshold:
                self._opened_at = now                       # 连续失败达标 → 开闸

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        return "open" if not self.allow() else "half_open"
