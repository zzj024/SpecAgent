"""test_breaker.py：熔断三态——closed → open → half_open → closed。"""
from reliability.breaker import CircuitBreaker


def test_closed_by_default():
    cb = CircuitBreaker(fail_threshold=3, window_s=60, cooldown_s=30)
    assert cb.allow() and cb.state == "closed"


def test_opens_after_consecutive_failures():
    cb = CircuitBreaker(fail_threshold=3, window_s=60, cooldown_s=0.01)
    for _ in range(3):
        cb.record(False)
    assert not cb.allow() and cb.state == "open"


def test_success_resets_streak():
    cb = CircuitBreaker(fail_threshold=3, window_s=60, cooldown_s=30)
    cb.record(False)
    cb.record(False)
    cb.record(True)                                # 未达阈值就被成功打断
    cb.record(False)
    assert cb.allow()                              # 窗口内失败 2+1 次 <3，仍闭合


def test_half_open_then_close_on_success():
    cb = CircuitBreaker(fail_threshold=2, window_s=60, cooldown_s=0.0)
    cb.record(False)
    cb.record(False)
    assert cb.state == "open" or cb.state == "half_open"   # cooldown=0 → 立即可试探
    assert cb.allow()
    cb.record(True)
    assert cb.state == "closed"
