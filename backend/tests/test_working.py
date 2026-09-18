"""test_working.py：工作记忆三策略——token 单调性 + compact 摘要保统计事实。"""
from memory.working import WorkingMemory


def _fill(wm, n=10):
    for i in range(n):
        kind = "non_compliant" if i % 3 == 0 else "compliant"
        wm.add(i + 1, f"检查项{i + 1}：粗糙度超差判定为不符合" * 3, kind)


def test_full_keeps_everything():
    wm = WorkingMemory(strategy="full")
    _fill(wm)
    assert "第1项" in wm.context() and "第10项" in wm.context()
    assert wm.dropped == 0


def test_window_forgets_early_items():
    wm = WorkingMemory(strategy="window", window_n=3)
    _fill(wm)
    ctx = wm.context()
    assert "第10项" in ctx and "第1项" not in ctx
    assert wm.dropped >= 7


def test_compact_summary_keeps_stats():
    wm = WorkingMemory(strategy="compact", window_n=3)
    _fill(wm)
    ctx = wm.context()
    assert "[历史摘要]" in ctx                     # 早期项压缩成统计摘要
    assert "non_compliant×" in ctx or "compliant×" in ctx   # 结论计数保留
    assert "第1项" not in ctx                       # 全文复述被压掉
    assert "第10项" in ctx                          # 近窗保明细


def test_compact_saves_tokens_vs_full():
    full, compact = WorkingMemory("full"), WorkingMemory("compact", window_n=3)
    _fill(full)
    _fill(compact)
    assert compact.token_est < full.token_est * 0.8   # 至少省 20%
