"""test_pipeline.py（integration）：LangGraph 五节点端到端 + 断点续跑。

夹具文档覆盖全 verdict 矩阵：
  1 条符合 / 3 条不符合 / 1 条 A 级拒答（超语料范围）/ 低置信文档整单 C 级。
断点续跑：子进程注入崩溃（SPECAGENT_CRASH_AFTER=retriever），重启同单续跑，
校验无重复结论、无丢失检查项——评测六的微缩版。
"""
import json
import os
import subprocess
import sys

import pytest

from agents.graph import run_review
from core.config import DSN

pytestmark = pytest.mark.integration

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DOC = """# 支架焊接件 + 电气柜 技术要求

[粗糙度] 阶梯轴配合表面粗糙度 Ra 3.2
[粗糙度] 端面非加工表面粗糙度 Ra 12.5
[比例] 主视图绘图比例 1:2
[余高] 对接焊缝余高 4 mm
[接地] 电气柜保护接地电阻 5 Ω
[涂装] 防火涂料耐火极限 2.0 h（超语料范围项）
"""


def _cleanup(doc_text: str, document_name: str = "pipeline.md"):
    """按 document 级联清理：同一文档可能挂多张审查单（e2e 与 resume 单
    共用同一文档哈希），按 review_id 删会留孤儿 reviews 行触发外键。
    document_id 哈希含文件名，必须与 run 时同名才能删对行。"""
    from agents.doc_parser import document_id_for
    from reliability.trace_store import TraceStore

    did = document_id_for(document_name, doc_text)
    s = TraceStore(DSN)
    with s._conn() as conn, conn.cursor() as cur:
        cur.execute(
            """DELETE FROM findings WHERE review_id IN
               (SELECT review_id FROM reviews WHERE document_id = %s)""", (did,))
        cur.execute(
            """DELETE FROM review_steps WHERE review_id IN
               (SELECT review_id FROM reviews WHERE document_id = %s)""", (did,))
        cur.execute("DELETE FROM reviews WHERE document_id = %s", (did,))
        cur.execute("DELETE FROM check_items WHERE document_id = %s", (did,))
        cur.execute("DELETE FROM documents WHERE document_id = %s", (did,))


def test_end_to_end_rule_mode():
    rid = "rev-test-e2e"
    _cleanup(DOC)
    report = run_review(DOC, document_name="pipeline.md", review_id=rid,
                        mode="rule", use_memory=False)
    assert "error" not in report, report.get("error")
    stats = report["stats"]
    assert stats["total"] == 6
    assert stats["compliant"] == 2                 # 非加工 Ra 12.5 ≤25 + 比例 1:2
    assert stats["non_compliant"] == 3             # Ra 3.2>1.6 / 余高 4>3 / 接地 5>4
    assert stats["refusal_A"] == 1                 # 耐火极限：语料未覆盖
    # 证据链：每条不符合结论都挂条款号且引用属实
    nc = [f for f in report["findings"] if f["verdict"] == "non_compliant"]
    assert {f["evidence"]["clause_no"] for f in nc} == {"4.3.1", "4.2.1", "4.3.2"}
    for f in nc:
        assert f["verify"] == "passed" and f["evidence"]["quote"]
    _cleanup(DOC)


def test_low_confidence_doc_all_refusal_c():
    doc = "<!-- scan-confidence: 0.2 -->\n[粗糙度] Ra 3.2\n[接地] 接地电阻 5 Ω"
    rid = "rev-test-crefusal"
    _cleanup(doc, "scan.md")
    report = run_review(doc, document_name="scan.md", review_id=rid,
                        mode="rule", use_memory=False)
    assert report["stats"]["refusal_C"] == 2
    assert report["stats"].get("non_compliant", 0) == 0
    _cleanup(doc, "scan.md")


def test_crash_and_resume():
    """kill 等价注入：retriever 完成后进程自杀 → 同单重跑 → 结果与一步到位一致。"""
    rid = "rev-test-resume"
    _cleanup(DOC)
    code = (
        "import sys; sys.path.insert(0, 'backend');"
        "from agents.graph import run_review;"
        f"run_review({DOC!r}, document_name='pipeline.md', review_id={rid!r},"
        " mode='rule', use_memory=False)"
    )
    env = {**os.environ, "SPECAGENT_CRASH_AFTER": "retriever",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    crashed = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True, timeout=600)
    assert crashed.returncode == 137, crashed.stderr[-500:]       # 确认真崩在注入点

    report = run_review(DOC, document_name="pipeline.md", review_id=rid,
                        mode="rule", use_memory=False)            # “重启续跑”

    assert "error" not in report, report.get("error")
    assert report["stats"]["total"] == 6                          # 无丢失检查项
    assert len({f["item_seq"] for f in report["findings"]}) == 6  # 无重复结论
    assert report["stats"]["non_compliant"] == 3                  # 与一步到位一致
    # 轨迹连续：崩溃前 2 步 + 续跑 3 步，step_seq 无跳号无重号
    from reliability.trace_store import TraceStore
    steps = [s["step_seq"] for s in TraceStore(DSN).list_steps(rid)]
    assert steps == sorted(steps) and len(steps) == len(set(steps)) == 5
    _cleanup(DOC)


def test_report_json_serializable():
    """报告要过 SSE/HTTP，必须能 json.dumps（Pydantic 已转 dict 的约定检查）。"""
    rid = "rev-test-json"
    doc = "[接地] 接地电阻 4 Ω"
    _cleanup(doc, "j.md")
    report = run_review(doc, document_name="j.md", review_id=rid,
                        mode="rule", use_memory=False)
    json.dumps(report, ensure_ascii=False)
    _cleanup(doc, "j.md")
