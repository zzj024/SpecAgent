"""MCP Server：把 SpecAgent 的三个核心能力封装为 MCP 工具，接入 Claude Code。

工具：
  standard_search(query, k)     混合检索标准条文（返回条款号+原文+分数）
  document_review(doc_text)     待审文档 → 带证据链的符合性审查报告
  citation_lookup(standard_id, clause_no)  条款引用溯源（校验条款存在 + 返回原文）

运行（Claude Code 配置）：
  {"mcpServers": {"specagent": {
      "command": ".venv/Scripts/python.exe",
      "args": ["mcp_server/server.py"]}}}

为什么 MCP 值得做（面试点）：审查能力不锁死在自己的 Web UI 里——
任何 MCP 宿主（Claude Code / 其他 Agent）都能把"查标准、审文档、核引用"
当工具调用；这是 Agent 互操作的标准协议，等于免费获得一个分发渠道。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from mcp.server.mcpserver import MCPServer  # noqa: E402  # mcp 2.x（1.x 叫 FastMCP）

from agents.graph import run_review     # noqa: E402
from core.config import DSN             # noqa: E402
from rag.retrieve import hybrid_search, load_model, rerank  # noqa: E402

mcp = MCPServer("specagent")
_model = None


def _model_or_load():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@mcp.tool()
def standard_search(query: str, k: int = 5) -> str:
    """按自然语言查询检索适用标准条文，返回条款号、原文与相关分。"""
    hits = hybrid_search(query, k=20, model=_model_or_load())
    hits = rerank(query, hits, k=k)
    return "\n\n".join(
        f"[{i}] {h['chunk_id'].split(':')[0]} §{h['clause_no']}"
        f"（相关分 {round(float(h.get('rerank', 0)), 2)}）\n{h['content']}"
        for i, h in enumerate(hits, 1))


@mcp.tool()
def document_review(doc_text: str) -> str:
    """审查一份技术文档（[标签] 行为检查项），返回带证据链的符合性报告。"""
    report = run_review(doc_text, document_name="mcp-input.md", use_memory=False)
    lines = [f"审查单 {report['review_id']}（mode={report['mode']}）："
             f"共 {report['stats']['total']} 项"]
    for f in report.get("findings", []):
        ev = f.get("evidence", {})
        lines.append(
            f"- [{f['verdict']}] {f['item_text']}"
            + (f" ← 依据 §{ev.get('clause_no', '?')}：“{ev.get('quote', '')[:60]}”"
               if ev.get("clause_no") else "")
            + f"（{f['rationale']}）")
    return "\n".join(lines)


@mcp.tool()
def citation_lookup(standard_id: str, clause_no: str) -> str:
    """校验一条条款引用是否真实存在，存在则返回原文（引用溯源）。"""
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(DSN, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT chunk_id, section_path, content FROM chunks
                WHERE standard_id = %s AND clause_no = %s""",
            (standard_id, clause_no))
        rows = cur.fetchall()
    if not rows:
        return f"未找到 {standard_id} §{clause_no}——引用可能为幻觉。"
    return "\n\n".join(
        f"{r['chunk_id']}（{' > '.join(r['section_path'])}）\n{r['content']}"
        for r in rows)


if __name__ == "__main__":
    mcp.run()          # stdio 传输：MCP 宿主通过标准输入输出交互
