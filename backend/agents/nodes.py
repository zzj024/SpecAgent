"""nodes.py：五个 Agent 节点——LangGraph 图的节点函数（L1 编排层）。

五节点流水线（对应 02-技术方案 §2）：
  router    快速路由：关键词投票定领域；解析置信度低 → 全单直落 C 级拒答
  retriever 检索 Agent：为每个检查项混合检索+重排，产候选条款（不判定）
  reviewer  审查 Agent（Plan-and-Execute）：逐项 A 级拒答判定 → 比对 → 初判
  verifier  校验 Agent：程序化证据门控（独立于产生结论的调用）+ 交叉复核
  writer    报告 Agent：结构化报告 + 结论落库 + 经验沉淀

多级拒答的判定责任划分（面试常被问"在哪判"）：
  C 级：router（文档级置信度，检索之前就能判，省一次检索）
  A 级：reviewer（检索之后——top 候选相关分低于阈值 = 标准库未覆盖）
  B 级：verifier（门控不过 / 双轨不一致 = 相关但依据不足）
每级只在自己掌握证据的那一层拍板，不越权。

rule/llm 双轨：mode 来自 state（executor 按 LLMClient.available 决定）。
llm 轨的判定仍要过 verifier 的程序化门控——LLM 说的条款号/引用必须
在库里真实存在，这条纪律两条轨一视同仁。
"""
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

from agents.compare import ATTRIBUTES, detect_attribute, judge_item
from agents.llm import LLMUnavailable, get_llm
from agents.state import ReviewState
from core.config import (
    BREAKER_FAIL_THRESHOLD, BREAKER_WINDOW_S, DSN, LLM_CONCURRENCY,
    REFUSE_A_SCORE, REFUSE_C_CONF, RETRIEVAL_K, CANDIDATE_K,
)
from core.schemas import CheckItem, Evidence, Finding
from memory.long_term import MemoryStore
from memory.working import WorkingMemory
from rag.retrieve import hybrid_search_batch, load_model, rerank
from rag.store import tokenize
from reliability.breaker import CircuitBreaker
from safety.evidence_gate import verify_evidence

# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


_MODEL = None


def _get_model():
    """embedding 模型进程级单例：2GB 权重只读一次（api/main.py 同款策略）。"""
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


_DOMAIN_KEYS = {
    "mechanical": ["粗糙度", "比例", "平面度", "公差", "幅面", "图幅", "制图", "Ra"],
    "welding": ["焊接", "焊缝", "咬边", "余高", "错边", "探伤", "射线", "焊条", "返修", "烘干"],
    "electrical": ["电气", "接地", "导体", "柜", "带电", "相序", "触电", "铭牌", "PE"],
}


def router_node(state: ReviewState) -> dict:
    """路由 Agent：关键词投票定领域（快速路由，无需 LLM）。

    低置信文档在此整单标 C——解析存疑时后续任何"符合/不符合"都不可信，
    最诚实的输出就是请人工复核，而不是带病判定。"""
    items: list[CheckItem] = state.get("items", [])
    votes: dict[str, int] = {}
    for it in items:
        for dom, keys in _DOMAIN_KEYS.items():
            if any(k in it.text or (it.tag and k in it.tag) for k in keys):
                votes[dom] = votes.get(dom, 0) + 1
    domain = max(votes, key=votes.get) if votes else "general"

    if state.get("doc_parse_confidence", 1.0) < REFUSE_C_CONF:
        c_refusals = [
            Finding(item=it, verdict="refusal_C", confidence=state["doc_parse_confidence"],
                    rationale="文档解析置信度低（疑似扫描件模糊），全部检查项转人工复核。")
            for it in items
        ]
        return {"domain": domain, "findings": c_refusals,
                "events": [{"node": "router", "detail": f"domain={domain}, 低置信→C级拒答 x{len(c_refusals)}"}]}

    return {"domain": domain,
            "events": [{"node": "router", "detail": f"domain={domain}, items={len(items)}"}]}


def _expand_query(item: CheckItem) -> str:
    """rule 版 Query Rewrite：属性锚词扩写。

    短数值型查询（"相线 25 mm²，PE 导体 10 mm²"）与条文的词汇重叠太小，
    rerank 分会卡在拒答线下（实测 0.583 < 0.60）——把该属性在条文里的
    锚词（"保护导体 截面积"）并进查询，与条文共享词汇，分数回到 0.72+。
    这是"快速路由优先，低置信 Query Rewrite 兜底"的检索级落点；
    llm 轨可换 LLM 改写，本函数即 fallback。"""
    from agents.compare import ATTRIBUTES, detect_attribute

    attr = item.attribute or detect_attribute(item.text)
    keys = ATTRIBUTES.get(attr, {}).get("clause_keys", [])
    return f"{item.text} {' '.join(keys)}".strip()


def _rerank_scored(query: str, hits: list[dict], k: int = RETRIEVAL_K) -> list[dict]:
    """rerank + sigmoid 归一（慢路专用；快路比较器不需要精排）。"""
    rk = rerank(query, hits, k=k)
    for h in rk:
        h["score"] = round(_sigmoid(h.get("rerank", 0.0)), 4)
    return rk


def retriever_node(state: ReviewState) -> dict:
    """检索 Agent：批量混合检索（RRF 序候选池，不 rerank）。

    级联架构下的分工（2026-09-20）：本节点只产"够比较器用"的候选池——
    一次批量 embedding + 单连接跑完全部查询（100 项的检索地板从 ~50s
    压到秒级）；精排（rerank）挪到慢路——只有比较器判不动的项才值得
    花 7~8s 的 CPU 精排，判得动的项 RRF 序已足够（比较器按锚词过滤后
    逐卡试判，见 compare.judge_item）。

    只检索不判定——检索质量的可归因性要求这一步的行为完全由检索配置
    决定（评测一的三档），混进判定逻辑就没法单独归因了。"""
    if state.get("findings") and state["findings"][0].verdict == "refusal_C":
        return {"candidates": []}                    # C 级单已整单拒答，免检索

    items = state.get("items", [])
    if not items:
        return {"candidates": []}
    queries = [f"{it.tag} {it.text}".strip() for it in items]
    pools = hybrid_search_batch(queries, k=CANDIDATE_K, model=_get_model())
    per_item = [{"seq": it.seq, "hits": hits} for it, hits in zip(items, pools)]
    return {"candidates": per_item,
            "events": [{"node": "retriever",
                        "detail": f"{len(per_item)} 项批量检索完成（RRF 池）"}]}


_LLM_SYSTEM = (
    "你是工程标准符合性审查员。给定检查项与候选标准条文，判定该检查项"
    "符合(compliant)/不符合(non_compliant)/依据不足(insufficient)。"
    "规则：只能依据给出的条文判定；引用必须逐字摘自所选条文正文；"
    '不确定时选 insufficient，禁止臆测。只输出 JSON：'
    '{"verdict":"compliant|non_compliant|insufficient",'
    '"clause_no":"...", "quote":"...", "rationale":"..."}'
)


def _llm_judge(item: CheckItem, hits: list[dict], context: str,
               hints: list[dict], breaker: CircuitBreaker) -> Finding:
    """llm 慢路单项判定（可并发调用）：候选条文 + 工作记忆快照 + 经验提示。

    context 是调用前串行抓取的快照——并发阶段只做网络往返，不碰共享的
    WorkingMemory（读序确定性：prompt 内容与串行版完全一致）。"""
    llm = get_llm()
    if not breaker.allow():
        return Finding(item=item, verdict="degraded", confidence=0.0,
                       rationale="LLM 服务熔断中，该检查项标记服务降级，请稍后重审。")
    clauses = "\n".join(
        f"[{i+1}] {h['clause_no']}（{h['chunk_id'].split(':')[0]}）：{h['content']}"
        for i, h in enumerate(hits))
    user = (f"检查项：{item.text}\n\n候选条文：\n{clauses}\n\n"
            + (f"此前审查记录（工作记忆）：\n{context}\n\n" if context else "")
            + (f"历史经验提示（该类属性常见不符合模式，供参考，判定仍以条文为准）：\n"
               f"{'；'.join(h['pattern'] for h in hints)}\n\n" if hints else "")
            + "输出 JSON。")
    try:
        out = llm.chat_json(_LLM_SYSTEM, user)
        breaker.record(True)
    except LLMUnavailable as e:
        breaker.record(False)
        return Finding(item=item, verdict="degraded", confidence=0.0,
                       rationale=f"LLM 调用失败（{e}），标记服务降级。")

    verdict = out.get("verdict", "insufficient")
    # 条款号归一化：LLM 会夹带空白/§/句读（肉眼不可见，SQL 等值匹配会挂），
    # 不清洗的话证据门控 gate1 把正确结论误杀成 refusal_B（评测二 llm 轨
    # 2 处漏报的根因——判对了，死在格式上）
    clause_no = re.sub(r"[\s§。；;，,]", "", str(out.get("clause_no", "")))
    hit = next((h for h in hits if h["clause_no"] == clause_no), hits[0] if hits else None)
    ev = Evidence(chunk_id=hit["chunk_id"] if hit else "",
                  standard_id=hit["chunk_id"].split(":")[0] if hit else "",
                  clause_no=clause_no,
                  quote=out.get("quote", ""),
                  score=hit.get("score", 0.0) if hit else 0.0)
    return Finding(
        item=item,
        verdict="refusal_B" if verdict == "insufficient" else verdict,
        confidence=0.8,
        evidence=ev,
        rationale=out.get("rationale", ""),
    )


def _finding_from_judgment(it: CheckItem, j, hits: list[dict], hint_txt: str) -> Finding:
    """快路产物：比较器 Judgment → Finding（证据三元组在此挂上）。"""
    chunk_id = next((h["chunk_id"] for h in hits if h["clause_no"] == j.clause_no),
                    hits[0]["chunk_id"] if hits else "")
    return Finding(
        item=it, verdict=j.verdict, confidence=j.confidence,
        evidence=Evidence(chunk_id=chunk_id, standard_id=chunk_id.split(":")[0],
                          clause_no=j.clause_no, quote=j.quote, score=j.evidence_score),
        rationale=j.rationale + (f"（经验提示：{hint_txt}）" if hint_txt else ""),
    )


def reviewer_node(state: ReviewState) -> dict:
    """审查 Agent：级联判定——快路 rule 比较器，慢路 rerank+阈值+LLM。

    为什么级联（2026-09-20 延迟优化的架构答案，不是调参）：
      全项问 LLM = 法院所有案子都走合议庭。比较器判"接地电阻 5 Ω"只要
      1ms，LLM 要一次 10s 网络往返——快路判得动的不该进慢路。
      拒答语义也随之更干净：A/B 只在慢路上判（判不动才需要问
      "是没覆盖还是依据不足"），C 级在检索前（router）。

    慢路三阶段：循环里串行做"精排→阈值→A/B 定级→抓工作记忆快照"，
    然后 ThreadPoolExecutor 并发调 LLM（LLM_CONCURRENCY 路），最后串行
    收尾写工作记忆——并发只覆盖网络往返，共享状态全在串行段碰。"""
    findings = state.get("findings", [])
    if findings and findings[0].verdict == "refusal_C":
        return {"verified": findings}                 # 透传给 writer

    mode = state.get("mode", "rule")
    strategy = state.get("strategy", "cascade")       # cascade | all（llm 轨消融用）
    force_slow = mode == "llm" and strategy == "all"
    by_seq = {c["seq"]: c["hits"] for c in state.get("candidates", [])}
    mem = MemoryStore(DSN)
    use_memory = state.get("use_memory", True)
    domain = state.get("domain", "general")
    wm = WorkingMemory(strategy=state.get("wm_strategy", "full"))
    breaker = CircuitBreaker(BREAKER_FAIL_THRESHOLD, BREAKER_WINDOW_S)
    out: list[Finding] = []
    slow: list[tuple] = []                            # (item, 精排候选, hints, 记忆快照)
    n_fast = 0

    for it in state.get("items", []):
        hits = by_seq.get(it.seq, [])
        hints = mem.search(domain, it.attribute, limit=2) if use_memory else []
        hint_txt = "；".join(h["pattern"] for h in hints) or ""

        # ---- 快路：比较器直接在 RRF 候选池上判（~1ms，免精排免 LLM）----
        # 门禁：只对"检测出已知属性"的项开放——attr=None 时比较器的锚词
        # 过滤形同虚设，通用数值判定会抓任何带"不得大于"的卡硬判
        # （"防火涂料耐火极限"就这样被判成 non_compliant）。未知属性
        # 一律慢路：精排分数 + 词面交集判 A/B，宁拒答不硬判。
        attr_known = detect_attribute(it.text) is not None
        j = judge_item(it.text, hits) if (hits and attr_known) else None
        if j is not None and not force_slow:
            out.append(_finding_from_judgment(it, j, hits, hint_txt))
            wm.add(it.seq, f"{it.text}→{j.verdict}", j.verdict)
            n_fast += 1
            continue

        if not hits:
            out.append(Finding(item=it, verdict="refusal_A", confidence=1.0,
                               rationale="检索无任何候选，标准库未覆盖该项。"))
            wm.add(it.seq, f"{it.text}→refusal_A", "refusal_A")
            continue

        # ---- 慢路准备：精排（含 Query Rewrite 兜底）→ 阈值 → A/B 定级 ----
        base_q = f"{it.tag} {it.text}".strip()
        rk = _rerank_scored(base_q, hits)
        if (not rk or rk[0]["score"] < REFUSE_A_SCORE) and _expand_query(it) != base_q:
            # 重写真的加了锚词才值得重排一次——attr=None 时扩写=原查询，
            # 重跑 7s 的 CPU 精排是纯浪费
            rk2 = _rerank_scored(_expand_query(it), hits)
            if rk2 and (not rk or rk2[0]["score"] > rk[0]["score"]):
                rk = rk2
        top = rk[0]["score"] if rk else 0.0
        if not rk or top < REFUSE_A_SCORE:
            related = _weak_related(it.text, rk)
            verdict = "refusal_B" if related else "refusal_A"
            out.append(Finding(
                item=it, verdict=verdict, confidence=1.0 - top,
                rationale=(f"检索 top 相关分 {top}（阈值 {REFUSE_A_SCORE}），"
                           + ("有擦边相关条文但证据不足以判定。" if related
                              else "标准库未覆盖该项。")),
            ))
            wm.add(it.seq, f"{it.text}→{verdict}", verdict)
            continue

        if mode == "llm":
            slow.append((it, rk, hints, wm.context()))    # 此刻的记忆快照进 prompt
        else:                                             # rule 轨：相关但抽不出约束
            out.append(Finding(item=it, verdict="refusal_B", confidence=0.5,
                               rationale="检索到相关条文，但未抽出可核对的数值/类别约束，依据不足。"))
            wm.add(it.seq, f"{it.text}→refusal_B", "refusal_B")

    # ---- 慢路执行：并发只包网络往返，收尾串行 ----
    if slow:
        with ThreadPoolExecutor(max_workers=LLM_CONCURRENCY) as ex:
            futures = [ex.submit(_llm_judge, it, rk, ctx, hints, breaker)
                       for it, rk, hints, ctx in slow]
            llm_results = [f.result() for f in futures]
        for (it, _rk, _hints, _ctx), f in zip(slow, llm_results):
            out.append(f)
            wm.add(it.seq, f"{it.text}→{f.verdict}", f.verdict)

    out.sort(key=lambda f: f.item.seq)
    return {"findings": out, "wm_token_est": wm.token_est,
            "events": [{"node": "reviewer",
                        "detail": f"级联：快路 {n_fast} / 慢路 {len(slow)}；{_summarize(out)}"}]}


def _weak_related(item_text: str, hits: list[dict]) -> bool:
    """A/B 拒答的边界判据：top1 候选与检查项有无 ≥2 字词的词汇交集。

    分数低于拒答线时还要分两种死法：
      词汇零交集（"防火涂料耐火极限" vs 粗糙度条文）→ A：真没覆盖；
      词汇有交集（"公差等级的定义" vs 3.3 术语条款）→ B：擦边相关，
      但证据不足以支撑结论。同一分数线，落在带内哪一侧靠词面定级。"""
    if not hits:
        return False
    toks = [t for t in tokenize(item_text).split() if len(t) >= 2]
    return any(t in hits[0]["content"] for t in toks)


def verifier_node(state: ReviewState) -> dict:
    """校验 Agent：证据门控（代码级，独立于产生结论的 LLM/规则调用）。

    三查：①条款真实存在 ②引用逐字属实（safety.evidence_gate）
          ③结论与证据一致——rule 轨由比较器交叉复核（双轨一致才放行），
          llm 轨 rule 比对作为第二意见，不一致降 B 级。
    门控不过 → 强制 refusal_B：宁可拒答不可硬答。"""
    mode = state.get("mode", "rule")
    by_seq = {c["seq"]: c["hits"] for c in state.get("candidates", [])}
    verified: list[Finding] = []

    for f in state.get("findings", []):
        if f.verdict in ("compliant", "non_compliant"):
            gate = verify_evidence(DSN, f.evidence.chunk_id,
                                   f.evidence.clause_no, f.evidence.quote)
            if not gate.passed:
                verified.append(Finding(
                    item=f.item, verdict="refusal_B", confidence=0.9,
                    rationale=f"证据门控未通过（{gate.reason}），强制降级为拒答。",
                ))
                continue

            # 第 3 查：rule 比较器交叉复核（llm 轨的第二意见；rule 轨的自校验）
            hits = by_seq.get(f.item.seq, [])
            cross = judge_item(f.item.text, hits) if hits else None
            if cross is None or cross.verdict != f.verdict:
                verified.append(Finding(
                    item=f.item, verdict="refusal_B", confidence=0.7,
                    rationale="独立复核与初判结论不一致（依据不足以支撑该结论），降级拒答。",
                    evidence=f.evidence, verify="rejected",
                ))
                continue
            verified.append(f.model_copy(update={"verify": "passed"}))
        else:
            verified.append(f)                        # 拒答/降级类不经门控

    n_blocked = sum(1 for a, b in zip(state.get("findings", []), verified)
                    if b.verdict == "refusal_B" and a.verdict != "refusal_B")
    return {"verified": verified,
            "events": [{"node": "verifier", "detail": f"门控拦截 {n_blocked} 条"}]}


def writer_node(state: ReviewState) -> dict:
    """报告 Agent：结构化报告 + 结论幂等落库 + 经验沉淀（writer 是唯一
    写 findings 的节点，与轨迹唯一写入口 TraceStore 配套）。

    findings 扁平化（item_seq/item_text/verdict/evidence/...）：与库表行
    同构——API、评测、断点恢复读到的都是同一形状，不需要两层转换。"""
    findings = state.get("verified") or state.get("findings") or []
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.verdict] = counts.get(f.verdict, 0) + 1
    report = {
        "review_id": state.get("review_id", ""),
        "document_id": state.get("document_id", ""),
        "document_name": state.get("document_name", ""),
        "domain": state.get("domain", ""),
        "mode": state.get("mode", "rule"),
        "stats": {
            "total": len(findings), **counts,
            "non_compliant_rate": round(
                counts.get("non_compliant", 0) / len(findings), 4) if findings else 0.0,
        },
        "findings": [{
            "item_seq": f.item.seq,
            "item_text": f.item.text,
            "verdict": f.verdict,
            "confidence": f.confidence,
            "evidence": f.evidence.model_dump(),
            "rationale": f.rationale,
            "verify": f.verify,
        } for f in findings],
        "memory_used": state.get("memory_used", 0),
    }

    # 经验沉淀：不符合项 → 长期记忆（评测四消融的 treatment 组来源）
    if state.get("use_memory", True):
        mem = MemoryStore(DSN)
        for f in findings:
            if f.verdict == "non_compliant":
                mem.upsert(state.get("domain", ""), f.item.attribute or "unknown",
                           f"{f.item.text}（依据 {f.evidence.clause_no}）",
                           f.evidence.clause_no)

    return {"report": report,
            "events": [{"node": "writer", "detail": f"报告完成 {report['stats']}"}]}


def _summarize(findings: list[Finding]) -> str:
    counts = {}
    for f in findings:
        counts[f.verdict] = counts.get(f.verdict, 0) + 1
    return ", ".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "无检查项"


NODE_FUNCS = {
    "router": router_node, "retriever": retriever_node, "reviewer": reviewer_node,
    "verifier": verifier_node, "writer": writer_node,
}
NODE_ORDER = ["router", "retriever", "reviewer", "verifier", "writer"]

_ = time  # noqa: F841 —— 计时在 executor 层做，这里不重复计时
