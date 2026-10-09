r"""The LangGraph workflow.

The supervisor looks at what's already in the state and picks the next worker. Every worker
writes its result into the state and goes back to the supervisor, until we reach finalize.

    START -> supervisor <-> fetcher / analyzer / advisor / human_review / narrator
                  \-> finalize -> END
"""
import logging
from operator import add
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from llm import narrate
from memory import LongTermMemory
from tools import call_tool

log = logging.getLogger("graph")
memory = LongTermMemory()                       # main.py can change memory.path
EQUITY_CAP = {"conservative": 0.40, "moderate": 0.70, "aggressive": 0.90}
DEFAULT_CRM = {"risk_tolerance": "moderate", "kyc_status": "unknown", "assumed_defaults": True}


class State(TypedDict, total=False):
    client_id: str
    client: dict
    issues: list
    crm: dict
    crm_source: str                              # live | cache | default
    analysis: dict
    anomalies: list
    risk: dict
    recommendations: list
    review: dict
    status: str                                  # failed | (unset = ok)
    narrative: str                               # LLM summary, empty if skipped or failed
    report: dict
    next: str
    errors: Annotated[list, add]                 # these lists keep growing as nodes add to them
    warnings: Annotated[list, add]
    trace: Annotated[list, add]


# supervisor: decides who runs next
def supervisor(state: State):
    if state.get("status") == "failed":
        nxt = "finalize"
    elif "client" not in state:
        nxt = "fetcher"
    elif "analysis" not in state:
        nxt = "analyzer"
    elif "recommendations" not in state:
        nxt = "advisor"
    elif "review" not in state:
        nxt = "human_review"
    elif "narrative" not in state:
        nxt = "narrator"
    else:
        nxt = "finalize"
    log.info("[%s] supervisor -> %s", state["client_id"], nxt)
    return {"next": nxt, "trace": [f"supervisor -> {nxt}"]}


# fetcher: load and clean the client data, then get the CRM profile
def fetcher(state: State):
    cid = state["client_id"]
    raw = call_tool("load_client", client_id=cid)
    if not raw["ok"]:
        return {"status": "failed", "errors": [raw["error"]], "trace": ["fetcher: load failed"]}
    cleaned = call_tool("clean_client", raw=raw["data"])
    if not cleaned["ok"]:
        return {"status": "failed", "errors": [cleaned["error"]], "trace": ["fetcher: clean failed"]}
    client, issues = cleaned["data"]
    if not client["holdings"] and not client["txns"]:
        return {"status": "failed", "errors": ["no usable holdings or transactions"], "issues": issues,
                "trace": ["fetcher: nothing to analyse"]}

    warnings = []
    crm = call_tool("crm_api", retries=2, client_id=cid)          # try CRM, then cache, then defaults
    if crm["ok"]:
        profile, source = crm["data"], "live"
        memory.save_crm(cid, profile)
    elif memory.get_crm(cid):
        profile, source = memory.get_crm(cid), "cache"
        warnings.append(f"CRM down ({crm['error']}); using cached profile")
    else:
        profile, source = DEFAULT_CRM, "default"
        warnings.append(f"CRM down ({crm['error']}); no cache, using default profile - suitability unverified")
    log.info("[%s] fetched: %d holdings, %d txns, crm=%s, %d data issues", cid, len(client["holdings"]),
             len(client["txns"]), source, len(issues))
    return {"client": client, "issues": issues, "crm": profile, "crm_source": source, "warnings": warnings,
            "trace": [f"fetcher: ok (crm={source}, issues={len(issues)})"]}


# analyzer: portfolio numbers, anomalies and the risk score
def analyzer(state: State):
    c, crm, warnings = state["client"], state["crm"], []
    analysis, anomalies = {}, []
    if c["holdings"]:
        r = call_tool("analyze_portfolio", holdings=c["holdings"], target=c["target"])
        analysis = r["data"] if r["ok"] else {}
        if not r["ok"]:
            warnings.append(f"portfolio analysis failed: {r['error']}")
    else:
        warnings.append("no holdings: portfolio analysis skipped")
    if c["txns"]:
        r = call_tool("detect_anomalies", txns=c["txns"], income=c["income"])
        anomalies = r["data"] if r["ok"] else []
        if not r["ok"]:
            warnings.append(f"anomaly detection failed: {r['error']}")
    else:
        warnings.append("no transactions: anomaly detection skipped")

    # risk score out of 100, made of three parts so we can always say why it is what it is
    top = (analysis.get("top_position") or {}).get("weight", 0)
    equity = analysis.get("allocation", {}).get("equity", 0)
    cap = EQUITY_CAP.get(crm.get("risk_tolerance"), 0.70)
    parts = {"concentration": min(30, max(0, (top - 0.10) / 0.40 * 30)),
             "profile_mismatch": min(30, max(0, (equity - cap) / 0.30 * 30)),
             "anomalies": min(40, sum(12 if a["severity"] == "HIGH" else 5 for a in anomalies))}
    score = round(sum(parts.values()))
    prev = memory.last_run(c["id"])
    risk = {"score": score, "level": "HIGH" if score >= 60 else "MEDIUM" if score >= 30 else "LOW",
            "parts": {k: round(v, 1) for k, v in parts.items()},
            "previous_score": prev["score"] if prev else None}
    return {"analysis": analysis, "anomalies": anomalies, "risk": risk, "warnings": warnings,
            "trace": [f"analyzer: risk={score} ({risk['level']}), anomalies={len(anomalies)}"]}


# advisor: turns the analysis into a prioritised list of actions
def advisor(state: State):
    a, crm, recs = state["analysis"], state["crm"], []

    def rec(priority, title, why):
        recs.append({"priority": priority, "title": title, "why": why})

    high = [x for x in state["anomalies"] if x["severity"] == "HIGH"]
    med = [x for x in state["anomalies"] if x["severity"] == "MEDIUM"]
    if high:
        rec("HIGH", "Verify high-severity transactions with client", "; ".join(x["detail"] for x in high))
    if med:
        rec("MEDIUM", f"Review {len(med)} unusual transaction(s)", "; ".join(x["detail"] for x in med))
    top = a.get("top_position")
    if top and top["weight"] > 0.20:
        rec("HIGH" if top["weight"] > 0.40 else "MEDIUM", f"Reduce concentration in {top['symbol']} ({top['weight']:.0%})",
            "single-name exposure above 20% guideline")
    drift = max((abs(v) for v in a.get("drift", {}).values()), default=0)
    if drift > 0.05:
        rec("HIGH" if drift > 0.15 else "MEDIUM", f"Rebalance toward target (max drift {drift:.0%})", "allocation drifted from target")
    equity, tol = a.get("allocation", {}).get("equity", 0), crm.get("risk_tolerance", "moderate")
    if equity > EQUITY_CAP.get(tol, 0.70):
        rec("HIGH", f"Equity {equity:.0%} exceeds '{tol}' risk tolerance", "suitability check required")
    if crm.get("kyc_status") not in ("valid", "unknown"):
        rec("HIGH", f"KYC is '{crm['kyc_status']}'", "refresh KYC before executing any advice")
    if state["crm_source"] == "default":
        rec("LOW", "Confirm risk profile with client", "CRM unavailable; defaults assumed")
    if not recs:
        rec("LOW", "No action required", "no issues detected")
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recs.sort(key=lambda r: order[r["priority"]])
    return {"recommendations": recs, "trace": [f"advisor: {len(recs)} recommendations"]}


# human review: pause here if something needs a person to look at it
def human_review(state: State):
    reasons = []
    if state["risk"]["score"] >= 60:
        reasons.append(f"risk score {state['risk']['score']} >= 60")
    if any(a["severity"] == "HIGH" for a in state["anomalies"]):
        reasons.append("high-severity anomaly")
    if state["crm_source"] == "default":
        reasons.append("advice based on assumed default profile")
    if any("exceeds" in r["title"] or r["title"].startswith("KYC") for r in state["recommendations"]):
        reasons.append("suitability/compliance issue")
    if not reasons:
        return {"review": {"status": "not_required"}, "trace": ["review: not required"]}

    # graph stops here until main.py resumes it with the reviewer's decision
    decision = interrupt({"client": state["client"]["name"], "reasons": reasons, "risk": state["risk"],
                          "recommendations": state["recommendations"]})
    action = decision.get("action", "defer")
    recs = state["recommendations"]
    review = {"reasons": reasons, "note": decision.get("note", ""), "status": {"approve": "approved", "modify": "modified",
              "reject": "rejected"}.get(action, "pending")}
    if action == "modify":
        drop = set(decision.get("drop", []))
        review["dropped"] = [r["title"] for i, r in enumerate(recs, 1) if i in drop]
        recs = [r for i, r in enumerate(recs, 1) if i not in drop]
    elif action == "reject":
        review["dropped"] = [r["title"] for r in recs]
        recs = []
    return {"review": review, "recommendations": recs, "trace": [f"review: {review['status']}"]}


# narrator: asks the LLM to write a short summary. It only describes the result, it doesn't decide anything
def narrator(state: State):
    review = state["review"]
    if review.get("status") in ("pending", "rejected"):          # advice is on hold, nothing to summarise
        return {"narrative": "", "trace": ["narrator: skipped (advice held)"]}
    facts = {"client": state["client"]["name"], "risk": state["risk"], "anomalies": state["anomalies"],
             "recommendations": state["recommendations"], "review": review, "crm_source": state["crm_source"],
             "data_issues": state["issues"]}
    try:
        text = narrate(facts)
        return {"narrative": text, "trace": ["narrator: ok"]}
    except Exception as e:                                       # no key or LLM is down: the report still goes out without a summary
        log.warning("[%s] narrator failed: %s", state["client_id"], e)
        return {"narrative": "", "warnings": [f"LLM summary unavailable: {e}"], "trace": ["narrator: failed"]}


# finalize: build the report and remember this run
def finalize(state: State):
    failed = state.get("status") == "failed"
    review = state.get("review", {})
    held = review.get("status") in ("pending", "rejected")
    report = {"client_id": state["client_id"], "client": (state.get("client") or {}).get("name"),
              "status": "failed" if failed else ("completed_with_warnings" if state.get("warnings") or held else "completed"),
              "advice": "NONE" if failed else "HELD_FOR_REVIEW" if held else "RELEASED",
              "risk": state.get("risk"), "anomalies": state.get("anomalies", []),
              "recommendations": state.get("recommendations", []), "human_review": review,
              "summary": state.get("narrative", ""),
              "data_issues": state.get("issues", []), "crm_source": state.get("crm_source"),
              "warnings": state.get("warnings", []), "errors": state.get("errors", []), "trace": state.get("trace", [])}
    if not failed:
        memory.save_run(state["client_id"], {"score": state["risk"]["score"], "level": state["risk"]["level"],
                                             "anomalies": len(state["anomalies"]), "review": review.get("status")})
    log.info("[%s] done: status=%s advice=%s", state["client_id"], report["status"], report["advice"])
    return {"report": report}


def build_app():
    g = StateGraph(State)
    for name, fn in [("supervisor", supervisor), ("fetcher", fetcher), ("analyzer", analyzer), ("advisor", advisor),
                     ("human_review", human_review), ("narrator", narrator), ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"],
                            ["fetcher", "analyzer", "advisor", "human_review", "narrator", "finalize"])
    for worker in ("fetcher", "analyzer", "advisor", "human_review", "narrator"):
        g.add_edge(worker, "supervisor")                # every worker goes back to the supervisor
    g.add_edge("finalize", END)
    return g.compile(checkpointer=MemorySaver())        # interrupt() needs a checkpointer
