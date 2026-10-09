"""Run the advisor for every client (or just the ones you pass in).

    python main.py --clients C001 C002 --hitl auto
"""
import argparse, json, logging, sys
from pathlib import Path

from langgraph.types import Command

import graph
from tools import call_tool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S",
                    handlers=[logging.FileHandler("run.log", mode="w"), logging.StreamHandler(sys.stderr)])


def ask_human(payload, mode):
    if mode == "auto":
        return {"action": "approve", "note": "auto-approved (demo)"}
    if mode == "defer":
        return {"action": "defer", "note": "no reviewer available"}
    print(f"\n=== HUMAN REVIEW: {payload['client']} | risk {payload['risk']['score']} | {'; '.join(payload['reasons'])}")
    for i, r in enumerate(payload["recommendations"], 1):
        print(f"  {i}. [{r['priority']}] {r['title']}")
    try:
        c = input("[a]pprove / [m]odify (drop recs) / [r]eject / [d]efer > ").strip().lower()
        if c.startswith("a"):
            return {"action": "approve", "note": input("note: ")}
        if c.startswith("m"):
            drop = [int(x) for x in input("rec numbers to drop (e.g. 6,7): ").split(",") if x.strip().isdigit()]
            return {"action": "modify", "drop": drop, "note": input("note: ")}
        if c.startswith("r"):
            return {"action": "reject", "note": input("reason: ")}
    except EOFError:
        pass
    return {"action": "defer", "note": "deferred"}


def run_client(app, cid, mode):
    cfg = {"configurable": {"thread_id": cid}}                    # each client gets its own thread
    out = app.invoke({"client_id": cid}, cfg)
    while "__interrupt__" in out:                                  # the graph stopped at human_review
        out = app.invoke(Command(resume=ask_human(out["__interrupt__"][0].value, mode)), cfg)
    return out["report"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clients", nargs="*")
    p.add_argument("--hitl", choices=["console", "auto", "defer"], default="console" if sys.stdin.isatty() else "defer")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()

    app, out = graph.build_app(), Path(a.out)
    out.mkdir(exist_ok=True)
    print(f"{'CLIENT':<7}{'STATUS':<25}{'RISK':<11}{'ANOM':<5}{'REVIEW':<13}ADVICE")
    for cid in a.clients or call_tool("list_clients")["data"]:
        try:
            r = run_client(app, cid, a.hitl)
        except Exception as e:                                     # don't let one bad client kill the whole batch
            logging.exception("unhandled error for %s", cid)
            r = {"client_id": cid, "status": "failed", "advice": "NONE", "errors": [str(e)], "anomalies": [], "human_review": {}}
        (out / f"report_{cid}.json").write_text(json.dumps(r, indent=2))
        risk = r.get("risk") or {}
        print(f"{cid:<7}{r['status']:<25}{(risk.get('level', '-') + ' ' + str(risk.get('score', ''))).strip():<11}"
              f"{len(r['anomalies']):<5}{r['human_review'].get('status', '-'):<13}{r['advice']}")
        for e in r.get("errors", []):
            print("       error:", e)
