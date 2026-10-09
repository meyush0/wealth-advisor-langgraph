"""Tool layer. Agents call tools by NAME through call_tool(); they never import the raw functions.
call_tool gives every tool the same behaviour: retry on transient errors, never raise, always log."""
import json, logging, statistics, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

log = logging.getLogger("tools")
DATA = Path(__file__).parent / "data"


class TransientError(Exception):
    """Worth retrying (timeout, 503)."""


# ---------- raw tools ----------
def list_clients():
    return [c["client_id"] for c in json.loads((DATA / "clients.json").read_text())["clients"]]


def load_client(client_id):
    for c in json.loads((DATA / "clients.json").read_text())["clients"]:
        if c["client_id"] == client_id:
            return c
    raise KeyError(f"client {client_id} not found")


_crm_calls = defaultdict(int)


def crm_api(client_id):
    """Mock CRM. data/crm_mock.json -> simulate_failures: N = fail first N calls, 'always' = permanently down."""
    crm = json.loads((DATA / "crm_mock.json").read_text())
    _crm_calls[client_id] += 1
    plan = crm["simulate_failures"].get(client_id)
    if plan == "always" or (isinstance(plan, int) and _crm_calls[client_id] <= plan):
        raise TransientError("CRM timeout / 503")
    if client_id not in crm["records"]:
        raise KeyError("CRM 404")
    return crm["records"][client_id]


def _num(v):
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def clean_client(raw):
    """Validate/normalise raw JSON. Bad rows are dropped and reported, not fatal."""
    issues, txns, holdings, seen = [], [], [], set()
    for h in (raw.get("portfolio") or {}).get("holdings", []):
        v = _num(h.get("value"))
        if v is None or v <= 0:
            issues.append(f"holding {h.get('symbol')}: bad value {h.get('value')!r}, dropped")
            continue
        cls = str(h.get("asset_class", "unknown")).lower()
        holdings.append({**h, "value": v, "asset_class": {"equities": "equity", "bonds": "bond"}.get(cls, cls)})
    for t in raw.get("transactions") or []:
        amt = _num(t.get("amount"))
        kind = t.get("type") or ("debit" if (amt or 0) < 0 else None)
        date = None
        for fmt in ("%Y-%m-%d", "%d-%b-%Y"):
            try:
                date = datetime.strptime(str(t.get("date")), fmt).strftime("%Y-%m-%d")
                break
            except ValueError:
                pass
        if not amt or not date or kind not in ("debit", "credit") or t.get("id") in seen:
            issues.append(f"txn {t.get('id', '?')}: invalid/duplicate row, dropped")
            continue
        seen.add(t.get("id"))
        txns.append({"id": t.get("id", f"auto{len(txns)}"), "date": date, "amount": abs(amt), "type": kind,
                     "category": t.get("category", "other"), "merchant": t.get("merchant", "?")})
    if not raw.get("monthly_income"):
        issues.append("monthly_income missing")
    total = sum(h["value"] for h in holdings)
    target = raw.get("target_allocation") or {}
    if target and abs(sum(target.values()) - 1) > 0.01:
        issues.append(f"target_allocation sums to {sum(target.values()):.2f}, rescaled")
        target = {k: v / sum(target.values()) for k, v in target.items()}
    return {"id": raw["client_id"], "name": raw.get("name", "Unknown"), "income": _num(raw.get("monthly_income")),
            "target": target, "holdings": holdings, "total": total, "txns": txns}, issues


def analyze_portfolio(holdings, target):
    total = sum(h["value"] for h in holdings)
    alloc = defaultdict(float)
    for h in holdings:
        alloc[h["asset_class"]] += h["value"] / total
    singles = [h for h in holdings if not h.get("diversified") and h["asset_class"] != "cash"]
    top = max(singles, key=lambda h: h["value"], default=None)
    drift = {k: round(alloc.get(k, 0) - target.get(k, 0), 3) for k in set(alloc) | set(target)} if target else {}
    return {"allocation": {k: round(v, 3) for k, v in alloc.items()}, "drift": drift,
            "top_position": {"symbol": top["symbol"], "weight": round(top["value"] / total, 3)} if top else None}


def detect_anomalies(txns, income):
    """4 simple, explainable checks: category outlier (z-score), duplicate, risky category, big vs income."""
    out, debits = [], [t for t in txns if t["type"] == "debit"]
    by_cat = defaultdict(list)
    for t in debits:
        by_cat[t["category"]].append(t)
    for cat, rows in by_cat.items():                       # 1. z-score within category
        amts = [r["amount"] for r in rows]
        if len(amts) >= 6 and statistics.pstdev(amts) > 0:
            mu, sd = statistics.mean(amts), statistics.pstdev(amts)
            out += [{"type": "outlier", "severity": "MEDIUM", "txn": r["id"],
                     "detail": f"{r['merchant']} ${r['amount']:,.0f} is {(r['amount'] - mu) / sd:.1f} std-devs above usual '{cat}' spend"}
                    for r in rows if (r["amount"] - mu) / sd > 3.5]
    seen = {}
    for t in sorted(debits, key=lambda t: t["date"]):      # 2. same merchant+amount within 2 days
        k = (t["merchant"], t["amount"])
        if k in seen and (datetime.fromisoformat(t["date"]) - datetime.fromisoformat(seen[k])).days <= 2:
            out.append({"type": "duplicate", "severity": "MEDIUM", "txn": t["id"],
                        "detail": f"possible duplicate charge {t['merchant']} ${t['amount']:,.2f}"})
        seen[k] = t["date"]
    for t in debits:
        share = t["amount"] / income if income else 0
        if t["category"] in ("crypto", "gambling"):        # 3. risky category
            out.append({"type": "risky_category", "severity": "HIGH" if share >= 0.25 else "MEDIUM", "txn": t["id"],
                        "detail": f"{t['category']} spend ${t['amount']:,.0f} ({share:.0%} of monthly income)"})
        elif share >= 0.5:                                 # 4. large vs income
            out.append({"type": "large_vs_income", "severity": "HIGH", "txn": t["id"],
                        "detail": f"{t['merchant']} ${t['amount']:,.0f} = {share:.0%} of monthly income"})
    return out


TOOLS = {f.__name__: f for f in (list_clients, load_client, crm_api, clean_client, analyze_portfolio, detect_anomalies)}


def call_tool(name, retries=0, **kwargs):
    """Uniform wrapper -> {'ok': bool, 'data'|'error'}. Retries TransientError with exponential backoff."""
    err = ""
    for attempt in range(retries + 1):
        try:
            return {"ok": True, "data": TOOLS[name](**kwargs)}
        except TransientError as e:
            err = str(e)
            log.warning("tool=%s transient failure (attempt %d/%d): %s", name, attempt + 1, retries + 1, e)
            time.sleep(0.2 * 2 ** attempt)
        except Exception as e:                              # permanent / unexpected: don't retry, don't crash
            log.error("tool=%s failed: %s", name, e)
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": err}
