# Wealth Advisor Assistant

A small multi-agent project built with LangGraph. It reads a client's financial data, pulls their profile from a
(mock) CRM, checks the portfolio and the transactions for problems, and writes up a report. If something looks risky,
it stops and waits for a human to look at it before any advice goes out.

## Running it
```bash
pip install -r requirements.txt
export GROQ_API_KEY=...                  # optional, only needed for the written summary
python main.py                           # all clients (asks you to review on a terminal)
python main.py --clients C001 C002 C999  # just these ones (C999 doesn't exist, on purpose)
python main.py --hitl auto|defer|console # auto-approve (demo) | hold the advice | ask me
```
You get `outputs/report_<id>.json`, `run.log` and `memory.json`. I left some example runs in `sample_output/`.
The summary uses `groq:openai/gpt-oss-120b` by default; set `WEALTH_LLM` if you want another model.

## How it fits together
```
START -> supervisor <-> fetcher / analyzer / advisor / human_review / narrator
              \-> finalize -> END
```
| Part | What it does |
|---|---|
| supervisor (`graph.py`) | Looks at the state and decides who goes next. If a client fails, it jumps straight to finalize. |
| fetcher | Loads and cleans the data, then gets the CRM profile (retry, then cache, then defaults). |
| analyzer | Portfolio breakdown, anomaly checks, and a risk score out of 100. |
| advisor | Turns all that into a prioritised list of recommendations. |
| human_review | Uses `interrupt()` to pause the graph until `main.py` resumes it with a decision. |
| narrator (`llm.py`) | Asks the LLM for a short, plain-English note for the advisor. |
| `tools.py` | Every tool goes through `call_tool`, so retries, errors and logging work the same everywhere. |
| `memory.py` | Long-term memory in a JSON file. Short-term memory is just the graph state with a checkpointer. |

The workers never talk to each other. They read from and write to the shared state and hand control back to the
supervisor.

A review is triggered when the risk score is 60 or more, there's a HIGH anomaly, we had to fall back to a default
profile, or there's a suitability or KYC problem. The reviewer can approve, modify (drop some recommendations),
reject or defer. Deferring, or having nobody around, means the advice is held. It is never approved by default.

Anomaly checks are all simple on purpose, so every flag has a reason: a z-score outlier within a spending
category, a duplicate charge (same merchant and amount within 2 days), spending on risky categories (crypto,
gambling), and any single purchase that's at least half of the monthly income.

If something breaks: transient tool errors are retried with backoff, a CRM outage falls back to the cache and then
to defaults, bad rows get dropped and listed under `data_issues`, and an unknown client just produces a `failed`
report while the rest of the batch carries on. If the LLM call fails (no key, network), the report still goes out
with an empty summary and a warning.

## Why I did it this way
- **Supervisor instead of a fixed chain.** All the routing is in one function, so adding a step is a one-place change. The cost is an extra hop between steps.
- **No LLM making decisions.** Advice has to be repeatable and explainable, so scores and recommendations come from rules. The LLM only describes the result, and it runs after the human review so it never summarises advice that was rejected.
- **Plain z-score, no ML.** There's very little history per client and I wanted a reason for every flag. The weakness is that one huge outlier inflates the standard deviation, which is why the income-ratio rule exists as a second check.
- **JSON file for memory.** Easy to read and good enough for one process. I'd move to SQLite or LangGraph's `Store` for anything bigger.
- **Held by default.** If nobody can review, nothing is released.

## Assumptions
Everything is in USD with US number formats. A negative amount with no type is treated as a debit. Funds and ETFs
marked `diversified` don't count toward single-stock concentration. Equity ceilings by risk tolerance are
conservative 40%, moderate 70%, aggressive 90%. All thresholds are illustrative. The CRM and the data are mocks;
`data/crm_mock.json` has a `simulate_failures` section (C002 fails twice then works, C004 is always down).

## Demo clients
- C001: healthy, with one outlier and one duplicate charge
- C002: 84% in one stock plus crypto spending, flaky CRM, so HIGH risk and a review
- C003: messy data and an expired KYC, goes to review
- C004: no transactions and CRM down, so default profile and a review
- C005: no usable data, fails
- C999: not found, fails
