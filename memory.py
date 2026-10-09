"""Long-term memory: JSON file with per-client CRM cache + run history.
(Short-term memory = the LangGraph state, checkpointed per thread by MemorySaver.)"""
import json
from pathlib import Path


class LongTermMemory:
    def __init__(self, path="memory.json"):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text())
        except (FileNotFoundError, ValueError):             # missing or corrupt -> start fresh
            self.data = {}

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def get_crm(self, cid):
        return self.data.get(cid, {}).get("crm")

    def save_crm(self, cid, crm):
        self.data.setdefault(cid, {"runs": []})["crm"] = crm
        self._save()

    def last_run(self, cid):
        runs = self.data.get(cid, {}).get("runs", [])
        return runs[-1] if runs else None

    def save_run(self, cid, summary):
        self.data.setdefault(cid, {"runs": []})["runs"].append(summary)
        self.data[cid]["runs"] = self.data[cid]["runs"][-20:]
        self._save()
