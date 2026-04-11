import json
import hashlib
from pathlib import Path
from datetime import datetime

class BitDriftAuditor:
    def __init__(self):
        self.log_dir = Path.home() / ".juniorllm" / "audit"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.log_dir / "bitdrift_ledger.jsonl"

    def _hash_entry(self, entry: dict) -> str:
        return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()

    def log(self, action: str, **kwargs):
        entry = {
            "timestamp_utc": datetime.utcnow().isoformat(),
            "action": action,
            **kwargs
        }
        entry["hash"] = self._hash_entry(entry)
        with open(self.ledger, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[AUDIT] {action} | Hash={entry['hash'][:8]}...")