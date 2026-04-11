# scripts/sdk_harvester.py
import os, json, hashlib, time
from pathlib import Path

FORBIDDEN_PATHS = {"01_Legal", "02_Assets"}
EXCLUDE_DIRS = {".git", "__pycache__", "env", "venv", ".venv", "build", "dist"}

def generate_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def harvest_suite(root_dir: str = "."):
    snapshot = {
        "suite_brand": "JuniorMemSys-Suite",
        "audit_timestamp": time.time(),
        "topology_version": "0.2.2",
        "binary_gate": "SVD_OMNI_CORE_PROTECTED",
        "ledger": {}
    }
    
    root_path = Path(root_dir).resolve()
    
    for current_root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not any(f in d for f in FORBIDDEN_PATHS)]
        
        for file in files:
            if file.endswith(('.py', '.so', '.md', '.toml', '.yml', '.jsonl')):
                full_path = Path(current_root) / file
                rel_path = str(full_path.relative_to(root_path))
                snapshot["ledger"][rel_path] = generate_hash(full_path)
                
    output_path = root_path / "jms_audit_ledger.json"
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[AUDIT] JuniorMemSys-Suite Ledger Generated: {output_path}")

if __name__ == "__main__":
    harvest_suite()