# junior-memsys-suite/pipelines/batch_ingest.py
import os
from pathlib import Path
from pipelines.dataset_miner import DatasetMiner

def execute_batch_ingestion(source_dir: str, target_wing: str):
    """Batch processes raw JSONL archives."""
    miner = DatasetMiner()
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"[FAIL] Target directory {source_dir} not mapped.")
        return

    for file in source_path.glob("*.jsonl"):
        print(f"Ingesting topological subset: {file.name}")
        miner.mine_conversations(file, wing=target_wing)
        
    print("[OK] Batch ingestion sequence complete.")

if __name__ == "__main__":
    execute_batch_ingestion("data/raw_exports", "historical_logs")