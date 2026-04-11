# junior_memsys_suite/pipelines/dataset_miner.py
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from junior_memsys_suite.core.palace import MemoryPalace

class DatasetMiner:
    """
    Data integration pipeline. Bridges external data lakes (JSONL, HuggingFace Parquet)
    into the Topological Memory Palace via SVD etching.
    """
    def __init__(self):
        self.palace = MemoryPalace()

    def mine_conversations(self, jsonl_path: str | Path, wing: str = "conversations"):
        """Batch mine conversation datasets (JSONL) into the palace."""
        path = Path(jsonl_path)
        with open(path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                content = data.get("content") or str(data)
                self.palace.store(
                    wing=wing,
                    hall=data.get("session_id", "default"),
                    room=f"turn_{i}",
                    content=content,
                    z_score=1.8
                )
        print(f"[OK] Mined {i+1} conversation turns into manifold wing '{wing}'")

    def mine_hf_llm_math(self, wing: str = "llm-math", sample_size: int = 5):
        """Fetches and etches LangChain LLM-Math directly from HuggingFace."""
        # Pandas directly resolves the Parquet byte stream
        url = "https://huggingface.co/datasets/LangChainDatasets/llm-math/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
        print(f"[SYNC] Loading llm-math dataset from HuggingFace → {wing}")
        
        try:
            df = pd.read_parquet(url)
            for i, row in df.head(sample_size).iterrows():
                # Reconstruct structured text for verbatim tensor conversion
                verbatim = (
                    f"Question: {row.get('question', 'N/A')}\n"
                    f"Parsed Equation: {row.get('parsed_equation', row.get('equation', 'N/A'))}\n"
                    f"Answer: {row.get('answer', row.get('answer_float', 'N/A'))}"
                )
                self.palace.store(
                    wing=wing, 
                    hall="hf_import", 
                    room=f"math_{i}", 
                    content=verbatim, 
                    z_score=1.9
                )
            print(f"[OK] Etched {min(sample_size, len(df))} verbatim math tensors into wing '{wing}'")
        except Exception as e:
            print(f"[ERROR] Failed to fetch or parse dataset: {e}")

    def export_to_parquet(self, wing: str, output_path: str | Path):
        """Exports high-density topological TS data to Parquet."""
        target_wing = self.palace.root / wing
        if not target_wing.exists():
            print("[ERROR] Wing manifold non-existent.")
            return

        records = []
        for hall in target_wing.iterdir():
            if not hall.is_dir(): continue
            for room in hall.iterdir():
                if not room.is_dir(): continue
                drawer_file = room / "drawers.jsonl"
                if drawer_file.exists():
                    with open(drawer_file, "r") as f:
                        for line in f:
                            records.append(json.loads(line))
        
        if not records:
            print("[WARNING] No active nodes found for export.")
            return

        # Vectorize to PyArrow Table
        table = pa.Table.from_pylist(records)
        pq.write_table(table, output_path)
        print(f"[OK] Exported {len(records)} nodes to {output_path}")