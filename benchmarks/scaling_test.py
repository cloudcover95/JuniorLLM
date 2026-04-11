# benchmarks/scaling_test.py
import time
import psutil
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from junior_memsys_suite.core.palace import MemoryPalace
from junior_memsys_suite.pipelines.dataset_miner import DatasetMiner

@dataclass
class ScalingMetrics:
    mesh_size: int
    ingest_time_sec: float
    throughput_rps: float
    delta_ram_mb: float
    retrieval_latency_ms: float
    q_mark_rejection_rate: float

class EnterpriseScalingHarvester:
    """
    High-fidelity scalability profiling for JuniorMemSys TDA manifolds.
    Optimized for Apple Silicon / Edge Node limit testing.
    """
    def __init__(self, target_wing: str = "stress_test"):
        self.palace = MemoryPalace()
        self.wing = target_wing
        self.process = psutil.Process()

    def execute_manifold_stress_test(self, max_nodes: int = 50000, step: int = 10000) -> pd.DataFrame:
        print(f"[INIT] Enterprise Scalability Harvester - Target: {max_nodes} Nodes")
        metrics_ledger = []
        
        # Isolate baseline memory
        base_mem = self.process.memory_info().rss / (1024**2)

        for n in range(step, max_nodes + 1, step):
            current_mem = self.process.memory_info().rss / (1024**2)
            t0 = time.perf_counter()

            # High-density batch ingestion
            rejected = 0
            for i in range(step):
                # Variable signal intensity simulates real-world noisy data streams
                success = self.palace.store(
                    wing=self.wing,
                    hall="batch_strata",
                    room=f"node_{n - step + i}",
                    content=f"Synthetic topological payload {i} for Gamma Signal verification. Dense state logic required.",
                    z_score=np.random.uniform(1.2, 3.5) 
                )
                if not success:
                    rejected += 1

            ingest_duration = time.perf_counter() - t0
            end_mem = self.process.memory_info().rss / (1024**2)

            # Bit Drift Retrieval Latency Audit
            t_ret_0 = time.perf_counter()
            _ = self.palace.semantic_search("Gamma Signal verification", wing=self.wing)
            retrieval_latency = (time.perf_counter() - t_ret_0) * 1000

            metrics = ScalingMetrics(
                mesh_size=n,
                ingest_time_sec=round(ingest_duration, 4),
                throughput_rps=round(step / ingest_duration, 2),
                delta_ram_mb=round(end_mem - current_mem, 2),
                retrieval_latency_ms=round(retrieval_latency, 3),
                q_mark_rejection_rate=round(rejected / step, 4)
            )
            metrics_ledger.append(asdict(metrics))
            print(f"[AUDIT] {n:,} Nodes | ΔRAM: {metrics.delta_ram_mb:0.2f}MB | Latency: {metrics.retrieval_latency_ms:0.3f}ms | Thruput: {metrics.throughput_rps:0.1f} RPS")

        # Export cryptographic/audit ledger equivalent for benchmarks
        df = pd.DataFrame(metrics_ledger)
        out_path = Path("benchmarks/results/enterprise_scaling_ledger.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[COMPLETE] Scalability ledger exported to {out_path}")
        return df

if __name__ == "__main__":
    harvester = EnterpriseScalingHarvester()
    harvester.execute_manifold_stress_test(max_nodes=10000, step=2000)