# benchmarks/abc_verification.py
import time
import psutil
import numpy as np
from junior_memsys_suite.core.palace import MemoryPalace

def audit_performance():
    palace = MemoryPalace()
    t0 = time.perf_counter()
    mem0 = psutil.Process().memory_info().rss / 1024**2
    
    # [A] JuniorMemSys TDA Ingestion (1000 nodes)
    for i in range(1000):
        palace.store("alpha", "bench", f"node_{i}", "Verbatim tensor state")
        
    t1 = time.perf_counter()
    mem1 = psutil.Process().memory_info().rss / 1024**2
    
    print(f"SYSTEM [A] JuniorMemSys: {t1-t0:.4f}s | ΔRAM: {mem1-mem0:.2f}MB")

if __name__ == "__main__":
    audit_performance()