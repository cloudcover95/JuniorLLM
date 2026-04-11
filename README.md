# JuniorLLM

**Local Manifold LLM Layering SDK** — Zero-latency hardware actuation via TDA + Bit Drift + Omni Math

Bypasses cloud LLM roadblocks for timers, algebra, word scrambles, file context, and OS commands.

## Visualizations

![Data Server Audit Across Cluster](assets/audit_cluster_tda_4k.png)

![Dataset Stack & Black Box Inference](assets/dataset_stack_tda_inference_4k.png)

## Quick Start

```bash
cd JuniorLLM
python -m venv .venv
source .venv/bin/activate
pip install -e .
juniorllm ask "start a timer for 30 seconds"
juniorllm chat

# JuniorMemSys-Suite v0.4.0

**Sovereign Topological Memory Palace SDK**

Apple Silicon Native · MLX-first · TDA + SVD · Bit Drift Inference · MCP Ready

JuniorMemSys-Suite is a production-grade topological memory system built for AI agents and enterprises. It combines **Topological Data Analysis (TDA)**, **Singular Value Decomposition (SVD)**, and **Bit Drift quantization** to create memory retrieval systems that are:

- **Power-efficient** on Apple Silicon
- **Logic-dense** and enterprise-auditable
- **Language-agnostic** (Python SDK, Swift client, MCP-compatible)

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/cloudcover95/JuniorMemSys-Suite
cd JuniorMemSys-Suite
pip install -e ".[dev,playground,benchmarks]"
```

### 2. Bootstrap the Node

The `omega_boot.sh` script handles environment alignment and target service initialization.

```bash
./omega_boot.sh --ui          # Launch Streamlit / Gradio TDA Sandbox
./omega_boot.sh --grpc        # Start gRPC receiver for Swift TrueDepth/ARKit
./omega_boot.sh --web         # Start FastAPI MCP / WebRTC bridge
```

---

## 🛠 Architecture & Directory Tree

```
JuniorMemSys-Suite/
├── 📦 junior_memsys_suite/          # Core installable SDK
│   ├── core/                        # TDA Engine & Math Kernels
│   │   ├── palace.py                # MemoryPalace logic (Provenance + Storage)
│   │   ├── tda_mesh.py              # SVD + Bit Drift Manifolds
│   │   ├── encoder.py               # MLX-Native Sovereign Encoder
│   │   └── audit.py                 # Enterprise Integrity & Benchmark Engine
│   └── pipelines/                   # Data integration layer (DatasetMiner, Chunker)
├── 🖥️ playground/                   # Streamlit Dashboard & Globe Brain Viz
├── 🔬 benchmarks/                    # LongMemEval QA & Scaling Tests
├── 🛠️ scripts/                       # Harvester, Kernel Builders, and Seeders
└── server.py                        # MCP Protocol / FastAPI Server
```

---

## ⚖️ Technical Baseline

JuniorMemSys utilizes **Bit Drift** instead of cosine similarity. Tensors are projected via SVD and quantized to a ±1 binary signature. Retrieval computes the mean Feature Distance across the manifold, enabling sub-millisecond lookups on embedded systems.

### Example: Storing a Memory

```python
from junior_memsys_suite.core import MemoryPalace

palace = MemoryPalace()

palace.store(
    wing="alpha", 
    hall="directives", 
    room="root_node",
    content="Optimize for power-efficient, logic-dense engineering.",
    z_score=2.5
)
```

---

## 📡 Integration Points

### 1. Native Python SDK
Import and use directly in your agent loops.

### 2. MCP Tool Integration

JuniorMemSys serves as a native tool for Claude or Cursor. Connect to the local node:

```
GET http://localhost:8000/mcp/tools
```

### 3. Server Integrity Audit

Prove data consistency across your memory fabric:

```bash
junior-memsys audit --wing alpha
```

---

## 🗺 Roadmap (v0.5+)

- **Incremental Indexing**: Automated file-watching and delta-etching
- **Distributed Swarm**: Local mesh synchronization across multi-agent clusters (Orange Pi/Linux)
- **Native Swift SDK**: Direct TrueDepth/ARKit memory capture
- **Quantum Kernel Mode**: Adaptive bit-width for constrained devices

---

## 📝 License

MIT License — See LICENSE file for details.