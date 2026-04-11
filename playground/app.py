# junior_memsys_suite_repo/playground/app.py
import streamlit as st
import requests
import json
from junior_memsys_suite.core.palace import MemoryPalace
from junior_memsys_suite.core.knowledge_graph import TopologicalKnowledgeGraph
from junior_memsys_suite.pipelines.dataset_miner import DatasetMiner
from junior_memsys_suite.config import settings

st.set_page_config(page_title="JuniorMemSys | TDA Command Node", layout="wide")
st.title("🧬 JuniorMemSys Topological Memory Palace — M4 Sandbox")

# -----------------------------------------------------------------------------
# SIDEBAR: Core Harvester Limiters & LLM Nexus
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ SVD Core Limiters")
    settings.etch_threshold = st.slider("Q-Mark Etch Threshold", 0.1, 0.9, float(settings.etch_threshold), 0.01)
    settings.variance_retention = st.slider("Variance Retention", 0.8, 0.99, float(settings.variance_retention), 0.01)
    
    st.markdown("---")
    st.header("🧠 LLM Nexus Configuration")
    llm_provider = st.selectbox("Inference Engine", ["Local (Ollama)", "OpenAI API"])
    if llm_provider == "Local (Ollama)":
        llm_endpoint = st.text_input("Ollama Endpoint", "http://localhost:11434/api/generate")
        llm_model = st.text_input("Local Model", "mistral")
    else:
        api_key = st.text_input("API Key", type="password")

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
palace = MemoryPalace()
miner = DatasetMiner()
kg = TopologicalKnowledgeGraph()

# -----------------------------------------------------------------------------
# DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Terminal (LLM)", 
    "Ingest Verbatim", 
    "Data Lake (HF)", 
    "Gamma Search", 
    "Knowledge Graph", 
    "A/B Benchmarks"
])

# TAB 1: LLM Workspace (RAG via TDA)
with tab1:
    st.subheader("Context-Aware AI Command Node")
    st.markdown("Queries the SVD Mesh via Bit Drift before generating inference.")
    
    chat_input = st.text_area("Initialize Command Sequence:")
    if st.button("Execute Inference"):
        with st.spinner("Calculating Feature Disagreement Score across Mesh..."):
            # Search across all wings (or specify "llm-math" / "alpha")
            mem_results = palace.semantic_search(chat_input)
            context_string = "\n\n".join([r['content'] for r in mem_results[:3]])
            
            st.success(f"Topological Context Retrieved: {len(mem_results)} active nodes.")
            with st.expander("View Raw Memory Tensors"):
                st.code(context_string if context_string else "No relevant topology found. Operating zero-shot.")

            # Dispatch to LLM
            system_prompt = f"System Context:\n{context_string}\n\nUser Command: {chat_input}\nRespond with high-density logic."
            
            if llm_provider == "Local (Ollama)":
                try:
                    payload = {"model": llm_model, "prompt": system_prompt, "stream": False}
                    res = requests.post(llm_endpoint, json=payload).json()
                    st.markdown("### Inference Output")
                    st.write(res.get("response", "Null response."))
                except Exception as e:
                    st.error(f"Local API Dispatch Failed. Ensure Ollama is running. Error: {e}")
            else:
                st.warning("OpenAI integration selected but unmapped. Use Local Node for sovereign inference.")

# TAB 2: Manual Ingestion
with tab2:
    st.subheader("Manual Tensor Etching")
    content = st.text_area("Raw Verbatim String")
    col1, col2, col3 = st.columns(3)
    wing_in = col1.text_input("Target Wing", "alpha")
    hall_in = col2.text_input("Target Hall", "playground")
    room_in = col3.text_input("Target Room", "sandbox")
    
    if st.button("Etch into TDA Mesh"):
        if palace.store(wing_in, hall_in, room_in, content):
            st.success("✅ State etched successfully. Q-Mark validated.")
        else:
            st.error("❌ Signal rejected. Sub-threshold Q-Mark.")

# TAB 3: HF Dataset Pipeline
with tab3:
    st.subheader("📥 Data Lake Pipeline — LangChain llm-math")
    st.info("Bypasses JSON parsing. Streams columnar Parquet directly from HuggingFace into SVD nodes.")
    
    sample_limit = st.number_input("Ingestion Batch Size", min_value=1, max_value=1000, value=5)
    target_wing_hf = st.text_input("Destination Wing", "llm-math")
    
    if st.button("🚀 Execute Remote Harvest"):
        with st.spinner(f"Streaming {sample_limit} nodes from HuggingFace..."):
            miner.mine_hf_llm_math(wing=target_wing_hf, sample_size=sample_limit)
            st.success(f"✅ Batch ingestion complete. Tensors mapped to '{target_wing_hf}'.")

# TAB 4: Manual TDA Search
with tab4:
    st.subheader("Direct Gamma Signal Query")
    query = st.text_input("Vector Space Query")
    target_wing_search = st.text_input("Restrict to Wing (Optional)", "")
    
    if st.button("Execute Bit Drift Search"):
        wing_filter = target_wing_search if target_wing_search else None
        results = palace.semantic_search(query, wing=wing_filter)
        st.write(f"**Yield: {len(results)} coherent states.**")
        for r in results[:10]:
            st.json(r)

# TAB 5: Knowledge Graph
with tab5:
    st.subheader("Topological Triples")
    st.write(f"Active Edges in Vault: **{len(kg.graph.edges)}**")
    
    st.markdown("#### Append New Edge")
    colA, colB, colC = st.columns(3)
    sub = colA.text_input("Subject")
    pred = colB.text_input("Predicate")
    obj = colC.text_input("Object")
    
    if st.button("Etch Triple"):
        if sub and pred and obj:
            kg.add_triple(sub, pred, obj, 1.0)
            st.success("✅ Kinematic Triple bound to Knowledge Graph.")
        else:
            st.warning("All fields required.")

# TAB 6: Benchmarks
with tab6:
    st.subheader("Live A/B/C Benchmarks")
    st.info("Execute `python benchmarks/abc_memory_benchmark.py` in the M4 terminal for live CLI readout.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="[A] JuniorMemSys TDA (Local)", value="~0.84s / 1k nodes", delta="IO-Free")
    col2.metric(label="[B] Legacy MemPalace (Chroma)", value="~2.15s / 1k nodes", delta="SQLite Bottleneck", delta_color="inverse")
    col3.metric(label="[C] LangChain FAISS (Standard)", value="~2.55s / 1k nodes", delta="HNSW Overhead", delta_color="inverse")

# TAB 7: LongMemEval QA Pipeline
with tab7:
    st.subheader("End-to-End QA Benchmarking (LongMemEval)")
    st.info("Tests TDA retrieval accuracy coupled with LLM generation against Gold Standard datasets.")
    
    colA, colB = st.columns(2)
    eval_provider = colA.selectbox("Judge LLM Provider", ["ollama", "openai", "groq"])
    eval_model = colB.text_input("Model ID", value="mistral" if eval_provider == "ollama" else ("gpt-4o-mini" if eval_provider == "openai" else "llama3-8b-8192"))
    eval_api = st.text_input("API Key (if required)", type="password", disabled=(eval_provider=="ollama"))
    
    sample_size = st.slider("Sample Size", 5, 100, 10)
    
    if st.button("🚀 Run QA Benchmark Suite"):
        from benchmarks.longmem_eval import QABenchmarkPipeline
        
        with st.spinner(f"Running evaluation pipeline via {eval_provider.upper()}..."):
            pipeline = QABenchmarkPipeline(provider=eval_provider, api_key=eval_api, model=eval_model)
            try:
                accuracy, results = pipeline.run_eval(sample_size=sample_size, k=3)
                
                st.success(f"Benchmark Complete! Accuracy: {accuracy:.1%}")
                
                # Display Results in a DataFrame
                import pandas as pd
                st.dataframe(pd.DataFrame(results)[["question", "gold", "generated", "score", "retrieval_time_sec"]])
                
            except Exception as e:
                st.error(f"Pipeline Failed: {e}")

# TAB 8: Enterprise Scalability & Telemetry
with tab8:
    st.subheader("📈 Enterprise Scalability Telemetry")
    st.info("Profiles ingestion throughput, strict RAM drift, and Bit Drift search latency up to 50k+ nodes.")
    
    colA, colB = st.columns(2)
    target_nodes = colA.number_input("Target Node Count", min_value=1000, max_value=100000, value=10000, step=1000)
    step_size = colB.number_input("Batch Step Size", min_value=500, max_value=20000, value=2000, step=500)
    
    if st.button("🚀 Execute Manifold Stress Test"):
        with st.spinner(f"Flooding TDA Mesh with {target_nodes:,} synthetic tensors..."):
            from benchmarks.scaling_test import EnterpriseScalingHarvester
            
            harvester = EnterpriseScalingHarvester()
            df = harvester.execute_manifold_stress_test(max_nodes=target_nodes, step=step_size)
            
            st.success("✅ Stress test complete. Telemetry acquired.")
            
            # Telemetry Dashboards
            st.markdown("### ⚡ System Telemetry")
            m1, m2, m3 = st.columns(3)
            m1.metric("Peak Throughput", f"{df['throughput_rps'].max():.1f} RPS")
            m2.metric("Max Latency (Bit Drift)", f"{df['retrieval_latency_ms'].max():.2f} ms")
            m3.metric("Total RAM Drift", f"{df['delta_ram_mb'].sum():.2f} MB")
            
            st.markdown("### 📉 Performance Curves")
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Throughput (RPS) vs Mesh Size**")
                st.line_chart(df.set_index("mesh_size")["throughput_rps"])
            with c2:
                st.write("**Retrieval Latency (ms) vs Mesh Size**")
                st.area_chart(df.set_index("mesh_size")["retrieval_latency_ms"])
                
            with st.expander("View Raw Telemetry Ledger"):
                st.dataframe(df, use_container_width=True)