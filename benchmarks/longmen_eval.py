# junior_memsys_suite_repo/benchmarks/longmem_eval.py
import pandas as pd
import requests
import json
import time
from pathlib import Path
from openai import OpenAI
from groq import Groq
from junior_memsys_suite.core.palace import MemoryPalace
from junior_memsys_suite.pipelines.dataset_miner import DatasetMiner

class QABenchmarkPipeline:
    def __init__(self, provider="ollama", api_key=None, model="mistral"):
        self.palace = MemoryPalace()
        self.miner = DatasetMiner()
        self.provider = provider.lower()
        self.model = model
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "groq":
            self.client = Groq(api_key=api_key)
        elif self.provider == "ollama":
            self.endpoint = "http://localhost:11434/api/generate"

    def _generate_answer(self, prompt: str) -> str:
        """Dynamic dispatch to the selected LLM provider."""
        if self.provider in ["openai", "groq"]:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return res.choices[0].message.content.strip()
        elif self.provider == "ollama":
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            res = requests.post(self.endpoint, json=payload).json()
            return res.get("response", "").strip()
        return ""

    def run_eval(self, sample_size: int = 10, k: int = 3):
        print(f"[INIT] Booting LongMemEval QA Pipeline using {self.provider.upper()} ({self.model})")
        
        # 1. Ensure data is in the manifold
        print(f"[SYNC] Ensuring {sample_size} nodes exist in 'longmem_eval' wing...")
        self.miner.mine_hf_llm_math(wing="longmem_eval", sample_size=sample_size)

        # Load reference dataset
        url = "https://huggingface.co/datasets/LangChainDatasets/llm-math/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
        df = pd.read_parquet(url).head(sample_size)

        results = []
        correct = 0

        for idx, row in df.iterrows():
            question = row["question"]
            gold_answer = str(row.get("answer", row.get("answer_float", "")))

            # 2. Retrieve via TDA Bit Drift
            t0 = time.time()
            retrieved = self.palace.semantic_search(question, wing="longmem_eval")[:k]
            context = "\n\n".join([r["content"] for r in retrieved])
            retrieval_time = time.time() - t0

            # 3. Generate Answer
            prompt = f"System Context (Topological Mesh):\n{context}\n\nQuestion: {question}\nAnswer concisely:"
            generated = self._generate_answer(prompt)

            # 4. Score (Strict Substring Match)
            score = 1 if gold_answer in generated else 0
            correct += score

            results.append({
                "question": question,
                "gold": gold_answer,
                "generated": generated,
                "score": score,
                "retrieval_time_sec": round(retrieval_time, 4),
                "retrieved_nodes": len(retrieved)
            })
            print(f"[{idx+1}/{sample_size}] Q: {question[:30]}... | Score: {score}")

        # 5. Export and Return
        accuracy = correct / sample_size
        Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv("benchmarks/results/longmem_eval_results.csv", index=False)
        
        print(f"\n[COMPLETE] LongMemEval Accuracy (@k={k}): {accuracy:.1%}")
        return accuracy, results

if __name__ == "__main__":
    # Example local run
    pipeline = QABenchmarkPipeline(provider="ollama", model="mistral")
    pipeline.run_eval(sample_size=5)