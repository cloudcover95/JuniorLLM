import mlx.core as mx
from jr_llm.core.manifold_actuator import JuniorLLMManifold
from jr_llm.utility.algebra_parser import AlgebraParser
from jr_llm.utility.chrono_node import ChronoNode
from jr_llm.bridge.local_llm import LocalLLMBridge
from jr_llm.bridge.juniorfetch_bridge import JuniorFetchBridge
from jr_llm.bridge.juniormemsys_bridge import JuniorMemSysBridge
import logging

class IntentRouter:
    def __init__(self):
        self.manifold = JuniorLLMManifold()
        self.algebra = AlgebraParser()
        self.chrono = ChronoNode()
        self.llm = LocalLLMBridge()
        self.fetch = JuniorFetchBridge()
        self.memory = JuniorMemSysBridge()

    def route(self, user_input: str):
        lower = user_input.lower()

        # New Feature 1: Deterministic Tool Calling with TDA Validation
        if any(op in lower for op in ['+', '-', '*', '/', '^', 'calculate', 'solve']):
            return self.algebra.compute(user_input)

        # New Feature 2: Voice / Chrono utilities
        if "time" in lower or "clock" in lower:
            return self.chrono.get_clock()
        if "stopwatch" in lower:
            return self.chrono.toggle_stopwatch()
        if "timer" in lower:
            secs = int(''.join(filter(str.isdigit, lower)) or 60)
            self.chrono.deploy_timer(secs)
            return f"Timer set for {secs} seconds."

        # New Feature 3: JuniorFetch File Context
        if "file" in lower or "document" in lower:
            context = self.fetch.get_context(user_input)
            return f"File context: {context[:300]}..."

        # New Feature 4: JuniorMemSys Long-Term Memory
        if "remember" in lower or "memory" in lower:
            context = self.memory.get_long_term_context(user_input)
            return f"Long-term memory: {context[:300]}..."

        # New Feature 5: Auto-Recovery from Roadblocks
        llm_response = self.llm.ask(user_input)
        if len(llm_response) < 10 or "I don't know" in llm_response.lower():
            logging.info("LLM roadblock detected — falling back to manifold")
            return "Roadblock detected. Using deterministic manifold routing."

        return llm_response