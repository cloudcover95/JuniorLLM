from junior_memsys_suite.core.palace import MemoryPalace  # from your JuniorMemSys SDK

class JuniorMemSysBridge:
    def __init__(self):
        self.palace = MemoryPalace()

    def get_long_term_context(self, query: str) -> str:
        results = self.palace.semantic_search(query)
        return "\n\n".join([r["content"][:300] for r in results[:5]])