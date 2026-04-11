from juniorfetch.core.crawler import JuniorFetchCrawler  # from your JuniorFetch SDK

class JuniorFetchBridge:
    def __init__(self):
        self.crawler = JuniorFetchCrawler()

    def get_context(self, query: str) -> str:
        results = self.crawler.palace.semantic_search(query, wing="files")
        return "\n\n".join([r["content"][:500] for r in results[:3]])