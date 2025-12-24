from typing import List, Dict, Any
import httpx


class SearchClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def parse_url(self, url: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.base_url}/parse",
                json={"url": url}
            )
            r.raise_for_status()
            return r.json()

    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "k": k
                }
            )
            r.raise_for_status()
            return r.json()
