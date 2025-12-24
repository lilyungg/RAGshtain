from typing import Dict, Any

import httpx
from config import EXTERNAL_SERVICE_URL


async def send_to_external_service(text: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            EXTERNAL_SERVICE_URL,
            json={"text": text},
            timeout=70
        )
        return response.json()["result"]

base_url = "http://0.0.0.0:8000"
async def parse_url(url: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{base_url}/parse",
            json={"url": url}
        )
        r.raise_for_status()
        return r.json()
