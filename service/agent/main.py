from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import traceback

from agent import agent_main
from search_client import SearchClient
app = FastAPI(
    title="DeepSeek RAG Agent API",
    description="API для QA / Interview / Course агента на FastAPI + DeepSeek + RAG",
    version="1.0.0",
)
search_client = SearchClient("http://0.0.0.0:8000")

# ---------
# SCHEMAS
# ---------

class AgentRequest(BaseModel):
    text: Optional[str] = Field(
        None,
        description="Пользовательский запрос (вопрос, тема, описание)"
    )
    query: Optional[str] = None
    message: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    result: str


class HealthResponse(BaseModel):
    status: str


# ---------
# ROUTES
# ---------

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.post("/agent", response_model=AgentResponse)
def run_agent(req: AgentRequest):
    try:
        # agent_main умеет принимать dict
        result = agent_main(req.dict(exclude_none=True), search_client)
        return {"result": result}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ---------
# ENTRYPOINT
# ---------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
