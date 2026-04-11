# junior_memsys_suite/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from junior_memsys_suite.core.palace import MemoryPalace
from junior_memsys_suite.core.knowledge_graph import TopologicalKnowledgeGraph

app = FastAPI(title="JCLLC Mem Sys API - MCP Protocol")
palace = MemoryPalace()
kg = TopologicalKnowledgeGraph()

class MineRequest(BaseModel):
    content: str
    wing: str
    hall: str
    room: str
    z_score: float = 1.5

class SearchRequest(BaseModel):
    query: str
    wing: Optional[str] = None

class GraphNode(BaseModel):
    subject: str
    predicate: str
    object: str
    q_mark: float = 1.0

@app.get("/")
async def root_status():
    """Health check endpoint resolving Root 404."""
    return {"status": "online", "system": "JCLLC TDA Memory Palace Node", "protocol": "MCP"}

@app.post("/mcp/mine")
async def mcp_mine(req: MineRequest):
    success = palace.store(req.wing, req.hall, req.room, req.content, req.z_score)
    if not success:
        raise HTTPException(status_code=406, detail="Signal rejected by Q-Mark threshold.")
    return {"status": "stored", "drift": "nominal"}

@app.post("/mcp/search")
async def mcp_search(req: SearchRequest):
    results = palace.semantic_search(req.query, req.wing)
    return {"hits": len(results), "data": results}

@app.post("/mcp/kg/add")
async def mcp_kg_add(req: GraphNode):
    kg.add_triple(req.subject, req.predicate, req.object, req.q_mark)
    return {"status": "triple_etched"}