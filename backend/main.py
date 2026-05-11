from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from backend.routers import market, analysis, sectors, portfolio

app = FastAPI(title="Topaz-Next API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market.router, prefix="/api/market", tags=["market"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(sectors.router, prefix="/api/sectors", tags=["sectors"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])


@app.get("/api/health")
def health():
    return {"status": "ok"}


FRONTEND_DIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")


@app.get("/assets/{rest:path}")
def serve_assets(rest: str):
    path = os.path.join(FRONTEND_DIST, "assets", rest)
    if os.path.isfile(path):
        return FileResponse(path)
    return _serve_index()


@app.get("/{filename:path}")
def serve_frontend(filename: str):
    if filename.startswith("api/"):
        return {"error": "not found"}
    path = os.path.join(FRONTEND_DIST, filename)
    if os.path.isfile(path):
        return FileResponse(path)
    return _serve_index()


def _serve_index():
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"error": "frontend not built. run: cd frontend && npm run build"}
