from __future__ import annotations
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from backend.routers import market, analysis, sectors, portfolio, settings

app = FastAPI(title="Topaz-Next API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market.router, prefix="/api/market", tags=["market"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(sectors.router, prefix="/api/sectors", tags=["sectors"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["portfolio"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])


@app.get("/api/health")
def health():
    return {"status": "ok"}


FRONTEND_DIST = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "frontend" / "dist"


def _safe_path(base: Path, *parts: str) -> Path:
    target = base.joinpath(*parts).resolve()
    if not str(target).startswith(str(base.resolve())):
        raise HTTPException(status_code=403, detail="Forbidden")
    return target


@app.get("/assets/{rest:path}")
def serve_assets(rest: str):
    path = _safe_path(FRONTEND_DIST, "assets", rest)
    if path.is_file():
        return FileResponse(str(path))
    return _serve_index()


@app.get("/{filename:path}")
def serve_frontend(filename: str):
    if filename.startswith("api/"):
        return {"error": "not found"}
    path = _safe_path(FRONTEND_DIST, filename)
    if path.is_file():
        return FileResponse(str(path))
    return _serve_index()


def _serve_index():
    index_path = FRONTEND_DIST / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path), media_type="text/html")
    return {"error": "frontend not built. run: cd frontend && npm run build"}
