from __future__ import annotations
import json
import os
import threading
from datetime import datetime
from fastapi import APIRouter
from src.analysis.query import load_analysis_results
from src.analysis.daily import analyze_stocks
from backend.routers.common import PROJECT_ROOT, patch_results

router = APIRouter()

CSI300_STOCK_LIST = os.path.join(PROJECT_ROOT, "config/csi300_stocks.json")
STATUS_FILE = os.path.join(PROJECT_ROOT, "data/raw/analysis_status.json")
RESULT_FILE = os.path.join(PROJECT_ROOT, "data/raw/latest_analysis_results.json")

_lock = threading.Lock()
_status = {"status": "idle", "progress": 0, "message": ""}


def _save_status(status: str, progress: int, message: str):
    global _status
    with _lock:
        _status = {"status": status, "progress": progress, "message": message}
    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(_status, f)


@router.get("/daily")
def daily_analysis():
    data = load_analysis_results()
    if data is None:
        return {"error": "暂无分析结果，请先触发分析"}
    regime = data.get("market_regime", "sideways")
    results = data.get("results", data.get("all_results", []))
    data["results"] = patch_results(results, regime)
    return data


@router.get("/status")
def analysis_status():
    with _lock:
        return dict(_status)


@router.post("/refresh")
def refresh_analysis():
    thread = threading.Thread(target=_run_analysis, daemon=True)
    thread.start()
    return {"message": "分析已开始"}


def _run_analysis():
    try:
        _save_status("running", 0, "开始分析...")
        result = analyze_stocks(CSI300_STOCK_LIST, use_csi300_model=False)
        all_results = result.get("all_results", [])

        if not all_results:
            _save_status("error", 0, "分析结果为空")
            return

        regime = result.get("market_regime", "sideways")
        all_results = patch_results(all_results, regime)

        sorted_results = sorted(all_results, key=lambda x: x.get("composite_score", 0), reverse=True)
        for i, r in enumerate(sorted_results):
            r["rank"] = i + 1

        output_data = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "market_regime": regime,
            "model_confidence": result.get("model_confidence", 0.5),
            "advance_ratio": result.get("advance_ratio", 0.5),
            "recommended_position": result.get("recommended_position", 0.5),
            "total_stocks": len(sorted_results),
            "results": sorted_results,
        }

        os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
        with open(RESULT_FILE, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        _save_status("done", 100, f"分析完成，共 {len(sorted_results)} 只股票")
    except Exception as e:
        _save_status("error", 0, str(e)[:200])
