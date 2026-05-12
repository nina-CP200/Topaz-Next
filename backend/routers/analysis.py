from __future__ import annotations
import json
import os
import threading
from datetime import datetime
from fastapi import APIRouter, HTTPException
from src.analysis.query import load_analysis_results
from src.analysis.daily import analyze_stocks
from src.reports.sender import send_score_ranking
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


@router.get("/report")
def analysis_report():
    data = load_analysis_results()
    if data is None:
        return {"error": "暂无分析结果", "text": ""}

    results = data.get("results", data.get("all_results", []))
    if not results:
        return {"error": "暂无分析结果", "text": ""}

    regime = data.get("market_regime", "sideways")
    confidence = data.get("model_confidence", 0.5)
    adv_ratio = data.get("advance_ratio", 0.5)
    position = data.get("recommended_position", 0.5)
    date = data.get("date", datetime.now().strftime("%Y-%m-%d %H:%M"))

    sorted_by_score = sorted(results, key=lambda x: x.get("composite_score", 0), reverse=True)

    regime_names = {"bull": "📈 牛市", "bear": "📉 熊市", "sideways": "➡️ 震荡", "recovery": "📊 复苏", "pullback": "📉 回调"}
    regime_name = regime_names.get(regime, regime)

    lines = []
    lines.append(f"📊 Topaz-Next 每日策略报告")
    lines.append(f"⏱ {date}")
    lines.append("")
    lines.append(f"【市场环境】{regime_name}  |  模型置信度: {confidence:.0%}  |  上涨比例: {adv_ratio:.1%}")
    lines.append(f"【建议仓位】{position:.0%}")
    lines.append("")

    buy_list = [r for r in sorted_by_score if r.get("composite_score", 0) >= 0.55]
    sell_list = [r for r in sorted_by_score if r.get("composite_score", 0) < 0.40]

    if buy_list:
        lines.append("🟢 建议买入")
        for i, s in enumerate(buy_list[:10], 1):
            adv = s.get("advice", "")
            score = s.get("composite_score", 0)
            rsn = " | ".join(s.get("reasons", []))[:80]
            pos = s.get("position_pct", 0)
            lines.append(f"  #{i} {s['symbol']} {s['name']} ({s.get('industry','')})")
            lines.append(f"     评分: {score:.3f} | 建议仓位: {pos:.0%} | {adv}")
            if rsn:
                lines.append(f"     理由: {rsn}")
        lines.append("")

    if sell_list:
        lines.append("🔴 建议回避/卖出")
        for i, s in enumerate(sell_list[:8], 1):
            score = s.get("composite_score", 0)
            adv = s.get("advice", "")
            lines.append(f"  #{i} {s['symbol']} {s['name']}  评分: {score:.3f} | {adv}")
        lines.append("")

    buy_count = len(buy_list)
    hold_count = len([r for r in sorted_by_score if 0.40 <= r.get("composite_score", 0) < 0.55])
    sell_count = len(sell_list)
    lines.append(f"📊 分布: 买入{buy_count} / 观望{hold_count} / 回避{sell_count}")
    lines.append("")
    lines.append("⚠️ 本分析仅供参考，不构成投资建议。市场有风险，投资需谨慎。")

    return {"text": "\n".join(lines), "error": ""}


@router.post("/send-slack")
def send_to_slack():
    data = load_analysis_results()
    if data is None:
        raise HTTPException(status_code=400, detail="暂无分析结果")

    results = data.get("results", data.get("all_results", []))
    if not results:
        raise HTTPException(status_code=400, detail="暂无分析结果")

    regime = data.get("market_regime", "sideways")
    confidence = data.get("model_confidence", 0.5)
    adv_ratio = data.get("advance_ratio", 0.5)

    ok = send_score_ranking(results, regime, confidence, adv_ratio)
    if ok:
        return {"message": "报告已发送到 Slack"}
    raise HTTPException(status_code=500, detail="Slack 发送失败，请检查 Token 配置")


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
