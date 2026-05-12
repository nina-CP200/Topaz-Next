from __future__ import annotations
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.routers.common import PROJECT_ROOT

router = APIRouter()

ENV_FILE = os.path.join(PROJECT_ROOT, ".env")


class SlackConfig(BaseModel):
    token: str = ""
    channel: str = ""


def _load_env() -> dict:
    vars = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vars[k.strip()] = v.strip()
    return vars


@router.get("/slack")
def get_slack_config():
    env = _load_env()
    return SlackConfig(
        token=env.get("SLACK_BOT_TOKEN", ""),
        channel=env.get("SLACK_CHANNEL", ""),
    )


@router.put("/slack")
def save_slack_config(cfg: SlackConfig):
    existing = _load_env()
    existing["SLACK_BOT_TOKEN"] = cfg.token
    existing["SLACK_CHANNEL"] = cfg.channel

    os.makedirs(os.path.dirname(ENV_FILE) or ".", exist_ok=True)
    with open(ENV_FILE, "w") as f:
        for k, v in existing.items():
            if v:
                f.write(f"{k}={v}\n")
    return {"message": "Slack 配置已保存"}
