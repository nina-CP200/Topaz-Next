from __future__ import annotations
import json
import os
from fastapi import APIRouter
from pydantic import BaseModel
from backend.routers.common import PROJECT_ROOT

router = APIRouter()

CONFIG_FILE = os.path.join(PROJECT_ROOT, "data/raw/slack_configs.json")


class SlackConfigItem(BaseModel):
    name: str
    token: str
    channel: str


def _load() -> list[dict]:
    if not os.path.exists(CONFIG_FILE):
        return []
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save(configs: list[dict]):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)


@router.get("/slack")
def list_configs():
    return {"configs": _load()}


@router.put("/slack")
def save_configs(data: list[SlackConfigItem]):
    configs = []
    for item in data:
        configs.append({"name": item.name, "token": item.token, "channel": item.channel})
    _save(configs)
    return {"message": f"已保存 {len(configs)} 个 Slack 配置"}
