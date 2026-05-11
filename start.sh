#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== 构建前端 ==="
cd frontend && npm run build --loglevel=error && cd ..

echo ""
echo "=== 启动 Topaz-Next ==="
echo ""

uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

# 等待服务就绪后打开浏览器
for i in $(seq 1 10); do
  if curl -s http://127.0.0.1:8000/api/health > /dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "http://localhost:8000"
open http://localhost:8000

wait $SERVER_PID
