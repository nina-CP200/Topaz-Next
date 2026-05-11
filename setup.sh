#!/bin/bash
# Topaz-Next 一键配置脚本
#
# 使用方法：
#   完整安装（含Web仪表盘）：
#     curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/setup.sh | bash
#     bash setup.sh
#
#   仅CLI（无前端，适合服务器）：
#     bash setup.sh --cli-only
#
#   中国镜像：
#     bash setup.sh --china

set -e

REPO_URL="https://github.com/nina-CP200/Topaz-Next"
INSTALL_DIR="$HOME/topaz-next"
MIRROR_URL=""
USE_CHINA=false
IS_REMOTE=false
CLI_ONLY=false

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
_STEP=0
log_step() {
    _STEP=$((_STEP + 1))
    echo ""
    echo -e "${BLUE}[Step $_STEP]${NC} $1"
    echo "-------------------------------------------"
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --china) USE_CHINA=true; shift ;;
            --cli-only) CLI_ONLY=true; shift ;;
            --help)
                echo "Topaz-Next 一键配置脚本"
                echo ""
                echo "使用方法:"
                echo "  bash setup.sh                    # 完整安装（含Web仪表盘）"
                echo "  bash setup.sh --cli-only         # 仅CLI（无前端）"
                echo "  bash setup.sh --china            # 中国镜像源"
                echo "  curl -sSL <URL> | bash           # 远程安装"
                exit 0 ;;
            *) shift ;;
        esac
    done
}

check_command() {
    command -v "$1" &> /dev/null
}

detect_mode() {
    if [ -f "src/analysis/daily.py" ] && [ -f "requirements.txt" ]; then
        PROJECT_DIR="$(pwd)"
        IS_REMOTE=false
        log_info "检测到本地项目目录"
    else
        IS_REMOTE=true
        PROJECT_DIR="$INSTALL_DIR"
        log_info "远程安装模式"
    fi
}

check_uv() {
    _STEP=0
    log_step "检查 uv 环境"

    if check_command uv; then
        log_success "uv 已安装: $(uv --version)"
        return
    fi

    log_info "未找到 uv，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if ! check_command uv; then
        log_error "uv 安装失败，请手动安装: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi

    log_success "uv 安装完成: $(uv --version)"
}

detect_network() {
    log_step "检测网络环境"

    if [ "$USE_CHINA" = true ]; then
        MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
        log_info "使用中国镜像源（清华大学）"
        return
    fi

    read -p "是否在中国大陆？(y/n): " in_china
    if [ "$in_china" = "y" ] || [ "$in_china" = "Y" ]; then
        MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
        log_info "使用中国镜像源"
    else
        MIRROR_URL=""
        log_info "使用官方 PyPI 源"
    fi
}

clone_repo() {
    if [ "$IS_REMOTE" = false ]; then
        return
    fi

    log_step "克隆项目"

    if [ -d "$PROJECT_DIR" ]; then
        log_warn "目录已存在: $PROJECT_DIR"
        read -p "是否删除并重新安装？(y/n): " choice
        if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
            rm -rf "$PROJECT_DIR"
        else
            log_info "使用现有目录"
            cd "$PROJECT_DIR"
            return
        fi
    fi

    if ! check_command git; then
        log_error "未找到 git"
        log_info "安装方法："
        log_info "  macOS:  brew install git"
        log_info "  Ubuntu: sudo apt install git"
        exit 1
    fi

    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    log_success "项目克隆完成"
}

install_python_deps() {
    log_step "安装 Python 依赖"

    if [ ! -d ".venv" ]; then
        uv venv
        log_success "虚拟环境已创建 (.venv)"
    else
        log_info "虚拟环境已存在 (.venv)"
    fi

    if [ -n "$MIRROR_URL" ]; then
        uv pip install -i "$MIRROR_URL" -r requirements.txt
        uv pip install -i "$MIRROR_URL" fastapi uvicorn
    else
        uv pip install -r requirements.txt
        uv pip install fastapi uvicorn
    fi

    log_success "Python 依赖安装完成"
}

install_frontend_deps() {
    if [ "$CLI_ONLY" = true ]; then
        log_info "CLI-only 模式，跳过前端依赖"
        return
    fi

    log_step "安装前端依赖（含 Node.js 检测）"

    if ! check_command node; then
        log_warn "未找到 Node.js，跳过前端安装"
        log_info "如需 Web 仪表盘，请安装 Node.js (>=18):"
        log_info "  macOS:  brew install node"
        log_info "  Ubuntu: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash - && sudo apt install nodejs"
        CLI_ONLY=true
        return
    fi

    log_success "Node.js: $(node --version)"
    log_success "npm: $(npm --version)"

    cd frontend
    npm install --include=dev
    npm run build
    cd ..

    log_success "前端构建完成"
}

setup_dirs() {
    log_step "创建目录结构"
    mkdir -p data/raw data/models data/cache
    log_success "目录创建完成"
}

fetch_data() {
    log_step "获取沪深300历史数据"

    if [ -f "$PROJECT_DIR/data/raw/csi300_full_history.csv" ]; then
        lines=$(wc -l < "$PROJECT_DIR/data/raw/csi300_full_history.csv" | tr -d ' ')
        if [ "$lines" -gt 1000 ]; then
            log_success "已有数据 ($lines 条记录)"
            read -p "是否重新获取？(y/n): " choice
            if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
                return
            fi
        fi
    fi

    log_info "获取数据（预计 3-5 分钟）..."
    uv run python -m src.data.fetchers.full_history --output data/raw/csi300_full_history.csv

    if [ ! -f "$PROJECT_DIR/data/raw/csi300_full_history.csv" ]; then
        log_error "数据获取失败"
        exit 1
    fi

    log_success "数据获取完成"
}

train_model() {
    log_step "训练模型"

    if [ -f "$PROJECT_DIR/data/models/ensemble_model.pkl" ]; then
        log_success "已有模型文件"
        read -p "是否重新训练？(y/n): " choice
        if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
            return
        fi
    fi

    if [ ! -f "$PROJECT_DIR/data/raw/csi300_full_history.csv" ]; then
        log_error "未找到训练数据"
        exit 1
    fi

    log_info "训练模型（预计 2-3 分钟）..."
    uv run python -m src.models.trainer --data data/raw/csi300_full_history.csv

    if [ ! -f "$PROJECT_DIR/data/models/ensemble_model.pkl" ]; then
        log_error "模型训练失败"
        exit 1
    fi

    log_success "模型训练完成"
}

test_run() {
    log_step "测试运行"

    read -p "是否运行测试？(y/n): " choice
    if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
        return
    fi

    log_info "运行测试分析..."
    uv run python -m src.analysis.query --top 5
    log_success "测试完成"
}

configure_slack() {
    log_step "配置 Slack（可选）"

    log_info "Slack 推送功能需要 Bot Token"
    read -p "是否配置？(y/n): " choice

    if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
        if [ -f "config/.env.example" ] && [ ! -f ".env" ]; then
            cp config/.env.example .env
            log_info "已创建 .env 文件，可后续手动配置"
        fi
        return
    fi

    echo ""
    log_info "获取 Token：https://api.slack.com/apps"
    log_info "添加 chat.postMessage 权限，安装后获取 Bot Token"
    echo ""

    read -p "请输入 Slack Bot Token: " token
    if [ -z "$token" ] || [[ ! "$token" =~ ^xoxb- ]]; then
        log_warn "Token 格式不正确，已跳过"
        return
    fi

    read -p "请输入 Channel（默认 #investments）: " channel
    [ -z "$channel" ] && channel="#investments"

    echo "SLACK_BOT_TOKEN=$token" > .env
    echo "SLACK_CHANNEL=$channel" >> .env
    log_success "配置保存到 .env"
}

show_complete() {
    local mode
    if [ "$CLI_ONLY" = true ]; then
        mode="CLI"
    else
        mode="完整（含Web仪表盘）"
    fi

    echo ""
    echo "=========================================="
    log_success "安装完成！(模式: $mode)"
    echo "=========================================="
    echo ""
    echo "项目目录: $PROJECT_DIR"
    echo ""

    if [ "$CLI_ONLY" = false ]; then
        echo "Web 仪表盘："
        echo "  cd $PROJECT_DIR"
        echo "  ./start.sh"
        echo "  浏览器打开 http://localhost:8000"
        echo ""
    fi

    echo "CLI 命令："
    echo "  cd $PROJECT_DIR"
    echo ""
    echo "  # 每日分析"
    echo "  uv run python -m src.analysis.daily"
    echo ""
    echo "  # 股票查询"
    echo "  uv run python -m src.analysis.query 600519"
    echo "  uv run python -m src.analysis.query --top 10"
    echo ""
    echo "  # 持仓分析（交互式）"
    echo "  ./portfolio.sh"
    echo ""
    echo "  # 重新训练"
    echo "  uv run python -m src.models"
    echo ""
    echo "  # 回测验证"
    echo "  uv run python -m src.backtest"
    echo ""
}

main() {
    parse_args "$@"

    echo ""
    echo "=========================================="
    echo "  Topaz-Next 一键配置"
    if [ "$CLI_ONLY" = true ]; then
        echo "  模式: 仅 CLI（无前端）"
    else
        echo "  模式: 完整安装（含 Web 仪表盘）"
    fi
    echo "=========================================="
    echo ""

    detect_mode
    check_uv
    detect_network
    clone_repo
    install_python_deps
    install_frontend_deps
    setup_dirs
    fetch_data
    train_model
    test_run
    configure_slack
    show_complete
}

main "$@"
