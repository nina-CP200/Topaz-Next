#!/bin/bash
# Topaz-Next 一键配置脚本
#
# 使用方法：
#   方式1（远程安装）：curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/setup.sh | bash
#   方式2（本地配置）：git clone ... && cd Topaz-Next && bash setup.sh
#   方式3（中国镜像）：bash setup.sh --china

set -e

REPO_URL="https://github.com/nina-CP200/Topaz-Next"
INSTALL_DIR="$HOME/topaz-next"
PYTHON_CMD=""
PIP_CMD=""
MIRROR_URL=""
USE_CHINA=false
IS_REMOTE=false

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_step()    { echo ""; echo -e "${BLUE}[$1]${NC} $2"; echo "-------------------------------------------"; }

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --china) USE_CHINA=true; shift ;;
            --help)
                echo "Topaz-Next 一键配置脚本"
                echo ""
                echo "使用方法:"
                echo "  bash setup.sh              # 本地配置"
                echo "  bash setup.sh --china      # 强制使用中国镜像源"
                echo "  curl -sSL <URL> | bash     # 远程安装"
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

check_python() {
    log_step 1 "检查 Python 环境"
    
    if check_command python3; then
        PYTHON_CMD="python3"
    elif check_command python; then
        PYTHON_CMD="python"
    else
        log_error "未找到 Python，请先安装 Python 3.8+"
        log_info "安装方法："
        log_info "  macOS:  brew install python3"
        log_info "  Ubuntu: sudo apt install python3"
        log_info "  Windows: https://www.python.org/downloads/"
        exit 1
    fi
    
    VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_success "Python 版本: $VERSION"
    
    MAJOR=$(echo $VERSION | cut -d. -f1)
    MINOR=$(echo $VERSION | cut -d. -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
        log_error "Python 版本过低，需要 3.8+"
        exit 1
    fi
}

check_pip() {
    log_step 2 "检查 pip"
    
    if check_command pip3; then
        PIP_CMD="pip3"
    elif check_command pip; then
        PIP_CMD="pip"
    else
        log_info "安装 pip..."
        $PYTHON_CMD -m ensurepip --default-pip
        PIP_CMD="pip"
    fi
    
    log_success "pip 已就绪"
}

detect_network() {
    log_step 3 "检测网络环境"
    
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
    
    log_step 4 "克隆项目"
    
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

install_deps() {
    log_step 5 "安装依赖"
    
    if [ -n "$MIRROR_URL" ]; then
        log_info "镜像源: $MIRROR_URL"
        $PIP_CMD install -i "$MIRROR_URL" -r requirements.txt
    else
        $PIP_CMD install -r requirements.txt
    fi
    
    log_success "依赖安装完成"
}

setup_dirs() {
    log_step 6 "创建目录结构"
    
    mkdir -p data/raw data/models data/cache
    mkdir -p config
    
    log_success "目录创建完成"
}

fetch_data() {
    log_step 7 "获取沪深300历史数据"
    
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
    
    $PYTHON_CMD -m src.data.fetchers.full_history --output data/raw/csi300_full_history.csv
    
    if [ ! -f "$PROJECT_DIR/data/raw/csi300_full_history.csv" ]; then
        log_error "数据获取失败"
        exit 1
    fi
    
    log_success "数据获取完成"
}

train_model() {
    log_step 8 "训练模型"
    
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
    
    $PYTHON_CMD -m src.models.trainer --data data/raw/csi300_full_history.csv
    
    if [ ! -f "$PROJECT_DIR/data/models/ensemble_model.pkl" ]; then
        log_error "模型训练失败"
        exit 1
    fi
    
    log_success "模型训练完成"
}

test_run() {
    log_step 9 "测试运行"
    
    read -p "是否运行测试？(y/n): " choice
    if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
        return
    fi
    
    log_info "运行测试分析..."
    $PYTHON_CMD -m src.analysis.query --top 5
    log_success "测试完成"
}

configure_slack() {
    log_step 10 "配置 Slack（可选）"
    
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
    echo ""
    echo "=========================================="
    log_success "安装完成！"
    echo "=========================================="
    echo ""
    echo "项目目录: $PROJECT_DIR"
    echo ""
    echo "常用命令："
    echo "  cd $PROJECT_DIR"
    echo ""
    echo "  # 每日分析"
    echo "  python3 -m src.analysis"
    echo ""
    echo "  # 股票查询"
    echo "  python3 -m src.analysis.query 600519"
    echo "  python3 -m src.analysis.query --top 10"
    echo ""
    echo "  # 重新训练"
    echo "  python3 -m src.models"
    echo ""
    echo "  # 回测验证"
    echo "  python3 -m src.backtest"
    echo ""
    echo "定时任务（可选）："
    echo "  crontab -e"
    echo "  添加："
    echo "  45 9 * * 1-5 /bin/bash $PROJECT_DIR/scripts/daily_report.sh"
    echo "  0 10 * * 1-5 /bin/bash $PROJECT_DIR/scripts/daily_decision.sh"
    echo ""
    echo "Slack 推送（可选）："
    echo "  编辑 .env 配置 SLACK_BOT_TOKEN"
    echo ""
}

main() {
    parse_args "$@"
    
    echo ""
    echo "=========================================="
    echo "  Topaz-Next 一键配置"
    echo "=========================================="
    echo ""
    
    detect_mode
    check_python
    check_pip
    detect_network
    clone_repo
    install_deps
    setup_dirs
    fetch_data
    train_model
    test_run
    configure_slack
    show_complete
}

main "$@"