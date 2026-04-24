#!/bin/bash
# Topaz-Next 一键安装脚本
# 使用方法：
#   curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/install.sh | bash
#   curl -sSL https://raw.githubusercontent.com/nina-CP200/Topaz-Next/main/install.sh | bash -s -- --china

set -e

# 配置
REPO_URL="https://github.com/nina-CP200/Topaz-Next"
INSTALL_DIR="$HOME/topaz-next"
PYTHON_CMD=""
PIP_CMD=""
MIRROR_URL=""
USE_CHINA=false

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --china)
                USE_CHINA=true
                shift
                ;;
            --help)
                echo "Topaz-Next 一键安装脚本"
                echo ""
                echo "使用方法:"
                echo "  bash install.sh              # 自动检测环境"
                echo "  bash install.sh --china      # 强制使用中国镜像源"
                echo ""
                exit 0
                ;;
            *)
                shift
                ;;
        esac
    done
}

# 检查命令是否存在
check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# 检测 Python
check_python() {
    log_info "检测 Python 环境..."
    
    if check_command python3; then
        PYTHON_CMD="python3"
    elif check_command python; then
        PYTHON_CMD="python"
    else
        log_error "未找到 Python，请先安装 Python 3.8+"
        log_info "安装方法:"
        log_info "  macOS: brew install python3"
        log_info "  Ubuntu: sudo apt install python3"
        log_info "  Windows: https://www.python.org/downloads/"
        exit 1
    fi
    
    VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    log_success "Python 版本: $VERSION"
    
    # 检查版本 >= 3.8
    MAJOR=$(echo $VERSION | cut -d. -f1)
    MINOR=$(echo $VERSION | cut -d. -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
        log_error "Python 版本过低，需要 3.8+"
        exit 1
    fi
}

# 检测 pip
check_pip() {
    log_info "检测 pip..."
    
    if check_command pip3; then
        PIP_CMD="pip3"
    elif check_command pip; then
        PIP_CMD="pip"
    else
        log_info "安装 pip..."
        $PYTHON_CMD -m ensurepip --default-pip
        if check_command pip3; then
            PIP_CMD="pip3"
        else
            PIP_CMD="pip"
        fi
    fi
    
    log_success "pip 已就绪"
}

# 检测网络环境
detect_network() {
    log_info "检测网络环境..."
    
    if [ "$USE_CHINA" = true ]; then
        MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
        log_info "使用中国镜像源（清华大学）"
        return
    fi
    
    # 尝试访问 GitHub 判断网络环境
    if curl -s --max-time 5 "https://github.com" > /dev/null 2>&1; then
        log_info "网络环境：海外（GitHub 可访问）"
        MIRROR_URL=""
    else
        log_info "网络环境：中国大陆（GitHub 不可访问）"
        MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
    fi
}

# 克隆项目
clone_repo() {
    log_info "克隆项目..."
    
    if [ -d "$INSTALL_DIR" ]; then
        log_info "目录已存在: $INSTALL_DIR"
        read -p "是否删除并重新安装？(y/n): " choice
        if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
            rm -rf "$INSTALL_DIR"
        else
            log_info "跳过克隆，使用现有目录"
            cd "$INSTALL_DIR"
            return
        fi
    fi
    
    # 检查 git
    if ! check_command git; then
        log_error "未找到 git，请先安装 git"
        log_info "安装方法:"
        log_info "  macOS: brew install git"
        log_info "  Ubuntu: sudo apt install git"
        exit 1
    fi
    
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    log_success "项目克隆完成: $INSTALL_DIR"
}

# 安装依赖
install_deps() {
    log_info "安装依赖库..."
    
    DEPS="pandas numpy requests scikit-learn lightgbm joblib"
    
    if [ -n "$MIRROR_URL" ]; then
        log_info "使用镜像源: $MIRROR_URL"
        $PIP_CMD install -i "$MIRROR_URL" $DEPS
    else
        $PIP_CMD install $DEPS
    fi
    
    log_success "依赖安装完成"
}

# 获取数据
fetch_data() {
    log_info "获取沪深300历史数据（预计 3-5 分钟）..."
    
    $PYTHON_CMD fetch_full_history.py
    
    if [ ! -f "csi300_full_history.csv" ]; then
        log_error "数据获取失败"
        log_info "请检查网络连接后重新运行: python3 fetch_full_history.py"
        exit 1
    fi
    
    log_success "数据获取完成"
}

# 训练模型
train_model() {
    log_info "训练模型（预计 2-3 分钟）..."
    
    $PYTHON_CMD retrain_model.py
    
    if [ ! -f "ensemble_model.pkl" ]; then
        log_error "模型训练失败"
        exit 1
    fi
    
    # 复制模型到默认位置
    cp ensemble_model.pkl ensemble_model_csi300_latest.pkl
    
    log_success "模型训练完成"
}

# 测试运行
test_run() {
    log_info "测试运行..."
    
    $PYTHON_CMD daily_decision.py 2>&1 | head -20
    
    log_success "测试完成"
}

# 显示完成信息
show_complete() {
    echo ""
    echo "=========================================="
    log_success "安装完成！"
    echo "=========================================="
    echo ""
    echo "项目目录: $INSTALL_DIR"
    echo ""
    echo "运行分析："
    echo "  cd $INSTALL_DIR"
    echo "  python3 daily_decision.py"
    echo ""
    echo "查询股票："
    echo "  python3 query_stock.py 600519"
    echo ""
    echo "定时任务（可选）："
    echo "  crontab -e"
    echo "  添加以下内容："
    echo "  45 9 * * 1-5 /bin/bash $INSTALL_DIR/daily_report.sh"
    echo "  0 10 * * 1-5 /bin/bash $INSTALL_DIR/daily_decision.sh"
    echo ""
}

# 主流程
main() {
    parse_args "$@"
    
    echo ""
    echo "=========================================="
    echo "  Topaz-Next 一键安装"
    echo "=========================================="
    echo ""
    
    check_python
    check_pip
    detect_network
    clone_repo
    install_deps
    fetch_data
    train_model
    test_run
    show_complete
}

main "$@"