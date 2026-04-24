#!/bin/bash
# Topaz-Next 启动引导脚本
# 引导用户完成环境配置、依赖安装、模型训练

set -e

# 项目目录
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# pip 镜像源（中国大陆用户推荐）
MIRROR_URL=""

choose_mirror() {
    echo ""
    echo_info "下载依赖库可能较慢，可选择使用镜像源加速"
    echo ""
    read -p "是否在中国大陆？(y/n): " in_china
    
    if [ "$in_china" = "y" ] || [ "$in_china" = "Y" ]; then
        echo ""
        echo_info "请选择镜像源："
        echo "  1) 清华大学 (推荐，速度快)"
        echo "  2) 阿里云"
        echo "  3) 中科大"
        echo "  4) 不使用镜像"
        echo ""
        read -p "请输入选项 (1-4，默认1): " mirror_choice
        
        case "$mirror_choice" in
            1|"")
                MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
                echo_info "已选择: 清华大学镜像"
                ;;
            2)
                MIRROR_URL="https://mirrors.aliyun.com/pypi/simple"
                echo_info "已选择: 阿里云镜像"
                ;;
            3)
                MIRROR_URL="https://pypi.mirrors.ustc.edu.cn/simple"
                echo_info "已选择: 中科大镜像"
                ;;
            4)
                MIRROR_URL=""
                echo_info "不使用镜像，将从官方源下载"
                ;;
            *)
                MIRROR_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
                echo_info "默认使用: 清华大学镜像"
                ;;
        esac
    else
        MIRROR_URL=""
        echo_info "将使用官方 PyPI 源"
    fi
}

echo_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

echo_step() {
    echo ""
    echo "[$1] $2"
    echo "------------------------------------------------------------"
}

echo_success() {
    echo "✓ $1"
}

echo_error() {
    echo "✗ $1"
}

echo_info() {
    echo "  $1"
}

# 检查命令是否存在
check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# Step 1: 检查 Python
check_python() {
    echo_step 1 "检查 Python 环境"
    
    if check_command python3; then
        PYTHON_CMD="python3"
    elif check_command python; then
        PYTHON_CMD="python"
    else
        echo_error "未找到 Python"
        echo_info "请先安装 Python 3.8+"
        echo_info "安装方法："
        echo_info "  macOS: brew install python3"
        echo_info "  Ubuntu: sudo apt install python3"
        echo_info "  Windows: https://www.python.org/downloads/"
        exit 1
    fi
    
    VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    echo_success "Python 版本: $VERSION"
    
    # 检查版本是否 >= 3.8
    MAJOR=$(echo $VERSION | cut -d. -f1)
    MINOR=$(echo $VERSION | cut -d. -f2)
    
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
        echo_error "Python 版本过低，需要 3.8+"
        exit 1
    fi
}

# Step 2: 检查 pip
check_pip() {
    echo_step 2 "检查 pip"
    
    if check_command pip3; then
        PIP_CMD="pip3"
    elif check_command pip; then
        PIP_CMD="pip"
    else
        echo_error "未找到 pip"
        echo_info "正在安装 pip..."
        $PYTHON_CMD -m ensurepip --default-pip
        if check_command pip3; then
            PIP_CMD="pip3"
        else
            PIP_CMD="pip"
        fi
    fi
    
    echo_success "pip 已就绪"
}

# Step 3: 检查依赖
check_dependencies() {
    echo_step 3 "检查依赖库"
    
    DEPS=("pandas" "numpy" "requests" "lightgbm" "sklearn" "joblib")
    MISSING=()
    
    for dep in "${DEPS[@]}"; do
        if $PYTHON_CMD -c "import $dep" 2>/dev/null; then
            echo_success "$dep"
        else
            echo_error "$dep (未安装)"
            MISSING+=("$dep")
        fi
    done
    
    if [ ${#MISSING[@]} -gt 0 ]; then
        echo ""
        echo_info "缺少依赖: ${MISSING[*]}"
        read -p "是否自动安装？(y/n): " choice
        
        if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
            install_dependencies
        else
            echo_info "请手动安装: $PIP_CMD install pandas numpy requests scikit-learn lightgbm joblib"
            echo_info "中国大陆用户推荐使用镜像："
            echo_info "  $PIP_CMD install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas numpy requests scikit-learn lightgbm joblib"
            exit 1
        fi
    fi
}

# Step 4: 安装依赖
install_dependencies() {
    echo_step 4 "安装依赖库"
    
    # 选择镜像源
    choose_mirror
    
    echo_info "正在安装..."
    
    # 根据是否使用镜像构建命令
    if [ -n "$MIRROR_URL" ]; then
        echo_info "使用镜像源: $MIRROR_URL"
        $PIP_CMD install -i "$MIRROR_URL" pandas numpy requests scikit-learn lightgbm joblib
    else
        $PIP_CMD install pandas numpy requests scikit-learn lightgbm joblib
    fi
    
    if [ $? -eq 0 ]; then
        echo_success "依赖安装完成"
    else
        echo_error "安装失败"
        echo_info "请检查网络连接，或尝试手动安装："
        if [ -n "$MIRROR_URL" ]; then
            echo_info "  $PIP_CMD install -i $MIRROR_URL pandas numpy requests scikit-learn lightgbm joblib"
        else
            echo_info "  $PIP_CMD install pandas numpy requests scikit-learn lightgbm joblib"
        fi
        exit 1
    fi
}

# Step 5: 配置 Slack
configure_slack() {
    echo_step 5 "配置 Slack Token（可选）"
    
    echo_info "Slack 报告推送功能需要配置 Slack Bot Token"
    echo_info "如不需要可跳过，后续手动配置 .env 文件"
    echo ""
    
    read -p "是否配置 Slack？(y/n): " choice
    
    if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
        echo_info "已跳过 Slack 配置"
        return
    fi
    
    echo ""
    echo_info "获取 Slack Token 步骤："
    echo_info "1. 访问 https://api.slack.com/apps"
    echo_info "2. 创建新 App 或使用已有 App"
    echo_info "3. 在 OAuth & Permissions 添加 chat.postMessage 权限"
    echo_info "4. 安装到工作区，获取 Bot User OAuth Token"
    echo_info "   Token 格式: xoxb-xxxxxxxxxx-xxxxxxxxxx-..."
    echo ""
    
    read -p "请输入 Slack Bot Token: " token
    
    if [ -z "$token" ] || [[ ! "$token" =~ ^xoxb- ]]; then
        echo_error "Token 格式不正确，已跳过"
        return
    fi
    
    echo ""
    echo_info "Slack Channel 配置："
    echo_info "发送到频道: #channel-name 或 Cxxxxxxxx"
    echo_info "发送到私聊: Uxxxxxxxx"
    echo ""
    
    read -p "请输入 Channel ID（默认 U0AGVSHJ08Z）: " channel
    if [ -z "$channel" ]; then
        channel="U0AGVSHJ08Z"
    fi
    
    # 写入 .env 文件
    echo "# Topaz 环境配置" > "$PROJECT_DIR/.env"
    echo "SLACK_BOT_TOKEN=$token" >> "$PROJECT_DIR/.env"
    echo "SLACK_CHANNEL=$channel" >> "$PROJECT_DIR/.env"
    
    echo_success "已保存配置到 .env"
}

# Step 6: 获取数据
fetch_data() {
    echo_step 6 "获取沪深300历史数据"
    
    # 检查是否已有数据
    if [ -f "$PROJECT_DIR/csi300_full_history.csv" ]; then
        lines=$(wc -l < "$PROJECT_DIR/csi300_full_history.csv")
        if [ "$lines" -gt 1000 ]; then
            echo_success "已有历史数据文件 ($lines 条记录)"
            read -p "是否重新获取？(y/n): " choice
            if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
                return
            fi
        else
            echo_info "现有数据太少 ($lines 条)，重新获取"
        fi
    fi
    
    echo_info "正在获取数据（预计 3-5 分钟）..."
    
    if [ -f "$PROJECT_DIR/fetch_full_history.py" ]; then
        $PYTHON_CMD "$PROJECT_DIR/fetch_full_history.py"
        
        # 检查是否获取成功
        if [ -f "$PROJECT_DIR/csi300_full_history.csv" ]; then
            lines=$(wc -l < "$PROJECT_DIR/csi300_full_history.csv")
            if [ "$lines" -gt 1000 ]; then
                echo_success "数据获取完成 ($lines 条记录)"
            else
                echo_error "数据获取失败（数据太少: $lines 条）"
                echo_info "请检查网络连接后重新运行 setup.sh"
                exit 1
            fi
        else
            echo_error "数据文件未生成"
            echo_info "请检查网络连接后重新运行 setup.sh"
            exit 1
        fi
    else
        echo_error "未找到 fetch_full_history.py"
        exit 1
    fi
}

# Step 7: 训练模型
train_model() {
    echo_step 7 "训练 LightGBM 模型"
    
    # 检查是否已有模型
    if [ -f "$PROJECT_DIR/ensemble_model.pkl" ]; then
        echo_success "已有模型文件"
        read -p "是否重新训练？(y/n): " choice
        if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
            # 复制模型到默认位置
            cp "$PROJECT_DIR/ensemble_model.pkl" "$PROJECT_DIR/ensemble_model_csi300_latest.pkl"
            return
        fi
    fi
    
    # 检查数据文件是否存在
    if [ ! -f "$PROJECT_DIR/csi300_full_history.csv" ]; then
        echo_error "未找到训练数据文件 csi300_full_history.csv"
        echo_info "请先运行数据获取步骤"
        exit 1
    fi
    
    echo_info "正在训练模型（预计 2-5 分钟）..."
    
    if [ -f "$PROJECT_DIR/retrain_model.py" ]; then
        $PYTHON_CMD "$PROJECT_DIR/retrain_model.py"
        
        if [ -f "$PROJECT_DIR/ensemble_model.pkl" ]; then
            # 复制模型到默认位置
            cp "$PROJECT_DIR/ensemble_model.pkl" "$PROJECT_DIR/ensemble_model_csi300_latest.pkl"
            echo_success "模型训练完成"
        else
            echo_error "模型文件未生成"
            exit 1
        fi
    else
        echo_error "未找到 retrain_model.py"
        exit 1
    fi
}

# Step 8: 测试分析
test_analysis() {
    echo_step 8 "测试分析功能"
    
    read -p "是否运行测试分析？(y/n): " choice
    
    if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
        return
    fi
    
    echo_info "正在运行测试..."
    
    if [ -f "$PROJECT_DIR/daily_decision.py" ]; then
        $PYTHON_CMD "$PROJECT_DIR/daily_decision.py" 2>&1 | head -20
        echo_success "测试完成"
    else
        echo_error "未找到 daily_decision.py"
    fi
}

# Step 9: 定时任务配置
show_cron_prompt() {
    echo_step 9 "定时任务配置 Prompt"
    
    echo ""
    echo_info "请将以下内容发送给 OpenCode 或其他 Agent 工具："
    echo ""
    echo "${YELLOW}------------------------------------------------------------${NC}"
    echo "请帮我配置以下定时任务："
    echo ""
    echo "项目路径: $PROJECT_DIR"
    echo ""
    echo "任务列表："
    echo "1. 每个工作日 09:45 - 运行 A股分析报告"
    echo "2. 每个工作日 10:00 - 运行分析并发送 Slack"
    echo ""
    echo "Crontab 配置："
    echo "45 9 * * 1-5 /bin/bash $PROJECT_DIR/daily_report.sh"
    echo "0 10 * * 1-5 /bin/bash $PROJECT_DIR/daily_decision.sh"
    echo ""
    echo "注意事项："
    echo "- 确保 Python 环境路径正确"
    echo "- 非交易日（周末/节假日）脚本会自动跳过"
    echo "${YELLOW}------------------------------------------------------------${NC}"
}

# 显示完成信息
show_complete() {
    echo_header "配置完成"
    
    echo ""
    echo_success "环境配置完成！"
    echo ""
    echo_info "后续操作："
    echo_info "  训练模型: $PYTHON_CMD retrain_model.py"
    echo_info "  运行分析: $PYTHON_CMD daily_decision.py"
    echo_info "  更新数据: $PYTHON_CMD fetch_full_history.py"
    echo ""
    echo_info "配置文件: $PROJECT_DIR/.env"
    echo_info "模型文件: $PROJECT_DIR/ensemble_model_csi300_latest.pkl"
    echo ""
}

# 主流程
main() {
    echo_header "Topaz-Next 启动引导"
    
    echo_info "本脚本将引导您完成:"
    echo_info "1. 检查/安装 Python 和依赖"
    echo_info "2. 配置 Slack Token（可选）"
    echo_info "3. 获取股票数据"
    echo_info "4. 训练预测模型"
    echo_info "5. 测试分析功能"
    echo_info "6. 配置定时任务"
    echo ""
    
    read -p "按 Enter 开始..."
    
    cd "$PROJECT_DIR"
    
    check_python
    check_pip
    check_dependencies
    configure_slack
    fetch_data
    train_model
    test_analysis
    show_cron_prompt
    show_complete
}

# 运行
main