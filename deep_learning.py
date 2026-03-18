#!/usr/bin/env python3
"""
深度学习模型 - LSTM/Transformer
专业量化时序预测架构
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 检查深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，深度学习模型不可用")


if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """LSTM模型"""
        
        def __init__(self, input_size: int, hidden_size: int = 64, 
                     num_layers: int = 2, dropout: float = 0.2,
                     bidirectional: bool = True):
            super(LSTMModel, self).__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
            
            # 注意力层
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1)
            )
            
            # 输出层
            fc_input = hidden_size * 2 if bidirectional else hidden_size
            self.fc = nn.Sequential(
                nn.Linear(fc_input, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            # LSTM
            lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
            
            # 注意力
            attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
            context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]
            
            # 输出
            out = self.fc(context)
            return out
    
    class TransformerModel(nn.Module):
        """Transformer模型"""
        
        def __init__(self, input_size: int, d_model: int = 64, 
                     nhead: int = 4, num_layers: int = 2,
                     dropout: float = 0.2):
            super(TransformerModel, self).__init__()
            
            self.input_projection = nn.Linear(input_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, dropout)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=128,
                dropout=dropout,
                batch_first=True
            )
            
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            self.fc = nn.Sequential(
                nn.Linear(d_model, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            x = self.input_projection(x)
            x = self.positional_encoding(x)
            x = self.transformer(x)
            x = x[:, -1, :]  # 取最后一个时间步
            out = self.fc(x)
            return out
    
    class PositionalEncoding(nn.Module):
        """位置编码"""
        
        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=dropout)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)
        
        def forward(self, x):
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


class DeepLearningPredictor:
    """深度学习预测器"""
    
    def __init__(self, model_dir: str = '.'):
        self.model_dir = model_dir
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None
        self.model_type = None
        self.feature_cols = []
        self.sequence_length = 20
        self.model_status = {'trained': False}
        
        # 加载已有模型
        self._load_model()
    
    def _load_model(self):
        """加载已保存的模型"""
        model_path = os.path.join(self.model_dir, 'dl_model.pth')
        status_path = os.path.join(self.model_dir, 'dl_model_status.json')
        
        if TORCH_AVAILABLE and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model_type = checkpoint.get('model_type', 'lstm')
                self.feature_cols = checkpoint.get('feature_cols', [])
                self.sequence_length = checkpoint.get('sequence_length', 20)
                
                # 重建模型
                input_size = len(self.feature_cols)
                if self.model_type == 'lstm':
                    self.model = LSTMModel(input_size=input_size)
                else:
                    self.model = TransformerModel(input_size=input_size)
                
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                if os.path.exists(status_path):
                    with open(status_path, 'r') as f:
                        self.model_status = json.load(f)
                
                print(f"已加载深度学习模型: {self.model_type}")
            except Exception as e:
                print(f"加载模型失败: {e}")
    
    def prepare_sequences(self, df: pd.DataFrame, feature_cols: List[str], 
                          target_col: str = 'target', 
                          sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备序列数据
        
        Args:
            df: 数据框
            feature_cols: 特征列
            target_col: 目标列
            sequence_length: 序列长度
        
        Returns:
            X: [n_samples, sequence_length, n_features]
            y: [n_samples]
        """
        X, y = [], []
        
        # 标准化特征
        features = df[feature_cols].values
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0) + 1e-8
        features = (features - mean) / std
        
        # 替换NaN
        features = np.nan_to_num(features, 0)
        
        # 按股票分组生成序列
        for code in df['code'].unique():
            code_data = df[df['code'] == code].sort_values('date')
            code_features = features[code_data.index]
            code_target = code_data[target_col].values
            
            for i in range(sequence_length, len(code_data)):
                X.append(code_features[i-sequence_length:i])
                y.append(code_target[i])
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, feature_cols: List[str],
              model_type: str = 'lstm',
              target_col: str = 'target',
              sequence_length: int = 20,
              epochs: int = 50,
              batch_size: int = 64,
              learning_rate: float = 0.001,
              validation_split: float = 0.2):
        """
        训练深度学习模型
        
        Args:
            df: 训练数据
            feature_cols: 特征列
            model_type: 模型类型 ('lstm' 或 'transformer')
            target_col: 目标列
            sequence_length: 序列长度
            epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            validation_split: 验证集比例
        """
        if not TORCH_AVAILABLE:
            print("❌ PyTorch未安装，无法训练深度学习模型")
            return False
        
        print("\n" + "="*60)
        print(f"训练深度学习模型: {model_type.upper()}")
        print("="*60)
        
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.model_type = model_type
        
        # 准备数据
        print("准备序列数据...")
        X, y = self.prepare_sequences(df, feature_cols, target_col, sequence_length)
        print(f"序列数: {len(X):,}, 特征数: {len(feature_cols)}")
        
        # 分割训练/验证集
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"训练集: {len(X_train):,}, 验证集: {len(X_val):,}")
        
        # 转为Tensor
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        
        # 创建模型
        input_size = len(feature_cols)
        if model_type == 'lstm':
            self.model = LSTMModel(input_size=input_size, hidden_size=64, 
                                   num_layers=2, bidirectional=True)
        else:
            self.model = TransformerModel(input_size=input_size, d_model=64,
                                          nhead=4, num_layers=2)
        
        self.model.to(self.device)
        
        # 损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 训练
        train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                  batch_size=batch_size, shuffle=True)
        
        best_val_acc = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                
                # 计算准确率
                val_pred = (torch.sigmoid(val_outputs) > 0.5).float()
                val_acc = (val_pred == y_val).float().mean().item()
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 记录最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_state = self.model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, "
                      f"Val Acc: {val_acc:.4f}")
        
        # 加载最佳模型
        self.model.load_state_dict(best_state)
        
        print(f"\n最佳验证准确率: {best_val_acc:.4f} (Epoch {best_epoch})")
        
        # 保存模型
        self._save_model(best_val_acc)
        
        return True
    
    def _save_model(self, val_acc: float):
        """保存模型"""
        model_path = os.path.join(self.model_dir, 'dl_model.pth')
        
        torch.save({
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length,
            'val_acc': val_acc
        }, model_path)
        
        # 保存状态
        self.model_status = {
            'trained': True,
            'model_type': self.model_type,
            'val_acc': val_acc,
            'sequence_length': self.sequence_length,
            'n_features': len(self.feature_cols),
            'train_time': datetime.now().isoformat()
        }
        
        status_path = os.path.join(self.model_dir, 'dl_model_status.json')
        with open(status_path, 'w') as f:
            json.dump(self.model_status, f, indent=2)
        
        print(f"模型已保存: {model_path}")
    
    def predict(self, df: pd.DataFrame, code: str) -> Dict:
        """
        预测单只股票
        
        Args:
            df: 历史数据
            code: 股票代码
        
        Returns:
            预测结果字典
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {'error': '模型不可用'}
        
        try:
            # 筛选该股票数据
            code_data = df[df['code'] == code].sort_values('date').tail(self.sequence_length)
            
            if len(code_data) < self.sequence_length:
                return {'error': '数据不足'}
            
            # 准备特征
            features = code_data[self.feature_cols].values
            mean = np.nanmean(features, axis=0)
            std = np.nanstd(features, axis=0) + 1e-8
            features = (features - mean) / std
            features = np.nan_to_num(features, 0)
            
            # 预测
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(X)
                prob = torch.sigmoid(output).item()
            
            return {
                'code': code,
                'method': f'dl_{self.model_type}',
                'probability': prob,
                'prediction': '涨' if prob > 0.5 else '跌',
                'confidence': max(prob, 1 - prob)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_status(self) -> Dict:
        """获取模型状态"""
        return self.model_status


def main():
    """测试深度学习模型"""
    if not TORCH_AVAILABLE:
        print("请先安装PyTorch: pip install torch")
        return
    
    print("深度学习模型可用")
    print(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")


if __name__ == '__main__':
    main()