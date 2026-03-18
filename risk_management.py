#!/usr/bin/env python3
"""
风控优化模块
因子正交化 + 风险预算 + 仓位优化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings('ignore')


class FactorOrthogonalizer:
    """因子正交化 - 消除共线性"""
    
    def __init__(self):
        self.orthogonal_factors = []
        self.rotation_matrix = None
        self.mean = None
        self.std = None
    
    def fit(self, factors: pd.DataFrame, method: str = 'gram_schmidt') -> 'FactorOrthogonalizer':
        """
        拟合正交化变换
        
        Args:
            factors: 因子DataFrame
            method: 正交化方法 ('gram_schmidt', 'pca', 'cholesky')
        
        Returns:
            self
        """
        # 标准化
        self.mean = factors.mean()
        self.std = factors.std() + 1e-8
        factors_std = (factors - self.mean) / self.std
        
        if method == 'gram_schmidt':
            # Gram-Schmidt正交化
            self.rotation_matrix = self._gram_schmidt(factors_std.values)
        elif method == 'pca':
            # PCA正交化
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(len(factors.columns), factors.shape[0]))
            pca.fit(factors_std)
            self.rotation_matrix = pca.components_.T
        elif method == 'cholesky':
            # Cholesky分解
            cov = factors_std.cov()
            self.rotation_matrix = np.linalg.cholesky(cov).T
        
        return self
    
    def transform(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        应用正交化变换
        
        Args:
            factors: 原始因子
        
        Returns:
            正交化后的因子
        """
        factors_std = (factors - self.mean) / self.std
        orthogonal = np.dot(factors_std.values, self.rotation_matrix)
        
        return pd.DataFrame(orthogonal, index=factors.index, 
                           columns=[f'orth_{i}' for i in range(orthogonal.shape[1])])
    
    def fit_transform(self, factors: pd.DataFrame, method: str = 'gram_schmidt') -> pd.DataFrame:
        """拟合并变换"""
        self.fit(factors, method)
        return self.transform(factors)
    
    def _gram_schmidt(self, A: np.ndarray) -> np.ndarray:
        """Gram-Schmidt正交化"""
        n, m = A.shape
        Q = np.zeros((n, m))
        R = np.zeros((m, m))
        
        for j in range(m):
            v = A[:, j]
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v = v - R[i, j] * Q[:, i]
            R[j, j] = np.linalg.norm(v)
            Q[:, j] = v / (R[j, j] + 1e-10)
        
        return Q


class RiskBudget:
    """风险预算 - 控制各因子风险暴露"""
    
    def __init__(self):
        self.factor_cov = None
        self.specific_risk = None
        self.risk_budgets = {}
    
    def fit(self, returns: pd.DataFrame, factors: pd.DataFrame) -> 'RiskBudget':
        """
        估计风险模型
        
        Args:
            returns: 股票收益
            factors: 因子暴露
        
        Returns:
            self
        """
        # 因子协方差矩阵
        self.factor_cov = factors.cov()
        
        # 特质风险（残差）
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(factors, returns)
        residuals = returns - model.predict(factors)
        self.specific_risk = np.diag(residuals.var())
        
        return self
    
    def calculate_risk(self, weights: np.ndarray, factor_exposures: np.ndarray) -> Dict:
        """
        计算组合风险
        
        Args:
            weights: 持仓权重
            factor_exposures: 因子暴露
        
        Returns:
            风险分解
        """
        # 因子风险
        factor_risk = np.sqrt(
            factor_exposures.T @ self.factor_cov @ factor_exposures
        )
        
        # 特质风险
        specific_risk = np.sqrt(
            weights @ self.specific_risk @ weights
        )
        
        # 总风险
        total_risk = np.sqrt(factor_risk**2 + specific_risk**2)
        
        return {
            'total_risk': total_risk,
            'factor_risk': factor_risk,
            'specific_risk': specific_risk,
            'factor_contribution': factor_risk / total_risk if total_risk > 0 else 0
        }
    
    def set_risk_budget(self, factor_budgets: Dict[str, float]):
        """
        设置风险预算
        
        Args:
            factor_budgets: 各因子风险预算 {因子名: 预算比例}
        """
        self.risk_budgets = factor_budgets
    
    def check_budget(self, factor_exposures: np.ndarray) -> Tuple[bool, Dict]:
        """
        检查是否超过风险预算
        
        Args:
            factor_exposures: 当前因子暴露
        
        Returns:
            (是否合规, 风险详情)
        """
        if not self.risk_budgets:
            return True, {}
        
        # 计算各因子风险贡献
        marginal_risk = self.factor_cov @ factor_exposures
        risk_contribution = factor_exposures * marginal_risk
        
        # 检查预算
        violations = {}
        for i, factor in enumerate(self.risk_budgets):
            if factor in self.factor_cov.index:
                idx = list(self.factor_cov.index).index(factor)
                contribution = risk_contribution[idx]
                budget = self.risk_budgets[factor]
                
                if abs(contribution) > budget:
                    violations[factor] = {
                        'contribution': contribution,
                        'budget': budget,
                        'excess': abs(contribution) - budget
                    }
        
        return len(violations) == 0, violations


class PortfolioOptimizer:
    """仓位优化器"""
    
    def __init__(self):
        self.risk_budget = RiskBudget()
    
    def optimize(self, 
                 expected_returns: np.ndarray,
                 factor_exposures: np.ndarray,
                 factor_cov: np.ndarray,
                 specific_risk: np.ndarray,
                 risk_aversion: float = 1.0,
                 max_weight: float = 0.1,
                 min_weight: float = -0.05) -> np.ndarray:
        """
        均值-方差优化
        
        Args:
            expected_returns: 预期收益
            factor_exposures: 因子暴露矩阵
            factor_cov: 因子协方差矩阵
            specific_risk: 特质风险对角矩阵
            risk_aversion: 风险厌恶系数
            max_weight: 单只股票最大权重
            min_weight: 单只股票最小权重（负数为做空）
        
        Returns:
            最优权重
        """
        n = len(expected_returns)
        
        # 简化版：使用风险平价
        # 计算各资产风险
        asset_risk = np.sqrt(np.diag(
            factor_exposures @ factor_cov @ factor_exposures.T + specific_risk
        ))
        
        # 风险平价权重（风险倒数加权）
        inv_risk = 1 / (asset_risk + 1e-8)
        risk_parity_weights = inv_risk / inv_risk.sum()
        
        # 考虑预期收益调整
        # 夏普比例调整
        sharpe = expected_returns / (asset_risk + 1e-8)
        adjusted_weights = risk_parity_weights * (1 + 0.5 * sharpe / sharpe.std())
        
        # 标准化
        weights = adjusted_weights / adjusted_weights.sum()
        
        # 限制权重范围
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()  # 重新标准化
        
        return weights
    
    def risk_parity(self, n_assets: int) -> np.ndarray:
        """
        风险平价权重
        
        Args:
            n_assets: 资产数量
        
        Returns:
            等风险贡献权重
        """
        return np.ones(n_assets) / n_assets


class PositionSizer:
    """仓位管理"""
    
    def __init__(self, 
                 max_position: float = 0.1,
                 max_sector: float = 0.3,
                 max_turnover: float = 0.2):
        self.max_position = max_position
        self.max_sector = max_sector
        self.max_turnover = max_turnover
    
    def calculate_position(self,
                          signal: float,
                          volatility: float,
                          target_volatility: float = 0.15) -> float:
        """
        根据波动率调整仓位
        
        Args:
            signal: 信号强度 [-1, 1]
            volatility: 股票波动率
            target_volatility: 目标波动率
        
        Returns:
            建议仓位
        """
        # 波动率调整
        vol_adjustment = target_volatility / (volatility + 1e-8)
        
        # 信号驱动的仓位
        position = signal * min(vol_adjustment, 1.0)
        
        # 限制最大仓位
        position = np.clip(position, -self.max_position, self.max_position)
        
        return position
    
    def apply_constraints(self,
                         weights: np.ndarray,
                         sectors: List[str] = None) -> np.ndarray:
        """
        应用仓位约束
        
        Args:
            weights: 原始权重
            sectors: 股票所属行业
        
        Returns:
            调整后的权重
        """
        # 单只股票限制
        weights = np.clip(weights, -self.max_position, self.max_position)
        
        # 行业限制
        if sectors:
            sector_weights = {}
            for i, sector in enumerate(sectors):
                sector_weights[sector] = sector_weights.get(sector, 0) + abs(weights[i])
            
            for i, sector in enumerate(sectors):
                if sector_weights[sector] > self.max_sector:
                    scale = self.max_sector / sector_weights[sector]
                    weights[i] *= scale
        
        # 标准化
        weights = weights / (np.abs(weights).sum() + 1e-8)
        
        return weights


class RiskManager:
    """风险管理器 - 整合所有风控模块"""
    
    def __init__(self):
        self.orthogonalizer = FactorOrthogonalizer()
        self.risk_budget = RiskBudget()
        self.optimizer = PortfolioOptimizer()
        self.sizer = PositionSizer()
    
    def process_signals(self,
                       signals: pd.DataFrame,
                       factors: pd.DataFrame,
                       returns: pd.DataFrame = None) -> pd.DataFrame:
        """
        处理信号并生成风控调整后的仓位
        
        Args:
            signals: 原始信号
            factors: 因子暴露
            returns: 历史收益（可选）
        
        Returns:
            调整后的仓位
        """
        # 1. 因子正交化
        factors_orth = self.orthogonalizer.fit_transform(factors)
        
        # 2. 计算波动率
        volatility = signals.std(axis=1) if signals.shape[1] > 1 else signals.abs()
        
        # 3. 计算仓位
        positions = pd.DataFrame(index=signals.index, columns=signals.columns)
        
        for col in signals.columns:
            signal = signals[col]
            vol = volatility[col] if col in volatility else 0.2
            positions[col] = self.sizer.calculate_position(signal, vol)
        
        return positions
    
    def check_risk(self, 
                   positions: np.ndarray,
                   factor_exposures: np.ndarray) -> Tuple[bool, Dict]:
        """
        检查组合风险
        
        Args:
            positions: 持仓
            factor_exposures: 因子暴露
        
        Returns:
            (是否合规, 风险详情)
        """
        # 风险预算检查
        budget_ok, violations = self.risk_budget.check_budget(factor_exposures)
        
        # 总风险检查
        risk = self.risk_budget.calculate_risk(positions, factor_exposures)
        
        return budget_ok, {
            'violations': violations,
            'risk': risk
        }


def main():
    """测试风控模块"""
    print("风控模块测试")
    print("="*60)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    n_factors = 10
    
    factors = pd.DataFrame(
        np.random.randn(n_samples, n_factors),
        columns=[f'factor_{i}' for i in range(n_factors)]
    )
    
    # 测试正交化
    orth = FactorOrthogonalizer()
    factors_orth = orth.fit_transform(factors)
    
    print(f"原始因子相关性矩阵:")
    print(factors.corr().iloc[:3, :3])
    print(f"\n正交化后相关性矩阵:")
    print(factors_orth.corr().iloc[:3, :3])
    
    # 测试仓位管理
    sizer = PositionSizer()
    signal = 0.5
    volatility = 0.3
    position = sizer.calculate_position(signal, volatility)
    print(f"\n信号: {signal}, 波动率: {volatility:.2f}")
    print(f"建议仓位: {position:.2%}")


if __name__ == '__main__':
    main()