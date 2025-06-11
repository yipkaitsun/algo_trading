"""
Utility class for calculating trading strategy performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

class PerformanceMetrics:
    @staticmethod
    def calculate_metrics(data: pd.DataFrame) -> Dict[str, float]:
        df = data.copy()
        
        # Calculate PnL
        df['chg'] = df['close'].pct_change()
        df['pos_t-1'] = df['signal'].shift(1)
        df['pnl'] = df['pos_t-1'] * df['chg']
        
        # Calculate cumulative returns
        df['cumu'] = df['pnl'].cumsum()
        df['dd'] = df['cumu'].cummax() - df['cumu']
        df['bnh_cumu'] = df['chg'].cumsum()
        
        # Basic metrics
        sharpe = round(df['pnl'].mean() / df['pnl'].std() * np.sqrt(365*24), 2)
        annual_return = round(df['pnl'].mean() * 365, 2)
        mdd = df['dd'].max()
        calmar = round(annual_return / mdd, 2) if mdd != 0 else 0
        
        # Trade statistics
        trades = df[df['signal'] != 0].copy()
        if len(trades) > 0:
            win_rate = round(len(trades[trades['pnl'] > 0]) / len(trades) * 100, 2)
            profit_factor = round(
                abs(trades[trades['pnl'] > 0]['pnl'].sum() / trades[trades['pnl'] < 0]['pnl'].sum())
                if trades[trades['pnl'] < 0]['pnl'].sum() != 0 else float('inf'),
                2
            )
        else:
            win_rate = 0
            profit_factor = 0
        
        return {
            'sharpe_ratio': sharpe,
            'annual_return': annual_return,
            'max_drawdown': mdd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }

   
    

    