"""
Utility class for calculating trading strategy performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict
import math
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
        sharpe = round(df['pnl'].mean() / df['pnl'].std() * np.sqrt(24*365), 2)
        annual_return = round(df['pnl'].mean() * 24*365, 2)
        
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

    
        
    @staticmethod
    def calculate_trading_metrics(
        data: pd.DataFrame,
        initial_capital=100,
        output_csv="/Users/admin/Project/algo_trading/data/trading_results.csv",
        save_csv="False",
        allow_fractional_shares=True,
        commission=0.0,
        slippage=0.0
    ):
        df = data.copy()
        prices = df['close']
        signals = df['signal']

        cash = initial_capital
        shares = 0
        holding = False
        equity = [initial_capital]
        trades = []
        buy_time = None
        buy_price = None

        trading_data = {
            'timestamp': [],
            'price': [],
            'signal': [],
            'shares': [],
            'cash': [],
            'equity': [],
            'position': [],
            'trade_type': [],
            'trade_pnl': []
        }

        for t in range(len(prices)):
            current_price = prices.iloc[t]
            current_signal = signals.iloc[t]
            trade_type = 'HOLD'
            trade_pnl = 0

            if current_signal == 1 and not holding:
                # Buy with all available cash
                buy_time = t
                buy_price = current_price * (1 + slippage)
                if allow_fractional_shares:
                    shares = cash / buy_price
                else:
                    shares = int(cash // buy_price)
                cost = shares * buy_price
                cash -= cost
                cash -= cost * commission
                holding = True
                trade_type = 'BUY'
            elif current_signal == 0 and holding:
                # Sell all shares
                sell_time = t
                sell_price = current_price * (1 - slippage)
                proceeds = shares * sell_price
                if buy_price is not None:
                    trade_pnl = proceeds - (shares * buy_price)
                    ROI = (sell_price - buy_price) / buy_price
                else:
                    trade_pnl = 0
                    ROI = 0
                cash += proceeds
                cash -= proceeds * commission
                holding = False
                trade_type = 'SELL'
                trades.append({
                    'buy_time': buy_time,
                    'buy_price': buy_price,
                    'sell_time': sell_time,
                    'sell_price': sell_price,
                    'ROI': ROI
                })
                shares = 0

            current_equity = cash + (shares * current_price)
            trading_data['timestamp'].append(df.index[t] if isinstance(df.index, pd.DatetimeIndex) else t)
            trading_data['price'].append(current_price)
            trading_data['signal'].append(current_signal)
            trading_data['shares'].append(shares)
            trading_data['cash'].append(cash)
            trading_data['equity'].append(current_equity)
            trading_data['position'].append('LONG' if holding else 'CASH')
            trading_data['trade_type'].append(trade_type)
            trading_data['trade_pnl'].append(trade_pnl)
            equity.append(current_equity)

        # Final liquidation if holding
        if holding and shares > 0:
            final_price = prices.iloc[-1]
            proceeds = shares * final_price * (1 - slippage)
            cash += proceeds
            cash -= proceeds * commission
            if buy_price is not None:
                trade_pnl = proceeds - (shares * buy_price)
                ROI = (final_price * (1 - slippage) - buy_price) / buy_price
            else:
                trade_pnl = 0
                ROI = 0
            trades.append({
                'buy_time': buy_time,
                'buy_price': buy_price,
                'sell_time': len(prices) - 1,
                'sell_price': final_price * (1 - slippage),
                'ROI': ROI
            })
            shares = 0
            current_equity = cash
            trading_data['timestamp'].append(df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else len(prices) - 1)
            trading_data['price'].append(final_price)
            trading_data['signal'].append(0)
            trading_data['shares'].append(shares)
            trading_data['cash'].append(cash)
            trading_data['equity'].append(current_equity)
            trading_data['position'].append('CASH')
            trading_data['trade_type'].append('FINAL_SELL')
            trading_data['trade_pnl'].append(trade_pnl)
            equity.append(current_equity)

        trading_df = pd.DataFrame(trading_data)
        if save_csv and output_csv:
            trading_df.to_csv(output_csv, index=False)

        total_pl = equity[-1] - initial_capital

        # Cumulative return (geometric)
        cumulative_return = np.prod([1 + trade['ROI'] for trade in trades]) - 1 if trades else 0

        # Sharpe/Sortino: Use log returns of equity curve
        equity_series = pd.Series(equity)
        returns = np.log(equity_series / equity_series.shift(1))
        returns = pd.Series(returns).dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        annualization_factor = math.sqrt(8760)  # hourly, 24/7

        sharpe_ratio = (mean_return / std_return) * annualization_factor if std_return != 0 else None
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (mean_return / downside_std) * annualization_factor if downside_std != 0 else None

        # Max drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / running_max
        max_drawdown = np.max(drawdown)
        return {
            'total_pl': total_pl,
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'trades': len(trades),
            'equity_curve': equity[-1],
            'trading_data': trading_df
        }