import pandas as pd

class CalculationUtils:
    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns and positions based on signals."""
        df = df.copy()
        df['chg'] = df['close'].pct_change()
        df['position'] = 0
        df['cumulative_return'] = 1.0

        for i in range(1, len(df)):
            df['position'].iloc[i] = 1 if df['signal'].iloc[i-1] == 1 else 0
            
            if df['position'].iloc[i-1] == 1:
                df['cumulative_return'].iloc[i] = df['cumulative_return'].iloc[i-1] * (1 + df['chg'].iloc[i])
            else:
                df['cumulative_return'].iloc[i] = df['cumulative_return'].iloc[i-1]
        
        return df
    @staticmethod
    def calculate_annualized_return(df: pd.DataFrame) -> float:
        if 'cumulative_return' not in df.columns:
            raise ValueError("DataFrame must contain 'cumulative_return' column")
        
        # Get the total return as last value minus first value
        total_return = df['cumulative_return'].iloc[-1] - df['cumulative_return'].iloc[0]

        years = len(df) / 365*24 
        
        # Calculate annualized return
        annualized_return = total_return / years
        
        return annualized_return * 100 
    