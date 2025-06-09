import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 9999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df = pd.read_csv('BTC_HOURLY.csv') # get data

df = df[['datetime', 'close']]
df['chg'] = df['close'].pct_change()

def backtesting(window, threshold):
    df['ma'] = df['close'].rolling(window).mean()
    df['sd'] = df['close'].rolling(window).std()
    df['z'] = (df['close'] - df['ma']) / df['sd']

    for i in range(len(df)):
        if df.loc[i, 'z'] > threshold:
            df.loc[i, 'pos'] = 1
        else:
            df.loc[i, 'pos'] = 0

    df['pos_t-1'] = df['pos'].shift(1)
    df['pnl'] = df['pos_t-1'] * df['chg']
    df['cumu'] = df['pnl'].cumsum()
    df['dd'] = df['cumu'].cummax() - df['cumu']
    df['bnh_cumu'] = df['chg'].cumsum()

    sharpe = round(df['pnl'].mean() / df['pnl'].std() * np.sqrt(365), 2)
    annual_return = round(df['pnl'].mean() * 365, 2)
    mdd = df['dd'].max()
    calmar = round(annual_return / mdd, 2)

    # print(df)
    print(window, threshold, 'sharpe', sharpe, 'calmar', calmar)

    return pd.Series([window, threshold, sharpe], index=['window', 'threshold', 'sharpe'])

# optimisation zone ##############################
window_list = np.arange(10, 100, 10)
threshold_list = np.arange(0, 2.5, 0.25)

result_list = [] # define result_list

for window in window_list:
    for threshold in threshold_list:
        result_list.append(backtesting(window, threshold))

print(result_list)
result_df = pd.DataFrame(result_list)

result_df = result_df.sort_values(by='sharpe', ascending=False)
print(result_df)

data_table = result_df.pivot(index='window', columns='threshold', values='sharpe')
print(data_table)
sns.heatmap(data_table, annot=True, cmap='Greens')
plt.show()
##################################################

# # backtesting zone ########################################
# backtesting(60, 0.75)
# fig = px.line(df, x='Date', y=['cumu', 'bnh_cumu'])
# fig.show()
# ##################################################