import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

def run_portfolio_analysis(processed_data_path='data/processed/adj_close.csv',
                           tsla_forecast_path='data/processed/tsla_12month_forecast.csv',
                           tickers=['TSLA', 'BND', 'SPY']):

    try:
        data = pd.read_csv(processed_data_path, index_col=0, parse_dates=True)
        if isinstance(data.columns, pd.MultiIndex):
            new_cols = []
            for col_tuple in data.columns:
                if 'TSLA' in col_tuple:
                    new_cols.append('TSLA')
                elif 'BND' in col_tuple:
                    new_cols.append('BND')
                elif 'SPY' in col_tuple:
                    new_cols.append('SPY')
                else:
                    new_cols.append('_'.join(map(str, col_tuple)).replace('UNNAMED: ', '').replace('_LEVEL_2', ''))
            data.columns = new_cols
            data = data[tickers] 
        print(f"Historical data loaded from {processed_data_path}")
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return


    tsla_forecast_series = None
    try:
        tsla_forecast_series = pd.read_csv(tsla_forecast_path, index_col=0, parse_dates=True).iloc[:, 0]
        print(f"TSLA forecast loaded from {tsla_forecast_path}")
    except Exception as e:
        print(f"Error loading TSLA forecast: {e}")
        print("Using historical TSLA mean return as a fallback for TSLA expected return.")
        tsla_forecast_series = data['TSLA']

    # Calculate daily returns for historical data
    daily_returns = data.pct_change().dropna()


    if not tsla_forecast_series.empty and tsla_forecast_series.name == 'TSLA':
        # Calculate daily returns from the forecasted prices
        tsla_forecast_daily_returns = tsla_forecast_series.pct_change().dropna()
        tsla_expected_daily_return = tsla_forecast_daily_returns.mean()
        tsla_expected_annual_return = (1 + tsla_expected_daily_return)**252 - 1
    else:
        tsla_expected_annual_return = daily_returns['TSLA'].mean() * 252
        print("Warning: TSLA forecast not fully utilized; falling back to historical mean for expected return.")

    bnd_expected_annual_return = daily_returns['BND'].mean() * 252
    spy_expected_annual_return = daily_returns['SPY'].mean() * 252

    expected_returns_array = np.array([tsla_expected_annual_return, bnd_expected_annual_return, spy_expected_annual_return])

    print("\nExpected Annual Returns (for optimization):")
    for i, asset in enumerate(tickers):
        print(f"{asset}: {expected_returns_array[i]:.4f}")


    cov_matrix_annual = daily_returns[tickers].cov() * 252
    print("\nAnnualized Covariance Matrix:")
    print(cov_matrix_annual)


    num_portfolios = 50000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.sum(weights * expected_returns_array)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))

        risk_free_rate = 0.0 # Consistent with Task 1 and 4
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

        results[0,i] = portfolio_return
        results[1,i] = portfolio_volatility
        results[2,i] = sharpe_ratio

    results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe_Ratio'])

    plt.figure(figsize=(15, 8))
    plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe_Ratio'], cmap='viridis', s=10)
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Efficient Frontier with Simulated Portfolios')
    plt.xlabel('Portfolio Volatility (Annualized)')
    plt.ylabel('Portfolio Return (Annualized)')
    plt.grid(True)
    plt.savefig('reports/figures/efficient_frontier_simulated.png')
    plt.show()

    max_sharpe_portfolio = results_df.loc[results_df['Sharpe_Ratio'].idxmax()]
    max_sharpe_weights = weights_record[results_df['Sharpe_Ratio'].idxmax()]

    min_volatility_portfolio = results_df.loc[results_df['Volatility'].idxmin()]
    min_volatility_weights = weights_record[results_df['Volatility'].idxmin()]

    print("\n--- Key Portfolios ---")
    print("\nMaximum Sharpe Ratio Portfolio:")
    print(f"Return: {max_sharpe_portfolio['Return']:.4f}")
    print(f"Volatility: {max_sharpe_portfolio['Volatility']:.4f}")
    print(f"Sharpe Ratio: {max_sharpe_portfolio['Sharpe_Ratio']:.4f}")
    print("Weights:")
    for i, asset in enumerate(tickers):
        print(f"  {asset}: {max_sharpe_weights[i]:.4f}")

    print("\nMinimum Volatility Portfolio:")
    print(f"Return: {min_volatility_portfolio['Return']:.4f}")
    print(f"Volatility: {min_volatility_portfolio['Volatility']:.4f}")
    print(f"Sharpe Ratio: {min_volatility_portfolio['Sharpe_Ratio']:.4f}")
    print("Weights:")
    for i, asset in enumerate(tickers):
        print(f"  {asset}: {min_volatility_weights[i]:.4f}")

    # Plot the Efficient Frontier with marked portfolios
    plt.figure(figsize=(15, 8))
    plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe_Ratio'], cmap='viridis', s=10, label='Simulated Portfolios')
    plt.colorbar(label='Sharpe Ratio')
    plt.title('Efficient Frontier with Optimal Portfolios Highlighted')
    plt.xlabel('Portfolio Volatility (Annualized)')
    plt.ylabel('Portfolio Return (Annualized)')
    plt.grid(True)
    plt.scatter(max_sharpe_portfolio['Volatility'], max_sharpe_portfolio['Return'],
                marker='*', color='red', s=500, label='Max Sharpe Ratio')
    plt.scatter(min_volatility_portfolio['Volatility'], min_volatility_portfolio['Return'],
                marker='X', color='blue', s=500, label='Min Volatility')
    plt.legend(labelspacing=0.8)
    plt.savefig('reports/figures/efficient_frontier_highlighted.png')
    plt.show()

    print("\n--- Optimal Portfolio Recommendation ---")
    recommended_portfolio = max_sharpe_portfolio
    recommended_weights = max_sharpe_weights
    recommendation_justification = "The Maximum Sharpe Ratio Portfolio is recommended as it provides the highest expected return for each unit of risk taken, making it suitable for investors seeking efficient risk-adjusted growth. It balances potential returns with controlled volatility."

    print(f"\nRecommended Portfolio: {recommendation_justification}")
    print("\nSummary of Recommended Portfolio (Maximum Sharpe Ratio):")
    print(f"Expected Annual Return: {recommended_portfolio['Return']:.4f}")
    print(f"Expected Annual Volatility: {recommended_portfolio['Volatility']:.4f}")
    print(f"Sharpe Ratio: {recommended_portfolio['Sharpe_Ratio']:.4f}")
    print("Optimal Weights:")
    for i, asset in enumerate(tickers):
        print(f"  {asset}: {recommended_weights[i]:.4f}")
    print("\nPortfolio Analysis Complete.")

if __name__ == "__main__":
    run_portfolio_analysis()
