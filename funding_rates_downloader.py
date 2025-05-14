import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from typing import Optional
import time

# Import fractal libraries
from fractal.loaders.binance import BinanceFundingLoader
from fractal.loaders import HyperliquidFundingRatesLoader

def get_binance_funding_rates(
    symbol: str = "BTCUSDT",
    days: int = 60,
    use_fixed_dates: bool = True
) -> pd.DataFrame:
    """
    Get historical funding rate data from Binance for a specified period
    
    Args:
        symbol: Trading pair
        days: Number of days to look back
        use_fixed_dates: If True, use fixed dates instead of relying on system time
        
    Returns:
        DataFrame with historical funding rate data
    """
    # Initialize Binance data loader
    loader = BinanceFundingLoader(symbol)
    
    # Calculate time range
    if use_fixed_dates:
        # Use fixed dates for the last two months (March-May 2024)
        end_time_dt = datetime.datetime(2024, 5, 13)
        start_time_dt = end_time_dt - timedelta(days=days)
    else:
        # Use system time
        end_time_dt = datetime.datetime.now()
        start_time_dt = end_time_dt - timedelta(days=days)
    
    print(f"Downloading funding rates for {symbol} from {start_time_dt.strftime('%Y-%m-%d')} to {end_time_dt.strftime('%Y-%m-%d')}")
    
    # Convert datetime to timestamp in milliseconds for the API
    end_time_ms = int(end_time_dt.timestamp() * 1000)
    start_time_ms = int(start_time_dt.timestamp() * 1000)
    
    # Use the get_funding_rates method with correct parameters
    # Note: Pass None for end_time to use current time, or timestamp for specific time
    funding_data = loader.get_funding_rates(
        ticker=symbol, 
        start_time=start_time_dt,  # This is used for filtering, not in the API call
        end_time=None  # Let the method use current time
    )
    
    # Convert to DataFrame
    funding_df = pd.DataFrame(funding_data)
    
    # Process the data
    if not funding_df.empty:
        # Set timestamp as index
        funding_df.set_index('fundingTime', inplace=True)
        
        # Filter by date range
        # funding_df = funding_df[funding_df.index >= start_time_dt]
        
        # Sort by timestamp
        funding_df.sort_index(inplace=True)
    
    print(f"Downloaded {len(funding_df)} funding rate records")
    
    return funding_df

def get_hyperliquid_funding_rates(
    symbol: str = "ETH",
    # days: int = 60,
    # use_fixed_dates: bool = True
) -> pd.DataFrame:
    """
    Get historical funding rate data from Hyperliquid for a specified period
    
    Args:
        symbol: Trading pair (without USDC suffix for Hyperliquid)
        
    Returns:
        DataFrame with historical funding rate data
    """
    # Calculate time range
    loader = HyperliquidFundingRatesLoader(symbol)


    funding_df = loader.read(with_run=True)
    funding_df.rename(columns={'rate': 'fundingRate'}, inplace=True)

    funding_df.to_csv(f"data/{symbol.lower()}_funding_rates.csv")

        
    print(f"Downloaded {len(funding_df)} funding rate records from Hyperliquid")
    
    return funding_df

def analyze_funding_rates(df: pd.DataFrame, symbol: str) -> None:
    """
    Analyze and print statistics about funding rates
    
    Args:
        df: DataFrame with funding rate data
        symbol: Trading pair symbol
    """
    if df.empty:
        print(f"No funding rate data available for {symbol}")
        return
    
    # Make sure we have the correct rate column
    rate_column = 'fundingRate'
    
    # Calculate statistics
    mean_rate = df[rate_column].mean()
    min_rate = df[rate_column].min()
    max_rate = df[rate_column].max()
    std_rate = df[rate_column].std()
    
    # Annualized funding rate (assuming 3 funding periods per day for Binance)
    annual_rate = mean_rate * 3 * 365 * 100  # Convert to percentage
    
    print(f"\nFunding Rate Analysis for {symbol}:")
    print(f"Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Number of records: {len(df)}")
    print(f"Mean funding rate: {mean_rate:.8f}")
    print(f"Min funding rate: {min_rate:.8f}")
    print(f"Max funding rate: {max_rate:.8f}")
    print(f"Standard deviation: {std_rate:.8f}")
    print(f"Annualized funding rate: {annual_rate:.2f}%")

def plot_funding_rates(df: pd.DataFrame, symbol: str) -> None:
    """
    Plot funding rates over time
    
    Args:
        df: DataFrame with funding rate data
        symbol: Trading pair symbol
    """
    if df.empty:
        print(f"No data to plot for {symbol}")
        return
    
    # Use the correct rate column
    rate_column = 'fundingRate'
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[rate_column], 'b-', label='Funding Rate')
    
    # Add horizontal line at zero
    plt.axhline(y=0, color='r', linestyle='--', label='Zero')
    
    # Calculate days range for title
    days_range = (df.index.max() - df.index.min()).days
    
    plt.title(f'Funding Rates for {symbol} ({days_range} Days: {df.index.min().strftime("%Y-%m-%d")} to {df.index.max().strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Funding Rate')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    filename = f"images/{symbol.lower()}_funding_rates.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Symbol and time period
    binance_symbol = "BTCUSDT"
    hyperliquid_symbol = "BTC"
    days = 60  # Last two months
    
    # Get Binance funding rates
    binance_funding_df = get_binance_funding_rates(symbol=binance_symbol, days=days, use_fixed_dates=True)
    
    # Save to CSV
    binance_csv_filename = f"{binance_symbol.lower()}_funding_rates.csv"
    binance_funding_df.to_csv(binance_csv_filename)
    print(f"Binance data saved to {binance_csv_filename}")
    
    # Analyze the Binance data
    analyze_funding_rates(binance_funding_df, binance_symbol)
    
    # Plot the Binance data
    try:
        plot_funding_rates(binance_funding_df, binance_symbol)
    except Exception as e:
        print(f"Error plotting Binance data: {e}")
    
    # Get Hyperliquid funding rates
    hyperliquid_funding_df = get_hyperliquid_funding_rates(symbol=hyperliquid_symbol, days=days, use_fixed_dates=True)
    
    # Save to CSV
    hyperliquid_csv_filename = f"hyperliquid_{hyperliquid_symbol.lower()}_funding_rates.csv"
    hyperliquid_funding_df.to_csv(hyperliquid_csv_filename)
    print(f"Hyperliquid data saved to {hyperliquid_csv_filename}")
    
    # Analyze the Hyperliquid data
    analyze_funding_rates(hyperliquid_funding_df, f"Hyperliquid {hyperliquid_symbol}")
    
    # Plot the Hyperliquid data
    try:
        plot_funding_rates(hyperliquid_funding_df, f"Hyperliquid {hyperliquid_symbol}")
    except Exception as e:
        print(f"Error plotting Hyperliquid data: {e}") 