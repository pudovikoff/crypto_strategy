#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ETH Funding Rate Basic Analysis
This script loads and visualizes ETH funding rate data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("Libraries imported successfully")

def load_data(file_path='data/eth_funding_rates.csv'):
    """
    Load ETH funding rate data from CSV file.
    Returns a DataFrame with the data.
    """
    # Read funding rates data
    df = pd.read_csv(file_path)
    
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Set time as index
    df.set_index('time', inplace=True)
    
    print(f"Loaded ETH funding rate data with {len(df)} rows")
    print(f"Data ranges from {df.index.min()} to {df.index.max()}")
    
    return df

def analyze_funding_rates(df):
    """
    Perform basic analysis on funding rates data
    """
    # Calculate basic statistics
    mean_rate = df['fundingRate'].mean()
    median_rate = df['fundingRate'].median()
    std_rate = df['fundingRate'].std()
    min_rate = df['fundingRate'].min()
    max_rate = df['fundingRate'].max()
    
    print("\nETH Funding Rate Statistics:")
    print(f"Mean: {mean_rate:.8f}")
    print(f"Median: {median_rate:.8f}")
    print(f"Std Dev: {std_rate:.8f}")
    print(f"Min: {min_rate:.8f}")
    print(f"Max: {max_rate:.8f}")
    
    # Calculate percentage of positive vs negative funding rates
    positive_rate = (df['fundingRate'] > 0).mean() * 100
    negative_rate = (df['fundingRate'] < 0).mean() * 100
    zero_rate = (df['fundingRate'] == 0).mean() * 100
    
    print(f"\nPercentage of positive funding rates: {positive_rate:.2f}%")
    print(f"Percentage of negative funding rates: {negative_rate:.2f}%")
    print(f"Percentage of zero funding rates: {zero_rate:.2f}%")
    
    return {
        'mean': mean_rate,
        'median': median_rate,
        'std': std_rate,
        'min': min_rate,
        'max': max_rate,
        'positive_pct': positive_rate,
        'negative_pct': negative_rate,
        'zero_pct': zero_rate
    }

def create_features(df):
    """
    Create additional features for analysis
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Calculate moving averages
    data['MA7'] = data['fundingRate'].rolling(window=7).mean()
    data['MA30'] = data['fundingRate'].rolling(window=30).mean()
    data['MA90'] = data['fundingRate'].rolling(window=90).mean()
    
    # Calculate volatility (standard deviation)
    data['Volatility7'] = data['fundingRate'].rolling(window=7).std()
    data['Volatility30'] = data['fundingRate'].rolling(window=30).std()
    
    # Calculate rate of change
    data['ROC1'] = data['fundingRate'].pct_change(1)
    data['ROC7'] = data['fundingRate'].pct_change(7)
    
    # Drop NaN values
    data = data.dropna()
    
    return data

def plot_time_series(df):
    """
    Plot the funding rate time series
    """
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['fundingRate'], label='Funding Rate')
    plt.title('ETH Funding Rates Time Series', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Funding Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/start/eth_funding_rates_timeseries.png')
    print("Time series plot saved as 'eth_funding_rates_timeseries.png'")

def plot_moving_averages(df):
    """
    Plot the funding rate with moving averages
    """
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['fundingRate'], label='Funding Rate', alpha=0.5)
    plt.plot(df.index, df['MA7'], label='7-period MA', linewidth=2)
    plt.plot(df.index, df['MA30'], label='30-period MA', linewidth=2)
    plt.plot(df.index, df['MA90'], label='90-period MA', linewidth=2)
    plt.title('ETH Funding Rates with Moving Averages', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Funding Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/start/eth_funding_rates_ma.png')
    print("Moving averages plot saved as 'eth_funding_rates_ma.png'")

def plot_volatility(df):
    """
    Plot the funding rate volatility
    """
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Volatility7'], label='7-period Volatility')
    plt.plot(df.index, df['Volatility30'], label='30-period Volatility')
    plt.title('ETH Funding Rates Volatility', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Volatility (Std Dev)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/start/eth_funding_rates_volatility.png')
    print("Volatility plot saved as 'eth_funding_rates_volatility.png'")

def plot_histogram(df):
    """
    Plot histogram of funding rates
    """
    plt.figure(figsize=(12, 6))
    plt.hist(df['fundingRate'], bins=50, alpha=0.7, color='blue', label='Funding Rate Distribution')
    plt.axvline(x=df['fundingRate'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    plt.text(df['fundingRate'].mean(), plt.ylim()[1]*0.9, f'Mean: {df["fundingRate"].mean():.5f}', color='black', ha='center')
    # plt.text(df['fundingRate'].mode()[0], plt.ylim()[1]*0.8, f'Mode: {df["fundingRate"].mode()[0]:.5f}', color='black', ha='center')
    plt.title('Distribution of ETH Funding Rates', fontsize=16)
    plt.xlabel('Funding Rate', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/start/eth_funding_rates_histogram.png')
    print("Histogram plot saved as 'eth_funding_rates_histogram.png'")

def plot_daily_pattern(df):
    """
    Plot average funding rate by hour of day to identify patterns
    """
    # Extract hour from datetime index
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly.index.hour
    
    # Group by hour and calculate mean funding rate
    hourly_avg = df_hourly.groupby('hour')['fundingRate'].mean()
    
    plt.figure(figsize=(12, 6))
    plt.bar(hourly_avg.index, hourly_avg.values, alpha=0.7)
    plt.title('Average ETH Funding Rate by Hour of Day', fontsize=16)
    plt.xlabel('Hour of Day (UTC)', fontsize=14)
    plt.ylabel('Average Funding Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig('images/start/eth_funding_rates_daily_pattern.png')
    print("Daily pattern plot saved as 'eth_funding_rates_daily_pattern.png'")

def plot_weekly_pattern(df):
    """
    Plot average funding rate by day of week to identify patterns
    """
    # Extract day of week from datetime index
    df_daily = df.copy()
    df_daily['day_of_week'] = df_daily.index.dayofweek
    
    # Group by day of week and calculate mean funding rate
    daily_avg = df_daily.groupby('day_of_week')['fundingRate'].mean()
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    plt.figure(figsize=(12, 6))
    plt.bar(days, daily_avg.values, alpha=0.7)
    plt.title('Average ETH Funding Rate by Day of Week', fontsize=16)
    plt.xlabel('Day of Week', fontsize=14)
    plt.ylabel('Average Funding Rate', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/start/eth_funding_rates_weekly_pattern.png')
    print("Weekly pattern plot saved as 'eth_funding_rates_weekly_pattern.png'")

def main():
    """Main function to run the analysis"""
    # Load data
    data = load_data()
    
    # Analyze funding rates
    stats = analyze_funding_rates(data)
    
    # Create features
    data_with_features = create_features(data)
    
    # Create plots
    plot_time_series(data)
    plot_moving_averages(data_with_features)
    plot_volatility(data_with_features)
    plot_histogram(data)
    plot_daily_pattern(data)
    plot_weekly_pattern(data)
    
    print("\nAnalysis complete!")
    return data, data_with_features, stats

if __name__ == "__main__":
    main() 