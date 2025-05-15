import os
import warnings
from typing import Dict, Any

import numpy as np
import pandas as pd
from datetime import datetime, UTC
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from ml_basis_strategy import MLBasisStrategy, MLBasisParams
from mb_hl_strategy import build_observations

warnings.filterwarnings('ignore')


class MLPredictor:
    def __init__(self, lookback_window: int = 24):
        self.lookback_window = lookback_window
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML model"""
        df = data.copy()
        print(df)
        # Validate that required columns exist
        required_columns = ['funding_rate', 'price']
        for col in required_columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in dataframe. Available columns: {df.columns.tolist()}")
                return pd.DataFrame()  # Return empty dataframe if required columns are missing
        
        # Create lagged features
        for i in range(1, self.lookback_window + 1):
            df[f'funding_rate_lag_{i}'] = df['funding_rate'].shift(i)
            df[f'price_lag_{i}'] = df['price'].shift(i)
        
        # Create rolling statistics
        df['funding_rate_mean'] = df['funding_rate'].rolling(window=self.lookback_window).mean()
        df['funding_rate_std'] = df['funding_rate'].rolling(window=self.lookback_window).std()
        df['price_mean'] = df['price'].rolling(window=self.lookback_window).mean()
        df['price_std'] = df['price'].rolling(window=self.lookback_window).std()
        
        # Drop NaN values
        df = df.dropna()
        return df
    
    def fit(self, data: pd.DataFrame):
        """Fit the model"""
        df = self.prepare_features(data)
        if len(df) == 0:
            print("Warning: No data available after preparing features. Model not fitted.")
            return
            
        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'funding_rate']]
        X = df[feature_cols]
        y = df['funding_rate']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
    def predict(self, data: pd.DataFrame) -> float:
        """Make prediction for next funding rate"""
        df = self.prepare_features(data)
        if len(df) == 0:
            return 0.0
            
        # Get latest data point
        latest = df.iloc[-1:]
        
        # Prepare features
        feature_cols = [col for col in latest.columns if col not in ['timestamp', 'funding_rate']]
        X = latest[feature_cols]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        return self.model.predict(X_scaled)[0]
        
    def __call__(self, data: pd.DataFrame) -> float:
        """Make the predictor callable directly"""
        return self.predict(data)


def build_grid(observations):
    """Build parameter grid for experiments"""
    # Prepare training data for ML model
    historical_data = pd.DataFrame({
        'timestamp': [obs.timestamp for obs in observations],
        'price': [obs.states['SPOT'].price for obs in observations],
        'funding_rate': [obs.states['HEDGE'].funding_rate for obs in observations]
    })

    # Create ML predictors with different lookback windows
    ml_predictors = {}
    for lookback in [12, 24, 48]:
        predictor = MLPredictor(lookback_window=lookback)
        predictor.fit(historical_data)
        ml_predictors[lookback] = predictor
    
    raw_grid = ParameterGrid({
        'MIN_LEVERAGE': np.arange(1, 12, 1).tolist(),
        'TARGET_LEVERAGE': np.arange(1, 12, 1).tolist(),
        'MAX_LEVERAGE': np.arange(1, 12, 1).tolist(),
        'EXECUTION_COST': [0.002],
        'INITIAL_BALANCE': [1_000_000],
        'ML_PREDICTION_THRESHOLD': [-0.0001, 0.0, 0.0001],  # Different thresholds for ML predictions
        'LOOKBACK_WINDOW': [12, 24, 48],  # Different lookback windows for ML model
    })

    # Create the full grid with appropriate ML predictors
    valid_grid = []
    for params in raw_grid:
        if round(params['MIN_LEVERAGE'], 1) < round(params['TARGET_LEVERAGE'], 1) < round(params['MAX_LEVERAGE'], 1):
            # Copy the parameters and add the correct predictor
            params_copy = params.copy()
            lookback = params_copy['LOOKBACK_WINDOW']
            # Use the whole predictor object, which is now callable
            params_copy['ml_predictor'] = ml_predictors[lookback]
            valid_grid.append(params_copy)
    
    print(f'Length of valid grid: {len(valid_grid)}')
    return valid_grid


if __name__ == '__main__':
    ticker: str = 'ETH'
    start_time = datetime(2023, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 1, 1, tzinfo=UTC)
    fidelity = '1h'
    experiment_name = f'ml_basis_{fidelity}_{ticker}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'

    # Define MLFlow configuration
    mlflow_uri = 'http://127.0.0.1:8080'
    if not mlflow_uri:
        raise ValueError("MLFLOW_URI isn't set.")

    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri=mlflow_uri,
        experiment_name=experiment_name,
    )

    # Build observations
    observations = build_observations(ticker, start_time, end_time, fidelity=fidelity)
    assert len(observations) > 0

    # Define experiment configuration
    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=MLBasisStrategy,
        backtest_observations=observations,
        window_size=24 * 30,  # 30 days window
        params_grid=build_grid(observations),
        debug=True
    )

    # Create and run pipeline
    pipeline: DefaultPipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    pipeline.run() 