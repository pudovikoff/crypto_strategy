import os
import sys
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

from funding_rate_analysis.eth_funding_rate_sign_prediction import LSTMModel
import torch

warnings.filterwarnings('ignore')


class MLPredictor:
    def __init__(self, lookback_window: int = 24):
        self.lookback_window = lookback_window
        # Initialize LSTM model for ML predictions

        # Load the fixed model weights
        project_root = os.path.abspath(os.path.join(os.path.dirname('__file__'), '..'))
        if project_root not in sys.path:
            sys.path.append(project_root)

        model_path = '/Users/adpudovnikov/Documents/Study/Crypto/crypto_strategy/model_weights/model_weights_24hr_20250518_235840.pth'
        
        if os.path.exists(model_path):
            # Initialize model with same architecture as in eth_funding_rate_sign_prediction.py
            input_size = 12  # Number of features used in original LSTM model
            hidden_size = 96
            num_layers = 3
            dropout = 0.3
            
            self.lstm_model = LSTMModel(input_size, hidden_size, num_layers, dropout)
            
            # Load the saved weights    
            self.lstm_model.load_state_dict(torch.load(model_path))
            self.lstm_model.eval()  # Set to evaluation mode
            
            print(f"Loaded LSTM model weights from {model_path}")
        else:
            print("No LSTM model weights found. Using stub predictor.")
            self.lstm_model = None
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Stub implementation - just returns the input data"""
        return data
    
    def fit(self, data: pd.DataFrame):
        """Stub implementation - does nothing"""
        print("Stub predictor: fit called (does nothing)")
        pass
        
    def predict(self, data: pd.DataFrame) -> float:
        """Stub implementation - randomly returns 0, 1, or 2"""
        # print(data)
        return float(np.random.choice([0, 1, 2]))
        
    def __call__(self, data: pd.DataFrame) -> float:
        """Make the predictor callable directly"""
        # print("I AM CALLED")
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
    for lookback in [ 24,]:
        predictor = MLPredictor(lookback_window=lookback)
        # predictor.fit(historical_data)
        ml_predictors[lookback] = predictor
    
    raw_grid = ParameterGrid({
        'MIN_LEVERAGE': np.arange(1, 12, 2).tolist(),
        'TARGET_LEVERAGE': np.arange(1, 12, 2).tolist(),
        'MAX_LEVERAGE': np.arange(1, 12, 2).tolist(),
        'EXECUTION_COST': [0.002],
        'INITIAL_BALANCE': [1_000_000],
        'ML_PREDICTION_THRESHOLD': [-0.0001, 0.0, 0.0001],  # Different thresholds for ML predictions
        'LOOKBACK_WINDOW': [ 24,],  # Different lookback windows for ML model
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

    # HyperliquidBasis.MAX_LEVERAGE = 45

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