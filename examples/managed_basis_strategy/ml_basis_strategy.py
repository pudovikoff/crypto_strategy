from typing import List, Optional, Callable
from datetime import datetime, UTC
from dataclasses import dataclass

import pandas as pd
import numpy as np

from fractal.core.base import Observation, Action, ActionToTake
from fractal.core.entities import UniswapV3LPGlobalState, HyperLiquidGlobalState
from fractal.core.entities import BaseHedgeEntity, BaseSpotEntity
from fractal.strategies.hyperliquid_basis import HyperliquidBasis
from fractal.strategies.basis_trading_strategy import BasisTradingStrategyHyperparams

@dataclass
class MLBasisParams(BasisTradingStrategyHyperparams):
    """
    Parameters for the MLBasis strategy.
    """
    EXECUTION_COST: float
    ML_PREDICTION_THRESHOLD: float = 0.0  # Threshold for ML predictions to trigger position changes
    ml_predictor: Optional[Callable[[pd.DataFrame], float]] = None  # Add ml_predictor to params

class MLBasisStrategy(HyperliquidBasis):
    """
    A modified version of HyperliquidBasis that uses ML predictions for funding rates
    to make trading decisions.
    """
    def __init__(
        self,
        params: MLBasisParams,
        debug: bool = False,
        **kwargs  # Add support for additional parameters
    ):
        """
        Initialize the MLBasis strategy.

        Args:
            params: Strategy parameters including ml_predictor
            debug: Whether to run in debug mode
            **kwargs: Additional parameters that might be passed by the pipeline
        """
        super().__init__(debug=debug, params=params, **kwargs)  # Pass additional parameters to parent
        self.ml_predictor = params['ml_predictor']  # Get ml_predictor from params dictionary
        self.current_position = 0

    def _get_historical_data(self, observation: Observation) -> pd.DataFrame:
        """
        Get historical data for ML prediction.
        """
        if self.observations_storage is None:
            return pd.DataFrame()
            
        # Get last 24 hours of observations
        last_24h = observation.timestamp - pd.Timedelta(hours=24)
        historical_observations = [
            obs for obs in self.observations_storage.get_observations()
            if obs.timestamp >= last_24h
        ]
        
        # Create DataFrame with historical data
        historical_data = pd.DataFrame({
            'timestamp': [obs.timestamp for obs in historical_observations],
            'price': [obs.states['SPOT'].price for obs in historical_observations],
            'funding_rate': [obs.states['HEDGE'].funding_rate for obs in historical_observations]
        })
        return historical_data

    def predict(self, *args, **kwargs):
        """
        Predict the actions to take based on the current state of the entities and ML predictions.
        """
        hedge: BaseHedgeEntity = self.get_entity('HEDGE')
        spot: BaseSpotEntity = self.get_entity('SPOT')
        
        # Get ML prediction for funding rate
        current_observation = kwargs.get('observation')
            
        historical_data = self._get_historical_data(current_observation)
        predicted_funding_rate = self.ml_predictor(historical_data)
        
        if predicted_funding_rate > self.params['ML_PREDICTION_THRESHOLD']:
            # Initial deposit if both balances are 0 and we predict positive funding rate
            if hedge.balance == 0 and spot.balance == 0:
                self._debug("Depositing initial funds into the strategy...")
                return self._deposit_into_strategy()
            # Rebalance if needed
            if hedge.balance == 0 and spot.balance > 0:
                self._debug(f"HEDGE balance is 0, but SPOT balance is {spot.balance}")
                return self._rebalance()
            if spot.balance > 0 and hedge.leverage > self.params['MAX_LEVERAGE'] or hedge.leverage < self.params['MIN_LEVERAGE']:
                self._debug(f"HEDGE leverage is {hedge.leverage}, rebalancing...")
                # spot_amount = spot.internal_state.amount
                # assert np.abs(hedge.size + spot_amount) / np.abs(hedge.size - spot_amount) <= 1e-6  # hedge.size ~= -spot_amount
                return self._rebalance()
        else:
            #If predicted funding rate is negative, exit position
            if spot.balance != 0:
                return self._exit_position()
        return []

    def _exit_position(self):
        """
        Exit position by selling spot and withdrawing from both entities.
        """
        hedge: BaseHedgeEntity = self.get_entity('HEDGE')
        spot: BaseSpotEntity = self.get_entity('SPOT')
        
        # Get current balances
        spot_balance = spot.balance
        hedge_balance = hedge.balance
        
        actions = []
        
        # Сначала закрываем хеджирующую позицию
        if hedge_balance > 0 and hedge.size != 0:
            actions.append(ActionToTake(entity_name='HEDGE', action=Action('open_position', {'amount_in_product': -hedge.size})))
        
        # Затем закрываем спотовую позицию
        if spot_balance > 0:
            actions.append(ActionToTake(entity_name='SPOT', action=Action('sell', {'amount_in_product': spot.internal_state.amount})))
        
        # Выводим средства после закрытия всех позиций
        if spot_balance > 0:
            actions.append(ActionToTake(entity_name='SPOT', action=Action('withdraw', {'amount_in_notional': lambda obj: obj.get_entity('SPOT').internal_state.cash})))
        
        if hedge_balance > 0:
            actions.append(ActionToTake(entity_name='HEDGE', action=Action('withdraw', {'amount_in_notional': lambda obj: obj.get_entity('HEDGE').internal_state.collateral})))
        
        return actions

# Example usage:
if __name__ == '__main__':
    # Example ML predictor function (replace with your actual ML model)
    def example_ml_predictor(historical_data: pd.DataFrame) -> float:
        # This is just an example - replace with your actual ML model
        return np.mean(historical_data['funding_rate'].tail(24))

    # Set up parameters
    params = MLBasisParams(
        MIN_LEVERAGE=1,
        MAX_LEVERAGE=8,
        TARGET_LEVERAGE=4,
        INITIAL_BALANCE=1_000_000,
        EXECUTION_COST=0.005,
        ML_PREDICTION_THRESHOLD=0.0,
        ml_predictor=example_ml_predictor
    )

    # Initialize strategy
    strategy = MLBasisStrategy(
        params=params,
        debug=True
    )

    # Run strategy (you'll need to provide observations)
    # result = strategy.run(observations) 