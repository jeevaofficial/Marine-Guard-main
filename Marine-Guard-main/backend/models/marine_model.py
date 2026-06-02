"""
GRU Marine Wave Prediction Model
=================================
This module implements a GRU neural network for predicting wave heights
by combining climate forecasts with marine data.

Model Purpose:
- Predict wave heights for the next 6-24 hours
- Classify sea safety conditions
- Support marine safety decision-making

Model Architecture:
- Input: Combined climate + marine features
- GRU layers for temporal pattern learning
- Output: Predicted wave heights for forecast horizon

Author: B.Tech AI&DS 
Date: 2026
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.saving import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import logging
from typing import Dict, Tuple, List, Optional
import joblib

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MARINE_MODEL_CONFIG, MODEL_SAVE_DIR, SCALER_SAVE_DIR, SAFETY_THRESHOLDS

# Set up logging
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class MarineGRUModel:
    """
    GRU-based model for marine wave height prediction.
    
    This model predicts wave heights by learning from:
    1. Historical wave patterns (from Open-Meteo)
    2. Climate variables (from NASA POWER / predictions)
    3. Temporal patterns (time of day, season)
    
    The predictions are then used for safety classification.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Marine GRU Model.
        
        Args:
            config: Model configuration dictionary
        """
        self.config = config or MARINE_MODEL_CONFIG
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.history = None
        
        # Model parameters
        self.sequence_length = self.config.get("sequence_length", 24)
        self.forecast_horizon = self.config.get("forecast_horizon", 24)
        self.gru_units_1 = self.config.get("gru_units_layer1", 32)
        self.gru_units_2 = self.config.get("gru_units_layer2", 16)
        self.dropout_rate = self.config.get("dropout_rate", 0.2)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        
        logger.info(f"Marine GRU Model initialized with config: {self.config}")
    
    def build_model(self, n_features: int) -> Sequential:
        """
        Build the GRU model architecture for wave prediction.
        
        Architecture optimized for wave height prediction:
        1. Input: (sequence_length, n_features)
        2. GRU Layer 1: 32 units, captures wave patterns
        3. GRU Layer 2: 16 units, refines predictions
        4. Dense Output: forecast_horizon wave height predictions
        
        Args:
            n_features: Number of input features
            
        Returns:
            Compiled Keras Sequential model
        """
        logger.info(f"Building Marine GRU model: {n_features} features")
        
        model = Sequential([
            # Input layer
            Input(shape=(self.sequence_length, n_features)),
            
            # First GRU layer
            GRU(
                units=self.gru_units_1,
                return_sequences=True,
                activation='tanh',
                recurrent_activation='sigmoid',
                name='marine_gru_1'
            ),
            
            # Dropout
            Dropout(self.dropout_rate),
            
            # Second GRU layer
            GRU(
                units=self.gru_units_2,
                return_sequences=False,
                activation='tanh',
                recurrent_activation='sigmoid',
                name='marine_gru_2'
            ),
            
            # Dropout
            Dropout(self.dropout_rate),
            
            # Dense layer before output
            Dense(32, activation='relu', name='dense_hidden'),
            
            # Output layer - predict wave heights for forecast horizon
            Dense(
                units=self.forecast_horizon,
                activation='linear',  # Linear for regression
                name='wave_height_output'
            )
        ])
        
        # Compile with Adam and MSE
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Marine model architecture built:")
        model.summary(print_fn=logger.info)
        
        return model
    
    def prepare_training_data(
        self,
        marine_df: pd.DataFrame,
        climate_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data from marine and climate DataFrames.
        
        Steps:
        1. Merge marine and climate data if climate provided
        2. Select relevant features
        3. Normalize data
        4. Create sequences for GRU
        
        Args:
            marine_df: DataFrame with marine data (must have 'wave_height')
            climate_df: Optional DataFrame with climate data
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        from utils.data_processor import create_sequences, add_time_features
        
        # Start with marine data
        df = marine_df.copy()
        
        # Merge with climate data if provided
        if climate_df is not None:
            df = pd.merge(
                df, climate_df,
                left_index=True, right_index=True,
                how='inner'
            )
        
        # Add time features
        df = add_time_features(df)
        
        # Select features (wave_height must be first for target)
        target_col = 'wave_height'
        
        # Define feature columns
        possible_features = [
            'wave_height', 'wave_period', 'wave_direction',
            'wind_wave_height', 'swell_wave_height',
            'wind_speed_10m', 'wind_direction_10m',
            'temperature_2m', 'surface_pressure',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        # Use available features
        feature_cols = [c for c in possible_features if c in df.columns]
        
        # Ensure target is first
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        feature_cols = [target_col] + feature_cols
        
        self.feature_columns = feature_cols
        
        # Extract data
        data = df[feature_cols].values
        
        logger.info(f"Training data shape: {data.shape}, Features: {feature_cols}")
        
        # Normalize
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = create_sequences(
            data_normalized,
            self.sequence_length,
            self.forecast_horizon,
            target_column_idx=0
        )
        
        logger.info(f"Sequences: X shape {X.shape}, y shape {y.shape}")
        
        # Split
        val_split = self.config.get("validation_split", 0.2)
        split_idx = int(len(X) * (1 - val_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return X_train, X_val, y_train, y_val
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Train the marine GRU model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            save_path: Path to save best model
            
        Returns:
            Training results dictionary
        """
        n_features = X_train.shape[2]
        
        if self.model is None:
            self.model = self.build_model(n_features)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get("early_stopping_patience", 5),
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(
                    save_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        logger.info("Starting marine model training...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get("epochs", 50),
            batch_size=self.config.get("batch_size", 32),
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate metrics
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        results = {
            "train_loss": float(train_loss[0]),
            "train_mae": float(train_loss[1]),
            "val_loss": float(val_loss[0]),
            "val_mae": float(val_loss[1]),
            "epochs_trained": len(self.history.history['loss']),
            "best_epoch": int(np.argmin(self.history.history['val_loss']) + 1)
        }
        
        logger.info(f"Training completed: {results}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate wave height predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted wave heights for forecast horizon
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        return self.model.predict(X, verbose=0)
    
    def predict_single(
        self,
        recent_data: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Make prediction from recent data DataFrame.
        
        Args:
            recent_data: DataFrame with recent observations
            
        Returns:
            Tuple of (predictions, timestamps)
        """
        from utils.data_processor import add_time_features
        
        # Add time features
        df = add_time_features(recent_data.copy())
        
        # Select features in correct order
        data = df[self.feature_columns].values
        
        # Normalize
        data_normalized = self.scaler.transform(data)
        
        # Create single sequence
        if len(data_normalized) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} timesteps")
        
        X = data_normalized[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        
        # Predict
        predictions_normalized = self.predict(X)[0]
        
        # Inverse transform
        n_features = len(self.scaler.min_)
        dummy = np.zeros((len(predictions_normalized), n_features))
        dummy[:, 0] = predictions_normalized  # wave_height is column 0
        
        predictions = self.scaler.inverse_transform(dummy)[:, 0]
        
        # Generate timestamps
        last_time = df.index[-1]
        timestamps = [
            (last_time + pd.Timedelta(hours=i+1)).isoformat()
            for i in range(self.forecast_horizon)
        ]
        
        return predictions, timestamps
    
    def classify_safety(self, wave_height: float, wind_speed: Optional[float] = None) -> Dict:
        """
        Classify marine safety based on predicted wave height.
        
        Args:
            wave_height: Predicted wave height in meters
            wind_speed: Optional wind speed in m/s
            
        Returns:
            Safety classification dictionary
        """
        from utils.helpers import classify_safety_status
        return classify_safety_status(wave_height, wind_speed)
    
    def predict_with_safety(
        self,
        recent_data: pd.DataFrame
    ) -> Dict:
        """
        Make predictions and include safety classification.
        
        Args:
            recent_data: DataFrame with recent observations
            
        Returns:
            Dictionary with predictions and safety status
        """
        predictions, timestamps = self.predict_single(recent_data)
        
        # Get wind speed from recent data if available
        wind_speed = None
        if 'wind_speed_10m' in recent_data.columns:
            wind_speed = float(recent_data['wind_speed_10m'].iloc[-1])
        
        # Classify based on max predicted wave height
        max_wave = float(np.max(predictions))
        avg_wave = float(np.mean(predictions))
        safety = self.classify_safety(max_wave, wind_speed)
        
        return {
            "predictions": predictions.tolist(),
            "timestamps": timestamps,
            "statistics": {
                "min": float(np.min(predictions)),
                "max": max_wave,
                "mean": avg_wave,
                "std": float(np.std(predictions))
            },
            "safety": safety,
            "forecast_horizon_hours": self.forecast_horizon
        }
    
    def save(self, model_path: str, scaler_path: str) -> None:
        """Save model and scaler."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        if self.scaler:
            joblib.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path: str, scaler_path: str) -> None:
        """Load model and scaler."""
        self.model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        saved_data = joblib.load(scaler_path)
        self.scaler = saved_data['scaler']
        self.feature_columns = saved_data['feature_columns']
        logger.info(f"Scaler loaded from {scaler_path}")


def train_marine_model(
    district_name: str = "Chennai",
    forecast_days: int = 7
) -> Dict:
    """
    Complete training pipeline for marine wave prediction model.
    
    This function:
    1. Fetches marine forecast data from Open-Meteo
    2. Prepares training sequences
    3. Trains the GRU model
    4. Saves model and scaler
    5. Returns training metrics
    
    Args:
        district_name: Name of coastal district
        forecast_days: Days of data for training
        
    Returns:
        Dictionary with training results
    """
    from services.open_meteo_service import OpenMeteoMarineService
    
    logger.info(f"Starting marine model training for {district_name}")
    
    # Step 1: Fetch data
    marine_service = OpenMeteoMarineService()
    df = marine_service.fetch_for_district(district_name, forecast_days=forecast_days)
    
    logger.info(f"Fetched {len(df)} records for training")
    
    # Step 2: Initialize model
    model = MarineGRUModel()
    
    # Step 3: Prepare data
    X_train, X_val, y_train, y_val = model.prepare_training_data(df)
    
    # Step 4: Train
    model_path = os.path.join(MODEL_SAVE_DIR, f"marine_gru_{district_name.lower()}.keras")
    scaler_path = os.path.join(SCALER_SAVE_DIR, f"marine_scaler_{district_name.lower()}.pkl")
    
    results = model.train(
        X_train, y_train,
        X_val, y_val,
        save_path=model_path
    )
    
    # Step 5: Save
    model.save(model_path, scaler_path)
    
    results["model_path"] = model_path
    results["scaler_path"] = scaler_path
    results["district"] = district_name
    
    logger.info(f"Marine model training completed: {results}")
    
    return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("MARINE GRU MODEL TRAINING")
    print("="*60)
    
    try:
        results = train_marine_model("Chennai", forecast_days=7)
        print("\nTraining Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
