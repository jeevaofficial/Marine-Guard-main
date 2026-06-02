"""
GRU Climate Forecasting Model
==============================
This module implements a GRU (Gated Recurrent Unit) neural network
for forecasting climate variables from NASA POWER historical data.

Why GRU over LSTM:
1. Fewer parameters = faster training on CPU
2. Similar performance for sequence lengths < 100
3. Lower memory footprint
4. Simpler architecture, easier to explain in viva

Model Architecture:
- Input: Sequence of climate features (temperature, humidity, wind, pressure, precipitation)
- GRU Layer 1: 32 units (captures temporal patterns)
- GRU Layer 2: 16 units (refines features)
- Dense Output: Predicts future climate values

Author: B.Tech AI&DS 
Date: 2026
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.saving import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import logging
from typing import Dict, Tuple, List, Optional
import joblib

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CLIMATE_MODEL_CONFIG, MODEL_SAVE_DIR, SCALER_SAVE_DIR

# Set up logging
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class ClimateGRUModel:
    """
    GRU-based model for climate variable forecasting.
    
    This model takes historical climate data (from NASA POWER) and
    predicts future values for temperature, humidity, wind, pressure,
    and precipitation.
    
    Key Features:
    - Lightweight architecture optimized for CPU training
    - Early stopping to prevent overfitting
    - Supports multi-step forecasting
    - Built-in data preprocessing and inverse transform
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Climate GRU Model.
        
        Args:
            config: Model configuration dictionary (optional)
        """
        self.config = config or CLIMATE_MODEL_CONFIG
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.history = None
        
        # Model parameters from config
        self.sequence_length = self.config.get("sequence_length", 24)
        self.forecast_horizon = self.config.get("forecast_horizon", 12)
        self.gru_units_1 = self.config.get("gru_units_layer1", 32)
        self.gru_units_2 = self.config.get("gru_units_layer2", 16)
        self.dropout_rate = self.config.get("dropout_rate", 0.2)
        self.learning_rate = self.config.get("learning_rate", 0.001)
        
        logger.info(f"Climate GRU Model initialized with config: {self.config}")
    
    def build_model(self, n_features: int, n_outputs: int = 1) -> Sequential:
        """
        Build the GRU model architecture.
        
        Architecture:
        1. Input Layer: (sequence_length, n_features)
        2. GRU Layer 1: 32 units with return_sequences=True
        3. Dropout: 20% for regularization
        4. GRU Layer 2: 16 units
        5. Dropout: 20%
        6. Dense Output: (forecast_horizon * n_outputs) values
        
        Args:
            n_features: Number of input features
            n_outputs: Number of variables to predict
            
        Returns:
            Compiled Keras Sequential model
        """
        logger.info(f"Building GRU model: {n_features} features, {n_outputs} outputs")
        
        model = Sequential([
            # Input layer
            Input(shape=(self.sequence_length, n_features)),
            
            # First GRU layer - captures long-term temporal dependencies
            # return_sequences=True passes full sequence to next layer
            GRU(
                units=self.gru_units_1,
                return_sequences=True,  # Output full sequence for stacking
                activation='tanh',
                recurrent_activation='sigmoid',
                name='gru_layer_1'
            ),
            
            # Dropout for regularization
            Dropout(self.dropout_rate, name='dropout_1'),
            
            # Second GRU layer - refines temporal features
            # return_sequences=False outputs only last timestep
            GRU(
                units=self.gru_units_2,
                return_sequences=False,  # Output only final state
                activation='tanh',
                recurrent_activation='sigmoid',
                name='gru_layer_2'
            ),
            
            # Dropout for regularization
            Dropout(self.dropout_rate, name='dropout_2'),
            
            # Output layer - predicts forecast_horizon future values
            Dense(
                units=self.forecast_horizon * n_outputs,
                activation='linear',  # Linear for regression
                name='output_layer'
            )
        ])
        
        # Compile with Adam optimizer and MSE loss
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',  # Mean Squared Error for regression
            metrics=['mae']  # Mean Absolute Error for interpretation
        )
        
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)
        
        return model
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'T2M',
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training and validation data from DataFrame.
        
        Steps:
        1. Select features and target
        2. Normalize data using MinMaxScaler
        3. Create sequences for GRU input
        4. Split into train/validation sets
        
        Args:
            df: DataFrame with climate data
            target_column: Column to predict
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        from sklearn.preprocessing import MinMaxScaler
        from utils.data_processor import create_sequences
        
        # Use all numeric columns as features if not specified
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_columns = feature_columns
        
        # Ensure target is first column
        if target_column in feature_columns:
            feature_columns.remove(target_column)
        feature_columns = [target_column] + feature_columns
        
        # Extract data
        data = df[feature_columns].values
        
        logger.info(f"Data shape: {data.shape}, Features: {len(feature_columns)}")
        
        # Normalize data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = create_sequences(
            data_normalized,
            self.sequence_length,
            self.forecast_horizon,
            target_column_idx=0  # Target is first column
        )
        
        logger.info(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
        
        # Split into train/validation
        val_split = self.config.get("validation_split", 0.2)
        split_idx = int(len(X) * (1 - val_split))
        
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"Train: {len(X_train)} samples, Validation: {len(X_val)} samples")
        
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
        Train the GRU model with early stopping.
        
        Training strategy:
        1. Use early stopping to prevent overfitting
        2. Reduce learning rate on plateau
        3. Save best model checkpoint
        
        Args:
            X_train: Training input sequences
            y_train: Training targets
            X_val: Validation input sequences
            y_val: Validation targets
            save_path: Path to save the best model
            
        Returns:
            Dictionary with training history and metrics
        """
        # Build model if not already built
        n_features = X_train.shape[2]
        if self.model is None:
            self.model = self.build_model(n_features, n_outputs=1)
        
        # Prepare callbacks
        callbacks = [
            # Early stopping: stop if val_loss doesn't improve for 5 epochs
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get("early_stopping_patience", 5),
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Add model checkpoint if save path provided
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
        
        logger.info("Starting model training...")
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get("epochs", 50),
            batch_size=self.config.get("batch_size", 32),
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate final metrics
        train_loss = self.model.evaluate(X_train, y_train, verbose=0)
        val_loss = self.model.evaluate(X_val, y_val, verbose=0)
        
        results = {
            "train_loss": float(train_loss[0]),
            "train_mae": float(train_loss[1]),
            "val_loss": float(val_loss[0]),
            "val_mae": float(val_loss[1]),
            "epochs_trained": len(self.history.history['loss']),
            "best_epoch": np.argmin(self.history.history['val_loss']) + 1
        }
        
        logger.info(f"Training completed: {results}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input sequences.
        
        Args:
            X: Input sequences of shape (samples, sequence_length, features)
            
        Returns:
            Predictions of shape (samples, forecast_horizon)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a saved model.")
        
        predictions = self.model.predict(X, verbose=0)
        
        return predictions
    
    def predict_inverse_scaled(
        self,
        X: np.ndarray,
        target_idx: int = 0
    ) -> np.ndarray:
        """
        Generate predictions and inverse transform to original scale.
        
        Args:
            X: Input sequences (normalized)
            target_idx: Index of target column in scaler
            
        Returns:
            Predictions in original scale
        """
        predictions_normalized = self.predict(X)
        
        # Create dummy array for inverse transform
        n_features = len(self.scaler.min_)
        dummy = np.zeros((len(predictions_normalized), n_features))
        
        # Handle multi-step predictions
        if predictions_normalized.ndim == 1:
            dummy[:, target_idx] = predictions_normalized
        else:
            # Take mean across forecast horizon for single value
            dummy[:, target_idx] = predictions_normalized.mean(axis=1)
        
        # Inverse transform
        inverse = self.scaler.inverse_transform(dummy)
        
        return inverse[:, target_idx]
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test input sequences
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X_test)
        
        # Flatten for metrics calculation
        y_true = y_test.flatten()
        y_pred = predictions.flatten()
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # R-squared score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save(self, model_path: str, scaler_path: str) -> None:
        """
        Save model and scaler to disk.
        
        Args:
            model_path: Path to save Keras model
            scaler_path: Path to save scaler
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        
        # Save feature columns
        feature_path = scaler_path.replace('.pkl', '_features.pkl')
        joblib.dump(self.feature_columns, feature_path)
    
    def load(self, model_path: str, scaler_path: str) -> None:
        """
        Load model and scaler from disk.
        
        Args:
            model_path: Path to saved Keras model
            scaler_path: Path to saved scaler
        """
        # Load model
        self.model = load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
        
        # Load feature columns if available
        feature_path = scaler_path.replace('.pkl', '_features.pkl')
        if os.path.exists(feature_path):
            self.feature_columns = joblib.load(feature_path)


def train_climate_model(district_name: str = "Chennai", days_back: int = 60) -> Dict:
    """
    Complete training pipeline for climate forecasting model.
    
    This function:
    1. Fetches historical data from NASA POWER
    2. Preprocesses and creates sequences
    3. Trains the GRU model
    4. Saves model and scaler
    5. Returns training metrics
    
    Args:
        district_name: Name of coastal district
        days_back: Days of historical data for training
        
    Returns:
        Dictionary with training results and metrics
    """
    from services.nasa_power_service import NASAPowerService
    from utils.data_processor import clean_nasa_data, add_time_features
    
    logger.info(f"Starting climate model training for {district_name}")
    
    # Step 1: Fetch data
    nasa_service = NASAPowerService()
    df_raw = nasa_service.fetch_for_district(district_name, days_back)
    
    # Step 2: Preprocess
    df_clean = clean_nasa_data(df_raw)
    df_features = add_time_features(df_clean)
    
    logger.info(f"Training data: {len(df_features)} samples, {len(df_features.columns)} features")
    
    # Step 3: Initialize and prepare model
    model = ClimateGRUModel()
    
    # Select features for training
    feature_cols = ['T2M', 'RH2M', 'WS2M', 'PS', 'PRECTOTCORR', 
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    available_cols = [c for c in feature_cols if c in df_features.columns]
    
    X_train, X_val, y_train, y_val = model.prepare_data(
        df_features,
        target_column='T2M',  # Primary target: temperature
        feature_columns=available_cols
    )
    
    # Step 4: Train
    model_path = os.path.join(MODEL_SAVE_DIR, f"climate_gru_{district_name.lower()}.keras")
    scaler_path = os.path.join(SCALER_SAVE_DIR, f"climate_scaler_{district_name.lower()}.pkl")
    
    results = model.train(
        X_train, y_train,
        X_val, y_val,
        save_path=model_path
    )
    
    # Step 5: Save model and scaler
    model.save(model_path, scaler_path)
    
    # Step 6: Evaluate
    eval_metrics = model.evaluate(X_val, y_val)
    results.update(eval_metrics)
    results["model_path"] = model_path
    results["scaler_path"] = scaler_path
    
    logger.info(f"Climate model training completed: {results}")
    
    return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("CLIMATE GRU MODEL TRAINING")
    print("="*60)
    
    try:
        results = train_climate_model("Chennai", days_back=30)
        print("\nTraining Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error during training: {e}")
