"""
Stock Price Predictor - Production-Grade Neural Network Time Series Model
Predicts future stock prices using historical price data and deep learning.
"""

import numpy as np
import pandas as pd
import logging
import os
import warnings
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_predictor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StockPricePredictor:
    """Production-grade neural network stock price prediction model."""
    
    def __init__(self, lookback_window=60, model_path='stock_model.pkl', scaler_path='stock_scaler.pkl'):
        self.lookback_window = lookback_window  # Number of days to look back
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.close_min = 0
        self.close_range = 1
        self.is_trained = False
        self.train_history = {}
        
    def _generate_synthetic_stock_data(self, n_days=500, initial_price=100):
        """Generate realistic synthetic stock price data with trend and volatility."""
        logger.info(f"Generating {n_days} days of synthetic stock price data...")
        
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        prices = [initial_price]
        volumes = []
        
        for i in range(1, n_days):
            # Random walk with drift and mean reversion
            trend = 0.0002
            volatility = 0.02
            mean_reversion = 0.001 * (100 - prices[-1]) / 100
            
            price_change = prices[-1] * (trend + mean_reversion + np.random.normal(0, volatility))
            prices.append(prices[-1] + price_change)
            volumes.append(np.random.uniform(1000000, 5000000))
        
        volumes.append(np.random.uniform(1000000, 5000000))
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices + np.random.normal(0, 0.5, n_days),
            'high': [p + abs(np.random.normal(0, 1)) for p in prices],
            'low': [p - abs(np.random.normal(0, 1)) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        df['close'] = df['close'].clip(lower=10)  # Minimum price floor
        df['high'] = df[['high', 'close']].max(axis=1)
        df['low'] = df[['low', 'close']].min(axis=1)
        df['open'] = df['open'].clip(lower=10)
        
        logger.info(f"Generated data shape: {df.shape}")
        logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
    
    def preprocess_data(self, df, fit_scaler=True):
        """Preprocess stock data with technical indicators."""
        data = df[['close']].copy()
        
        # Add technical indicators
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['SMA_50'] = data['close'].rolling(window=50).mean()
        data['volatility'] = data['close'].rolling(window=20).std()
        
        # Fill NaN values
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        if fit_scaler:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scaler.fit_transform(data)
            # Store the close price range for later inverse transform
            self.close_min = self.scaler.data_min_[0]
            self.close_range = self.scaler.data_range_[0]
            logger.info("Fitted scaler on training data")
        else:
            if self.scaler is None:
                logger.error("Scaler not fitted. Train model first.")
                raise ValueError("Scaler not fitted")
            scaled_data = self.scaler.transform(data)
        
        return scaled_data
    
    def create_sequences(self, data):
        """Create sequences for neural network training."""
        X, y = [], []
        
        for i in range(len(data) - self.lookback_window):
            X.append(data[i:i + self.lookback_window].flatten())
            y.append(data[i + self.lookback_window, 0])  # Close price
        
        return np.array(X), np.array(y)
    
    def train(self, df=None):
        """Train the neural network model on stock price data."""
        logger.info("=" * 60)
        logger.info("Starting Neural Network model training...")
        logger.info("=" * 60)
        
        # Generate or use provided data
        if df is None:
            df = self._generate_synthetic_stock_data()
        else:
            logger.info(f"Using provided data with shape {df.shape}")
        
        # Preprocess
        scaled_data = self.preprocess_data(df, fit_scaler=True)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        logger.info(f"Created sequences - X shape: {X.shape}, y shape: {y.shape}")
        
        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Build and train model - Multi-layer Perceptron with 3 hidden layers
        logger.info("Training Multi-Layer Perceptron (3 hidden layers)...")
        
        self.model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),  # 3 hidden layers
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            batch_size=32,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Inverse transform using stored scale parameters
        y_train_actual = y_train * self.close_range + self.close_min
        y_test_actual = y_test * self.close_range + self.close_min
        y_pred_train_actual = y_pred_train * self.close_range + self.close_min
        y_pred_test_actual = y_pred_test * self.close_range + self.close_min
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual))
        test_mae = mean_absolute_error(y_test_actual, y_pred_test_actual)
        test_mape = np.mean(np.abs((y_test_actual - y_pred_test_actual) / y_test_actual)) * 100
        
        logger.info(f"Train R² Score: {train_r2:.4f}")
        logger.info(f"Test R² Score: {test_r2:.4f}")
        logger.info(f"Train RMSE: ${train_rmse:.4f}")
        logger.info(f"Test RMSE: ${test_rmse:.4f}")
        logger.info(f"Test MAE: ${test_mae:.4f}")
        logger.info(f"Test MAPE: {test_mape:.2f}%")
        
        self.train_history = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_mape': test_mape,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        self.is_trained = True
        logger.info("Model training completed successfully!")
        logger.info("=" * 60)
        
        return self.model
    
    def predict_next_days(self, df, days=7):
        """Predict next N days of stock prices."""
        if not self.is_trained or self.model is None:
            logger.error("Model not trained. Train model first.")
            raise ValueError("Model not trained")
        
        logger.info(f"\nPredicting next {days} days...")
        
        # Preprocess recent data
        scaled_data = self.preprocess_data(df, fit_scaler=False)
        
        # Get last window
        last_window = scaled_data[-self.lookback_window:].flatten()
        
        predictions = []
        current_window = last_window.copy()
        
        for i in range(days):
            next_price = self.model.predict(current_window.reshape(1, -1))[0]
            predictions.append(next_price)
            
            # Update window - shift and add new prediction
            # For simplicity, update only the close price part
            current_window = np.roll(current_window, -4)  # Shift by 4 features
            current_window[-4] = next_price  # Update close price
        
        # Inverse transform predictions
        predictions = np.array(predictions)
        predictions_actual = predictions * self.close_range + self.close_min
        
        # Generate future dates
        last_date = df['date'].max() if 'date' in df.columns else datetime.now()
        future_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted_price': predictions_actual
        })
        
        logger.info(f"Predictions for next {days} days:")
        for idx, row in predictions_df.iterrows():
            logger.info(f"  {row['date'].date()}: ${row['predicted_price']:.2f}")
        
        return predictions_df
    
    def save_model(self):
        """Save trained model and scaler."""
        if self.model is None:
            logger.error("No model to save. Train first.")
            return
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        scale_info = {
            'scaler': self.scaler,
            'close_min': self.close_min,
            'close_range': self.close_range
        }
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(scale_info, f)
        logger.info(f"Model saved to {self.model_path}")
        logger.info(f"Scaler saved to {self.scaler_path}")
    
    def load_model(self):
        """Load trained model and scaler."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.scaler_path, 'rb') as f:
            scale_info = pickle.load(f)
            self.scaler = scale_info['scaler']
            self.close_min = scale_info['close_min']
            self.close_range = scale_info['close_range']
        
        self.is_trained = True
        logger.info("Model loaded successfully!")
        return True


def main():
    """Main execution function."""
    logger.info("Stock Price Predictor - Neural Network - Starting Application")
    
    # Initialize predictor
    predictor = StockPricePredictor(lookback_window=60)
    
    # Generate data
    df = predictor._generate_synthetic_stock_data(n_days=500)
    
    # Train model
    predictor.train(df)
    predictor.save_model()
    
    # Make predictions
    future_prices = predictor.predict_next_days(df, days=7)
    
    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics:")
    logger.info(f"  Total Training Samples: {predictor.train_history['train_samples']}")
    logger.info(f"  Total Test Samples: {predictor.train_history['test_samples']}")
    logger.info(f"  Best Test R² Score: {predictor.train_history['test_r2']:.4f}")
    logger.info("=" * 60)
    logger.info("Application completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
