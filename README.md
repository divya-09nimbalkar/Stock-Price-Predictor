# Stock Price Predictor - Neural Network Time Series Model

## Overview
A production-grade multi-layer perceptron (MLP) neural network for predicting stock prices using historical time-series data. Built with scikit-learn, featuring sophisticated preprocessing, deep learning architecture with technical indicators, and comprehensive evaluation metrics.

## Features
✅ **Deep Neural Network** - 3-layer MLP with ReLU activation (256→128→64 units)  
✅ **Technical Indicators** - SMA-20, SMA-50, and volatility features  
✅ **Time-Series Preprocessing** - MinMax scaling and sequence windowing  
✅ **Synthetic Data Generation** - Realistic stock data with trend and mean reversion  
✅ **Early Stopping** - Prevents overfitting with validation monitoring  
✅ **Multi-Step Forecasting** - Predict 7+ days into the future  
✅ **Model Persistence** - Save/load trained models with pickle  
✅ **Comprehensive Logging** - Track training and predictions  
✅ **Performance Metrics** - R², RMSE, MAE, MAPE evaluation  

## Model Architecture
```
Input Features (60 timesteps × 4 features: close, SMA20, SMA50, volatility)
    ↓
Dense Layer 1: 256 units (ReLU activation) → Early Stopping
    ↓
Dense Layer 2: 128 units (ReLU activation)
    ↓
Dense Layer 3: 64 units (ReLU activation)
    ↓
Output Layer: 1 unit (Price prediction)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train and Predict
```python
from stock_price_predictor import StockPricePredictor

# Initialize with 60-day lookback window
predictor = StockPricePredictor(lookback_window=60)

# Train on data
df = predictor._generate_synthetic_stock_data(n_days=500)
predictor.train(df)
predictor.save_model()

# Predict next 7 days
future_prices = predictor.predict_next_days(df, days=7)
print(future_prices)
```

### Run the Demo
```bash
python stock_price_predictor.py
```

## Training Parameters
- **Lookback Window**: 60 days (features for each prediction)
- **Hidden Layers**: [256, 128, 64] units (3-layer deep network)
- **Activation Function**: ReLU (non-linear transformation)
- **Optimizer**: Adam (adaptive learning rate)
- **Max Iterations**: 200 epochs
- **Batch Size**: 32 samples
- **Early Stopping**: 10 iterations without improvement
- **Train/Test Split**: 80/20 stratified

## Technical Features
- **SMA-20**: 20-day Simple Moving Average
- **SMA-50**: 50-day Simple Moving Average  
- **Volatility**: 20-day rolling standard deviation
- Normalized to [0, 1] range for optimal neural network performance

## Performance Metrics
- **R² Score**: Explains proportion of price variance (max 1.0)
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAE** (Mean Absolute Error): Average absolute deviation
- **MAPE** (Mean Absolute Percentage Error): Percentage error

## Output Files
- `stock_model.pkl` - Trained MLP neural network
- `stock_scaler.pkl` - MinMax scaler for data normalization
- `stock_predictor.log` - Execution logs with detailed metrics

## Key Capabilities
- **Non-linear Relationships**: Captures complex price patterns
- **Automatic Feature Learning**: Network learns optimal feature combinations
- **Regularization**: Prevents overfitting on training data
- **Gradient-Based Optimization**: Efficient parameter updates
- **Time-Series Awareness**: Sequential prediction with rolling windows

## Data Requirements
- Daily historical stock prices (OHLCV format)
- Minimum 200+ days of data for reliable training
- Continuous time series without large gaps
- Stationary or normalized data for best results

## Customization
Modify in `stock_price_predictor.py`:
- `hidden_layer_sizes`: Change from (256,128,64) to (512,256,128)
- `lookback_window`: Change from 60 to 30/90/120 days
- `max_iter`: Increase epochs for more training iterations
- Technical indicators: Add RSI, MACD, Bollinger Bands

## Model Hyperparameters
- Learning Rate: 0.001 (controlled by Adam optimizer)
- Batch Size: 32 (samples per gradient update)
- Validation Split: 0.1 (10% for early stopping)
- Random State: 42 (reproducibility)

## Author
Divya Nimbalkar

## License
MIT
