# Stock Price Movement Predictor

Machine learning project that predicts whether a stock's price will increase or decrease the next trading day using historical market data.

The model uses time-series features and neural networks to classify short-term stock price movement.

Features
- Uses historical stock data
- Feature engineering with technical indicators
- Sliding time window for sequence modeling
- Two model architectures:
  - MLP (Feedforward Neural Network)
  - LSTM (Recurrent Neural Network)
- Early stopping to prevent overfitting
- Evaluation using multiple metrics

Technologies
- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn

Dataset

The model uses historical price data for major stocks:
- AAPL
- MSFT
- NVDA
- AMZN
- GOOGL

Each dataset includes:

Date
Open
High
Low
Close
Adj Close
Volume

Data is stored as CSV files in the data/ directory.

Feature Engineering

Additional technical indicators are computed:

- SMA 10 → 10-day moving average
- SMA 20 → 20-day moving average
- Daily Return
- Volatility (10-day standard deviation)

These indicators help capture short-term market trends.

Target Variable

The model predicts whether the next day's closing price will increase.

Target = 1 if Close(t+1) > Close(t)
Target = 0 otherwise

Model Architectures

MLP

Input → Linear → ReLU → Dropout → Linear → Output

LSTM

Input Window → LSTM Layers → Dense Layer → Output

Training

Training includes:
- train/validation split
- Adam optimizer
- Binary cross entropy loss
- early stopping with patience

Example training output:

Epoch 001: train_loss=0.6896 val_loss=0.6924
Epoch 002: train_loss=0.6893 val_loss=0.6932

Evaluation Metrics

Models are evaluated using:
- Accuracy
- ROC AUC
- Balanced Accuracy

How to Run

Install dependencies:

pip install torch pandas numpy scikit-learn

Run the training script:

python stockpricepredictor.py

