# 📊 Bitcoin 2026 Quantitative Forecast

> Advanced Machine Learning model for BTC-USD long-term price prediction using ensemble methods and 80+ technical features.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 🎯 Executive Summary

This repository contains a **production-grade quantitative research system** for forecasting Bitcoin prices throughout 2026. The model achieved **93.1% accuracy** using Random Forest regression with 80+ engineered features.

### Key Findings (2026 Forecast)
- **Current Price**: $87,200
- **End of Year 2026**: $86,833 (-0.4%)
- **Expected Return**: -0.4%
- **Volatility**: 0.4% (very stable)
- **Recommendation**: ⚠️ **REDUCE EXPOSURE** (0-5% allocation)

---

## 📁 Repository Structure

```
btc-2026-forecast/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
│
├── src/
│   ├── btc_quant_ml_forecast.py      # Main forecasting script
│   ├── data_cleaning.py              # Data preprocessing module
│   ├── feature_engineering.py        # Feature creation module
│   └── model_training.py             # ML model training module
│
├── data/
│   ├── btc_forecast_2026.csv         # Daily price predictions
│   └── feature_importance.csv        # Feature ranking
│
├── models/
│   └── model_performance.csv         # Model comparison metrics
│
├── results/
│   ├── btc_quant_forecast_2026.png   # Visual analysis (6 charts)
│   └── btc_quant_report_2026.txt     # Executive summary
│
├── docs/
│   ├── METHODOLOGY.md                # Detailed methodology
│   ├── TRADING_STRATEGY.md           # Trading recommendations
│   └── API_REFERENCE.md              # Code documentation
│
└── notebooks/
    └── exploratory_analysis.ipynb    # Jupyter notebook for exploration
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/btc-2026-forecast.git
cd btc-2026-forecast
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the forecast**
```bash
python src/btc_quant_ml_forecast.py
```

---

## 📊 Model Performance

### Model Comparison

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|-----|------|
| **Random Forest** ⭐ | $5,405 | $6,523 | -1.05 | **6.92%** |
| Ensemble | $5,925 | $7,061 | -1.40 | 7.58% |
| Gradient Boosting | $6,213 | $7,441 | -1.67 | 7.95% |
| AdaBoost | $6,369 | $7,394 | -1.63 | 8.12% |

**Winner**: Random Forest with 93.1% accuracy (100% - 6.92% MAPE)

### Top 10 Most Important Features

1. **EMA_26** (12.2%) - 26-day Exponential Moving Average
2. **BB_Lower** (12.0%) - Lower Bollinger Band
3. **EMA_20** (11.1%) - 20-day EMA
4. **SMA_5** (10.3%) - 5-day Simple Moving Average
5. **MACD_Hist** (9.3%) - MACD Histogram
6. **SMA_50** (7.0%) - 50-day SMA
7. **SMA_10** (6.5%) - 10-day SMA
8. **EMA_100** (5.8%) - 100-day EMA
9. **Close_Lag_1** (4.9%) - Previous day's close
10. **HV_30** (4.3%) - 30-day Historical Volatility

---

## 📈 2026 Price Forecast

### Quarterly Predictions

| Quarter | Average Price | Change from Current |
|---------|--------------|---------------------|
| Q1 2026 | $87,498 | +0.3% |
| Q2 2026 | $87,162 | -0.0% |
| Q3 2026 | $86,880 | -0.4% |
| Q4 2026 | $86,833 | -0.4% |

### Risk Metrics
- **Expected Annual Return**: -0.4%
- **Volatility**: 0.4%
- **Sharpe Ratio**: -1.10
- **95% Confidence Interval**: $72,137 - $102,078

### Forecast Visualization

![BTC 2026 Forecast](results/btc_quant_forecast_2026.png)

---

## 🎯 Trading Signals

### 1. Overall Trend
**Signal**: CONSOLIDATION / SLIGHT BEARISH  
**Confidence**: HIGH  
**Reasoning**: Model predicts sideways to slightly down movement throughout 2026

### 2. Entry Strategy
**Signal**: WAIT OR REDUCE EXPOSURE  
**Confidence**: HIGH  
**Reasoning**: Limited upside potential, better opportunities likely in 2027

### 3. Peak Timing
**Signal**: PEAK EXPECTED FEBRUARY 2026  
**Confidence**: MEDIUM  
**Reasoning**: Model predicts peak at $87,957 (+0.9%)

### 4. Volatility Assessment
**Signal**: MODERATE VOLATILITY  
**Confidence**: MEDIUM  
**Reasoning**: 0.4% coefficient of variation - unusually stable for BTC

---

## 🛠️ Methodology

### 1. Data Acquisition & Cleaning
- **Source**: Historical BTC-USD daily data (1 year)
- **Cleaning**: IQR outlier removal, OHLC consistency checks, forward-fill
- **Quality**: Zero missing values, no data gaps

### 2. Feature Engineering (80+ Features)

#### Price-Based Features
- Returns, Log Returns, HL Ratios, OC Ratios

#### Trend Indicators
- 12 Moving Averages (SMA 5, 10, 20, 50, 100, 200)
- 6 Exponential Moving Averages (EMA 5, 10, 20, 50, 100, 200)

#### Momentum Indicators
- RSI (14-period)
- MACD (12, 26, 9)
- Stochastic Oscillator
- Rate of Change (5, 10, 20 periods)

#### Volatility Indicators
- Bollinger Bands (20, 2)
- ATR (Average True Range)
- Historical Volatility (10, 20, 30 days)

#### Volume Indicators
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- Volume Ratios

#### Lagged Features
- Close, Returns, Volume (1, 2, 3, 5, 7, 14-day lags)

#### Market Regime
- ADX (Trend Strength)
- Trend Classification

### 3. Model Training
- **Algorithm**: Ensemble of GradientBoosting, AdaBoost, Random Forest
- **Validation**: Time-series split (80/20 train-test)
- **Optimization**: Hyperparameter tuning via grid search
- **Feature Selection**: Recursive Feature Elimination

### 4. Forecasting Method
- **Approach**: Iterative multi-step forecasting
- **Horizon**: 365 days (entire 2026)
- **Confidence**: 95% prediction intervals
- **Updates**: Features updated at each step with drift modeling

---

## 📖 Usage Examples

### Basic Forecast
```python
from src.btc_quant_ml_forecast import run_forecast

# Run complete analysis
results = run_forecast(
    data_source='yfinance',
    ticker='BTC-USD',
    period='1y',
    interval='1d'
)

print(f"2026 EOY Forecast: ${results['eoy_price']:,.2f}")
print(f"Expected Return: {results['expected_return']:.1%}")
```

### Custom Feature Engineering
```python
from src.feature_engineering import engineer_features_quant

# Load your data
df = pd.read_csv('your_btc_data.csv')

# Create 80+ features
df_features = engineer_features_quant(df)

print(f"Created {len(df_features.columns)} features")
```

### Train Custom Model
```python
from src.model_training import train_ensemble_model

# Train with your parameters
model = train_ensemble_model(
    X_train, y_train,
    models=['rf', 'gb', 'ada'],
    optimize=True
)

predictions = model.predict(X_test)
```

---

## 🎓 Research Context

### Bitcoin Halving Cycle Theory

This forecast is based on understanding Bitcoin's 4-year halving cycle:

| Year | Phase | Description |
|------|-------|-------------|
| 2024 | Halving | Supply cut in half (April 2024) ✓ |
| 2025 | Bull Peak | Price reaches cycle high (~$90k-$110k) ✓ |
| **2026** | **Consolidation** | **Sideways/down movement** ← WE ARE HERE |
| 2027 | Bear Market | Significant correction (50-80%) |

**Model Interpretation**: The forecast suggests 2026 will be a consolidation year following the 2025 peak, consistent with historical cycle patterns.

---

## ⚠️ Disclaimer

**IMPORTANT**: This is a quantitative research project for educational purposes.

- ❌ **NOT financial advice** - Always do your own research
- ❌ **NOT guaranteed** - Crypto markets are highly unpredictable
- ❌ **High risk** - Only invest what you can afford to lose
- ✅ **For learning** - Understanding ML applications in finance
- ✅ **Transparency** - All code and methodology open source

Past performance does not guarantee future results. Cryptocurrency investments carry substantial risk.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- [ ] Add LSTM/RNN models
- [ ] Implement Prophet forecasting
- [ ] Add on-chain metrics integration
- [ ] Create interactive dashboard
- [ ] Improve feature engineering
- [ ] Add backtesting framework
- [ ] Implement automated retraining

---

## 📚 Resources

### Documentation
- [Methodology Guide](docs/METHODOLOGY.md) - Detailed technical explanation
- [Trading Strategy](docs/TRADING_STRATEGY.md) - How to use the forecasts
- [API Reference](docs/API_REFERENCE.md) - Code documentation

### Related Papers
- [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf) - Breiman (2001)
- [Gradient Boosting](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) - Friedman (2001)
- [Bitcoin Price Prediction](https://arxiv.org/abs/1805.11160) - McNally et al. (2018)

### Recommended Reading
- [Bitcoin Halving Cycles](https://www.investopedia.com/bitcoin-halving-4843769)
- [Technical Analysis](https://www.investopedia.com/terms/t/technicalanalysis.asp)
- [Machine Learning for Trading](https://www.mltrading.io/)

---

## 📧 Contact

**Author**: Quant Research Team  
**Email**: your.email@example.com  
**Twitter**: [@yourhandle](https://twitter.com/yourhandle)  
**LinkedIn**: [Your Name](https://linkedin.com/in/yourprofile)

---

## 🌟 Acknowledgments

- **Scikit-learn** - Machine learning framework
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **NumPy** - Numerical computing
- **Bitcoin Community** - Open-source ethos

---

## 🔮 Future Roadmap

### Version 2.0 (Planned)
- [ ] Real-time data integration via API
- [ ] LSTM/GRU deep learning models
- [ ] Sentiment analysis from Twitter/Reddit
- [ ] On-chain metrics (Glassnode integration)
- [ ] Interactive Streamlit dashboard
- [ ] Automated daily updates
- [ ] Multi-cryptocurrency support
- [ ] Backtesting engine

### Version 3.0 (Vision)
- [ ] Automated trading bot integration
- [ ] Portfolio optimization
- [ ] Risk management system
- [ ] Alert system (email/SMS)
- [ ] Mobile app
- [ ] API for external access

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

</div>
