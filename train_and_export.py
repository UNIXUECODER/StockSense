"""
train_and_export.py
───────────────────
Academic-Grade Training Pipeline for StockSense
Trains an ensemble classifier (Random Forest + Logistic Regression + Gradient Boosting)
across a multi-asset panel of major stocks to learn stationary, cross-market dynamics.
Saves model.pkl and features.pkl for instant deployment in app.py.
"""

import sys
import io

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ── Multi-Asset Universe Configuration ─────────────────────────────────────────
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'JPM', 'SPY']
START   = '2021-01-01'
END     = '2025-01-01'

FEATURES = [
    'Daily_Return', 'Return_Lag1', 'MA_10_ratio', 'MA_30_ratio', 'MA_ratio',
    'Volatility_10', 'RSI_14', 'MACD_norm', 'MACD_Signal_norm', 'MACD_Hist_norm',
    'BB_position', 'BB_width', 'Volume_Change', 'Price_Range'
]

# ── Synthetic Fallback Generator (used if offline) ─────────────────────────────
def make_synthetic(ticker, seed, base=150, drift=0.15, vol=0.25, n=1000):
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(START, periods=n)
    ret   = rng.normal(drift / 252, vol / np.sqrt(252), n)
    close = base * np.exp(np.cumsum(ret))
    noise = rng.uniform(0.995, 1.005, n)
    high  = close * rng.uniform(1.002, 1.015, n)
    low   = close * rng.uniform(0.985, 0.998, n)
    opn   = close * noise
    vol_  = rng.integers(20_000_000, 120_000_000, n).astype(float)
    df    = pd.DataFrame({'Open': opn, 'High': high, 'Low': low,
                          'Close': close, 'Volume': vol_}, index=dates)
    df.index.name = 'Date'
    return df

# ── Stationary Feature Engineering ────────────────────────────────────────────
def add_features(df):
    """
    Computes scale-invariant, stationary technical indicators.
    Normalizes indicators relative to price levels so the model generalizes
    across $20 stocks and $2000 stocks alike.
    """
    d = df.copy()
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.ffill(inplace=True)
    d.bfill(inplace=True)

    d['Daily_Return'] = d['Close'].pct_change()
    d['Return_Lag1']  = d['Daily_Return'].shift(1)

    ma10 = d['Close'].rolling(10).mean()
    ma30 = d['Close'].rolling(30).mean()
    d['MA_10_ratio']   = (d['Close'] - ma10) / (ma10 + 1e-8)
    d['MA_30_ratio']   = (d['Close'] - ma30) / (ma30 + 1e-8)
    d['MA_ratio']      = (ma10 - ma30) / (ma30 + 1e-8)
    d['Volatility_10'] = d['Daily_Return'].rolling(10).std()

    # RSI (14)
    delta = d['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss.replace(0, np.nan) + 1e-8)
    d['RSI_14'] = (100 - (100 / (1 + rs))).fillna(50.0).clip(0, 100)

    # MACD normalized by Close for cross-asset scale invariance
    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist   = macd - macd_signal
    d['MACD_norm']        = macd / (d['Close'] + 1e-8)
    d['MACD_Signal_norm'] = macd_signal / (d['Close'] + 1e-8)
    d['MACD_Hist_norm']   = macd_hist / (d['Close'] + 1e-8)

    # Bollinger Bands (normalized position: 0 = lower band, 1 = upper band)
    bb_mid   = d['Close'].rolling(20).mean()
    bb_std   = d['Close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    d['BB_width']    = (bb_upper - bb_lower) / (bb_mid + 1e-8)
    d['BB_position'] = (d['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)

    d['Volume_Change'] = d['Volume'].pct_change()
    d['Price_Range']   = (d['High'] - d['Low']) / (d['Close'] + 1e-8)

    # Binary Target: 1 if Next Day Close > Today's Close, else 0
    d['Target'] = np.where(d['Close'].shift(-1) > d['Close'], 1, 0)

    # Clean residual infinities
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    return d

# ── Data Ingestion ─────────────────────────────────────────────────────────────
print("Loading multi-asset training data (7 diverse equities)...")
raw_datasets = []

for idx, ticker in enumerate(TICKERS):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if len(df) < 200:
            raise ValueError("Insufficient rows from yfinance")
        clean_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        feat_df  = add_features(clean_df)
        # Exclude the final row because its shift(-1) target is unobservable
        trainable = feat_df.iloc[:-1].dropna()
        raw_datasets.append(trainable)
        print(f"  [+] {ticker:6s}: {len(trainable)} trading sessions loaded")
    except Exception as e:
        print(f"  [!] {ticker:6s} failed ({e}), using synthetic generator.")
        synth_df = make_synthetic(ticker, seed=idx+1)
        feat_df  = add_features(synth_df)
        trainable = feat_df.iloc[:-1].dropna()
        raw_datasets.append(trainable)

# Merge panel dataset
full_panel = pd.concat(raw_datasets).sort_index()
print(f"\nTotal Multi-Asset Training Corpus: {len(full_panel)} samples")

X = full_panel[FEATURES].values
y = full_panel['Target'].values

# Temporal Train / Test Split (Strict temporal holdout - no future lookahead)
split_idx = int(len(full_panel) * 0.80)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set: {len(X_train)} samples | Test set: {len(X_test)} samples")

# ── Model Architecture: Tuned Regularized Ensemble ────────────────────────────
print("\nTraining models with StandardScaler pipelines...")

pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=0.1, max_iter=1000, random_state=42))
])

pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=15, random_state=42))
])

pipe_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=3, min_samples_leaf=20, random_state=42))
])

# Soft-Voting Ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', pipe_lr),
        ('rf', pipe_rf),
        ('gb', pipe_gb)
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
test_preds = ensemble.predict(X_test)
test_proba = ensemble.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_proba)

print("\n" + "=" * 55)
print("  MODEL EVALUATION (Out-of-Sample Test Set)")
print("=" * 55)
print(f"  Test Accuracy   : {acc * 100:.2f}%")
print(f"  ROC-AUC Score   : {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=['Down (0)', 'Up (1)']))

# ── Save Model & Feature Schema ───────────────────────────────────────────────
joblib.dump(ensemble, 'model.pkl')
joblib.dump(FEATURES, 'features.pkl')

print("  [SAVED] model.pkl (Ensemble Pipeline)")
print("  [SAVED] features.pkl (14 stationary features)")
print("\n[OK] Model successfully trained and exported. Ready to run app.py.")
