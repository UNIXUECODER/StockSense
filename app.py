"""
app.py  —  Academic-Grade Backend for StockSense
─────────────────────────────────────────────────
Provides:
  - CSV Dataset Analysis or Live Ticker Lookups (via yfinance)
  - Scale-invariant Technical Feature Engineering
  - Soft-Voting Ensemble Inference (Random Forest + Logistic Regression + Gradient Boosting)
  - Quantitative Strategy Backtesting (Cumulative Strategy vs. Buy & Hold Benchmark)
  - Risk & Performance Analytics (Sharpe Ratio, Win Rate, Max Drawdown, Precision)
  - Technical Indicator Explainability Signals
  - Base64 Dark-Themed Analytical Charts
"""

import sys
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Safe UTF-8 encoding across Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_score, recall_score

from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# ── Dark Theme Styling ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.6,
    'axes.grid':        True,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'font.family':      'monospace',
    'axes.titlesize':   12,
    'axes.labelsize':   10,
})

# ── Load Pretrained Model & Feature Schema ────────────────────────────────────
try:
    MODEL    = joblib.load('model.pkl')
    FEATURES = joblib.load('features.pkl')
    print("[OK] Model and feature schema loaded successfully.")
except FileNotFoundError:
    raise SystemExit("[ERROR] model.pkl or features.pkl not found. Run train_and_export.py first.")

# ── Stationary Feature Engineering ────────────────────────────────────────────
def add_features(df):
    """
    Computes scale-invariant, stationary technical indicators matching train_and_export.py.
    """
    d = df.copy()
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.ffill(inplace=True)
    d.bfill(inplace=True)

    d['Daily_Return'] = d['Close'].pct_change()
    d['Return_Lag1']  = d['Daily_Return'].shift(1)

    ma10 = d['Close'].rolling(10).mean()
    ma30 = d['Close'].rolling(30).mean()
    d['MA_10'] = ma10
    d['MA_30'] = ma30
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

    # MACD normalized by Close
    ema12 = d['Close'].ewm(span=12, adjust=False).mean()
    ema26 = d['Close'].ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist   = macd - macd_signal
    d['MACD']             = macd
    d['MACD_Signal']      = macd_signal
    d['MACD_Hist']        = macd_hist
    d['MACD_norm']        = macd / (d['Close'] + 1e-8)
    d['MACD_Signal_norm'] = macd_signal / (d['Close'] + 1e-8)
    d['MACD_Hist_norm']   = macd_hist / (d['Close'] + 1e-8)

    # Bollinger Bands
    bb_mid   = d['Close'].rolling(20).mean()
    bb_std   = d['Close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    d['BB_mid']      = bb_mid
    d['BB_upper']    = bb_upper
    d['BB_lower']    = bb_lower
    d['BB_width']    = (bb_upper - bb_lower) / (bb_mid + 1e-8)
    d['BB_position'] = (d['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)

    d['Volume_Change'] = d['Volume'].pct_change()
    d['Price_Range']   = (d['High'] - d['Low']) / (d['Close'] + 1e-8)

    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    return d.dropna()

# ── Helper: Figure to Base64 PNG ──────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#0d1117')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

# ── Chart Generators ──────────────────────────────────────────────────────────
def chart_price_trend(df):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df.index, df['Close'], color='#58a6ff', lw=1.6, label='Close Price')
    ax.fill_between(df.index, df['Close'], df['Close'].min(), alpha=0.08, color='#58a6ff')
    if 'MA_10' in df.columns:
        ax.plot(df.index, df['MA_10'], color='#ffa657', lw=1.1, ls='--', label='10-Day MA')
    if 'MA_30' in df.columns:
        ax.plot(df.index, df['MA_30'], color='#d2a8ff', lw=1.1, ls='--', label='30-Day MA')
    if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
        ax.fill_between(df.index, df['BB_upper'], df['BB_lower'], color='#8b949e', alpha=0.12, label='Bollinger Bands (20,2)')
    ax.set_title('Asset Price Dynamics + Moving Average & Volatility Envelope', color='#58a6ff')
    ax.set_ylabel('Price')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_backtest(df_feat, cum_strategy, cum_bh):
    fig, ax = plt.subplots(figsize=(13, 4))
    idx = df_feat.index[:len(cum_strategy)]
    strat_pct = (cum_strategy - 1) * 100
    bh_pct    = (cum_bh - 1) * 100

    ax.plot(idx, strat_pct, color='#3fb950', lw=2.0, label='ML Long/Cash Strategy')
    ax.plot(idx, bh_pct,    color='#8b949e', lw=1.4, ls='--', label='Passive Buy & Hold Benchmark')
    ax.axhline(0, color='#30363d', lw=1, ls='-')
    ax.fill_between(idx, strat_pct, 0, alpha=0.15, color='#3fb950')
    ax.set_title('Quantitative Backtest: Cumulative Strategy Alpha vs. Benchmark (%)', color='#58a6ff')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend(fontsize=9, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_predictions(df_feat, preds, actual_direction):
    close = df_feat['Close']
    idx   = df_feat.index
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # Upper panel: Price overlaid with prediction points
    ax = axes[0]
    ax.plot(idx, close, color='#58a6ff', lw=1.3, label='Close Price', zorder=2)
    up_idx   = idx[preds == 1]
    down_idx = idx[preds == 0]
    ax.scatter(up_idx,   close.loc[up_idx],   color='#3fb950', s=14, alpha=0.7, label='Signal: Long (UP)', zorder=3)
    ax.scatter(down_idx, close.loc[down_idx], color='#f78166', s=14, alpha=0.7, label='Signal: Flat/Short (DOWN)', zorder=3)
    ax.set_ylabel('Price')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('Sequential Next-Day Machine Learning Directional Signals', color='#58a6ff')

    # Lower panel: Daily returns colored by prediction
    returns = close.pct_change().fillna(0) * 100
    bar_colors = ['#3fb950' if p == 1 else '#f78166' for p in preds]
    ax2 = axes[1]
    ax2.bar(idx, returns, color=bar_colors, alpha=0.70, width=1)
    ax2.axhline(0, color='#8b949e', lw=0.8)
    ax2.set_ylabel('Daily Return (%)')
    ax2.set_xlabel('Date')
    ax2.set_title('Realized Asset Return Profile (Bar color = Predicted Stance)', color='#58a6ff')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)

    for a in axes:
        a.set_facecolor('#161b22')
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_rsi(df):
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(df.index, df['RSI_14'], color='#3fb950', lw=1.2)
    ax.axhline(70, color='#f78166', ls='--', lw=0.9, label='Overbought Threshold (70)')
    ax.axhline(30, color='#58a6ff', ls='--', lw=0.9, label='Oversold Threshold (30)')
    ax.fill_between(df.index, df['RSI_14'], 70, where=df['RSI_14'] >= 70, alpha=0.20, color='#f78166')
    ax.fill_between(df.index, df['RSI_14'], 30, where=df['RSI_14'] <= 30, alpha=0.20, color='#58a6ff')
    ax.set_ylim(0, 100)
    ax.set_title('RSI — Relative Strength Index (14-Period Momentum Oscillator)', color='#58a6ff')
    ax.set_ylabel('RSI (0-100)')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_macd(df):
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(df.index, df['MACD'],        color='#58a6ff', lw=1.2, label='MACD Line')
    ax.plot(df.index, df['MACD_Signal'], color='#ffa657', lw=1.2, label='Signal Line')
    hist_colors = ['#3fb950' if v >= 0 else '#f78166' for v in df['MACD_Hist']]
    ax.bar(df.index, df['MACD_Hist'], color=hist_colors, alpha=0.55, width=1, label='Convergence Histogram')
    ax.axhline(0, color='#8b949e', lw=0.8)
    ax.set_title('MACD — Moving Average Convergence Divergence', color='#58a6ff')
    ax.set_ylabel('MACD')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_volume(df_raw, df_feat):
    fig, ax = plt.subplots(figsize=(13, 3))
    shared_idx = df_raw.index.intersection(df_feat.index)
    sub = df_raw.loc[shared_idx]
    vol_colors = ['#3fb950' if c >= o else '#f78166' for c, o in zip(sub['Close'], sub['Open'])]
    ax.bar(sub.index, sub['Volume'], color=vol_colors, alpha=0.75, width=1)
    ax.set_title('Trading Volume Distribution (Green = Positive Session, Red = Negative Session)', color='#58a6ff')
    ax.set_ylabel('Volume')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=30)
    plt.tight_layout()
    return fig_to_b64(fig)

def chart_return_dist(df_feat):
    returns = df_feat['Daily_Return'].dropna() * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(returns, ax=ax, bins=50, color='#58a6ff', alpha=0.7, kde=True, line_kws={'lw': 2})
    ax.axvline(returns.mean(), color='white', ls='--', lw=1.2, label=f'Empirical Mean = {returns.mean():.2f}%')
    ax.axvline(0, color='#8b949e', ls='-', lw=0.8)
    ax.set_title('Empirical Daily Return Distribution & KDE Fit', color='#58a6ff')
    ax.set_xlabel('Daily Return (%)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig_to_b64(fig)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker_input = request.form.get('ticker', '').strip().upper()
    period_input = request.form.get('period', '2y').strip()

    df_raw = None
    asset_name = ""

    # Mode 1: Live Ticker Fetch via yfinance
    if ticker_input:
        asset_name = ticker_input
        try:
            df_raw = yf.download(ticker_input, period=period_input, progress=False, auto_adjust=True)
            if df_raw.empty:
                return jsonify({'error': f"Ticker '{ticker_input}' returned no data. Check symbol (e.g. AAPL, NVDA, RELIANCE.NS)."}), 400
            df_raw.columns = [c[0] if isinstance(c, tuple) else c for c in df_raw.columns]
            df_raw.index.name = 'Date'
        except Exception as e:
            return jsonify({'error': f"Failed to download market data for {ticker_input}: {e}"}), 400

    # Mode 2: File Upload (CSV)
    elif 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        asset_name = file.filename
        try:
            df_raw = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Could not read CSV file: {e}'}), 400

        # Identify Date column
        date_col = None
        for col in df_raw.columns:
            if col.strip().lower() in ('date', 'timestamp', 'datetime', 'time'):
                date_col = col
                break
        if date_col is None:
            return jsonify({'error': 'No Date column found in CSV. Required headers: Date, Open, High, Low, Close, Volume.'}), 400

        df_raw = df_raw.rename(columns={date_col: 'Date'})
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Date']).set_index('Date').sort_index()

        # Case-insensitive column map
        col_map = {c: c.strip().capitalize() for c in df_raw.columns}
        df_raw.rename(columns=col_map, inplace=True)
    else:
        return jsonify({'error': 'Please provide a stock ticker or upload a CSV dataset.'}), 400

    # Verify required price columns
    required = {'Open', 'High', 'Low', 'Close', 'Volume'}
    missing = required - set(df_raw.columns)
    if missing:
        return jsonify({'error': f'Missing columns in dataset: {", ".join(missing)}'}), 400

    df_raw = df_raw[list(required)].apply(pd.to_numeric, errors='coerce')
    df_raw = df_raw.dropna()

    if len(df_raw) < 50:
        return jsonify({'error': f'Dataset contains only {len(df_raw)} valid sessions. Minimum 50 required for indicator burn-in.'}), 400

    # ── Feature Engineering & ML Inference ────────────────────────────────────
    try:
        df_feat = add_features(df_raw)
        if len(df_feat) < 20:
            return jsonify({'error': 'Insufficient rows after indicator rolling burn-in. Provide a larger dataset.'}), 400

        X = df_feat[FEATURES].values.astype(np.float64)

        # Impute any residual non-finite values safely
        X = np.where(np.isfinite(X), X, np.nan)
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(col_means, inds[1])
        X = np.clip(X, -1e6, 1e6)

        preds  = MODEL.predict(X)
        probas = MODEL.predict_proba(X)[:, 1]
    except Exception as e:
        return jsonify({'error': f'Prediction pipeline error: {e}'}), 500

    # ── Quantitative Backtesting (Long/Cash Model Strategy vs. Buy & Hold) ────
    actual_next_close = df_feat['Close'].shift(-1)
    actual_direction  = (actual_next_close > df_feat['Close']).astype(int)

    # Historical testable bars exclude the very last row (whose future return is unknown)
    hist_preds  = preds[:-1]
    hist_actual = actual_direction[:-1]

    if len(hist_preds) > 0:
        hist_acc  = float(accuracy_score(hist_actual, hist_preds) * 100)
        hist_prec = float(precision_score(hist_actual, hist_preds, zero_division=0) * 100)
        hist_rec  = float(recall_score(hist_actual, hist_preds, zero_division=0) * 100)

        # Strategy Return = position (1 for Long, 0 for Cash) * realized next-day return
        realized_next_return = df_feat['Daily_Return'].shift(-1).iloc[:-1]
        strategy_daily_ret   = hist_preds * realized_next_return
        bh_daily_ret         = realized_next_return

        cum_strat = (1 + strategy_daily_ret).cumprod()
        cum_bh    = (1 + bh_daily_ret).cumprod()

        strat_return = float((cum_strat.iloc[-1] - 1) * 100)
        bh_return    = float((cum_bh.iloc[-1] - 1) * 100)

        # Annualized Sharpe Ratio (assuming 0 risk-free rate)
        strat_std = float(strategy_daily_ret.std())
        sharpe_ratio = float(np.sqrt(252) * strategy_daily_ret.mean() / (strat_std + 1e-8)) if strat_std > 0 else 0.0

        # Maximum Drawdown
        running_peak = cum_strat.cummax()
        drawdown = (cum_strat - running_peak) / (running_peak + 1e-8)
        max_drawdown = float(drawdown.min() * 100)
    else:
        hist_acc = hist_prec = hist_rec = strat_return = bh_return = sharpe_ratio = max_drawdown = 0.0
        cum_strat = pd.Series([1.0])
        cum_bh    = pd.Series([1.0])

    # ── Latest Day Prediction (For Tomorrow) ──────────────────────────────────
    latest_pred = 'UP ▲' if preds[-1] == 1 else 'DOWN ▼'
    latest_prob = float(probas[-1] * 100)
    up_stance   = bool(preds[-1] == 1)

    # ── Technical Indicator Explainability Signals ────────────────────────────
    latest_rsi = float(df_feat['RSI_14'].iloc[-1])
    if latest_rsi < 30:
        rsi_signal = f"Oversold ({latest_rsi:.1f}) — Bullish Reversal Bias"
    elif latest_rsi > 70:
        rsi_signal = f"Overbought ({latest_rsi:.1f}) — Pullback Risk"
    else:
        rsi_signal = f"Neutral ({latest_rsi:.1f}) — Balanced Momentum"

    latest_macd_hist = float(df_feat['MACD_Hist'].iloc[-1])
    macd_signal = "Bullish Acceleration" if latest_macd_hist > 0 else "Bearish Momentum"

    ma_trend = "Bullish (10-MA > 30-MA)" if df_feat['MA_ratio'].iloc[-1] > 0 else "Bearish (10-MA < 30-MA)"
    ann_vol  = float(df_feat['Volatility_10'].iloc[-1] * np.sqrt(252) * 100)

    # ── Generate Base64 Analytical Charts ─────────────────────────────────────
    charts = {
        'price':       chart_price_trend(df_feat),
        'backtest':    chart_backtest(df_feat, cum_strat, cum_bh),
        'predictions': chart_predictions(df_feat, preds, actual_direction),
        'rsi':         chart_rsi(df_feat),
        'macd':        chart_macd(df_feat),
        'volume':      chart_volume(df_raw, df_feat),
        'dist':        chart_return_dist(df_feat),
    }

    return jsonify({
        'summary': {
            'asset_name':    asset_name,
            'latest_pred':   latest_pred,
            'latest_prob':   round(latest_prob, 1),
            'is_up':         up_stance,
            'hist_acc':      round(hist_acc, 1),
            'hist_prec':     round(hist_prec, 1),
            'hist_rec':      round(hist_rec, 1),
            'strat_return':  round(strat_return, 2),
            'bh_return':     round(bh_return, 2),
            'sharpe_ratio':  round(sharpe_ratio, 2),
            'max_drawdown':  round(max_drawdown, 2),
            'total_days':    len(df_feat),
            'start_date':    str(df_feat.index[0].date()),
            'end_date':      str(df_feat.index[-1].date()),
            'signals': {
                'rsi':        rsi_signal,
                'macd':       macd_signal,
                'trend':      ma_trend,
                'volatility': f"{ann_vol:.1f}% Annualized",
            }
        },
        'charts': charts,
    })

if __name__ == '__main__':
    print("[SERVER] Starting StockSense Flask Server on http://127.0.0.1:5000 ...")
    app.run(debug=True, port=5000)
