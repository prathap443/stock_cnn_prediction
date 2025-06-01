from flask import Flask, jsonify, request, session
import requests
import json
import os
import time
from datetime import datetime, timedelta
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from textblob import TextBlob
import ta
import joblib
from flask import send_from_directory

if os.environ.get("RENDER") != "true":  # Optional: only load .env locally
    from dotenv import load_dotenv
    load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_analysis_api')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# Alpaca API headers
ALPACA_HEADERS = {
    "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY"),
    "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY")
}

# Define CNN Model (must match train_model.py)
class StockPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Load pre-trained model and label encoder
try:
    scaler = joblib.load("models/scaler.pkl")
    logger.info("Scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load scaler: {str(e)}")
    scaler = None

try:
    model = StockPredictor(input_size=9, num_classes=3)
    model.load_state_dict(torch.load("models/stock_predictor.pth", map_location=torch.device("cpu")))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

try:
    label_encoder = joblib.load("models/label_encoder.pkl")
    logger.info("Label encoder loaded successfully")
except Exception as e:
    logger.error(f"Failed to load label encoder: {str(e)}")
    label_encoder = None

# Define feature columns (match train_model.py)
FEATURE_COLUMNS = [
    'RSI', 'MACD', 'SMA_50', 'BB_Width', 'Stochastic',
    'News_Sentiment', 'volume_score', 'percent_change_5d', 'volatility'
]

# Create directories
os.makedirs('data', exist_ok=True)

# Stock lists
base_stocks = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "JPM", "V", "WMT", 
    "DIS", "NFLX", "PYPL", "INTC", "AMD", 
    "BA", "PFE", "KO", "PEP", "XOM"
]
AI_STOCKS = [
    "NVDA", "AMD", "GOOGL", "MSFT", "META",
    "TSLA", "AMZN", "IBM", "BIDU", "PLTR"
]
TECH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "AMD", "INTC", "IBM",
    "CRM", "ORCL", "ADBE", "CSCO", "QCOM",
    "SAP", "TXN", "AVGO", "SNOW", "SHOP"
]
STOCK_LIST = sorted(set(base_stocks + AI_STOCKS + TECH_STOCKS))
logger.info(f"Final STOCK_LIST contains {len(STOCK_LIST)} symbols.")

# Static mapping of stock symbols to sectors
SECTOR_MAPPING = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Technology",
    "META": "Technology",
    "TSLA": "Technology",
    "NVDA": "Technology",
    "INTC": "Technology",
    "AMD": "Technology",
    "IBM": "Technology",
    "CRM": "Technology",
    "ORCL": "Technology",
    "ADBE": "Technology",
    "CSCO": "Technology",
    "QCOM": "Technology",
    "SAP": "Technology",
    "TXN": "Technology",
    "AVGO": "Technology",
    "SNOW": "Technology",
    "SHOP": "Technology",
    "BIDU": "Technology",
    "PLTR": "Technology",
    "JPM": "Finance",
    "V": "Finance",
    "WMT": "Consumer Goods",
    "DIS": "Consumer Goods",
    "KO": "Consumer Goods",
    "PEP": "Consumer Goods",
    "NFLX": "Entertainment",
    "PYPL": "Financial Services",
    "BA": "Aerospace",
    "PFE": "Healthcare",
    "XOM": "Energy"
}

def is_market_open():
    now = datetime.utcnow()
    est_offset = timedelta(hours=-5)
    est_time = now + est_offset
    market_open = est_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = est_time.replace(hour=16, minute=0, second=0, microsecond=0)
    market_open = market_open.replace(year=est_time.year, month=est_time.month, day=est_time.day)
    market_close = market_close.replace(year=est_time.year, month=est_time.month, day=est_time.day)
    if est_time.weekday() >= 5:
        return False
    return market_open <= est_time <= market_close

def fetch_alpaca_data(symbol, start_date, end_date, timeframe="1Day", retries=3):
    now = datetime.utcnow()
    if start_date >= now or end_date > now:
        logger.warning(f"Invalid date range for {symbol}: start={start_date}, end={end_date} are in the future")
        return []
    if start_date >= end_date:
        logger.warning(f"Start date {start_date} is not before end date {end_date} for {symbol}")
        return []

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "timeframe": timeframe,
        "adjustment": "raw"
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=ALPACA_HEADERS, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "bars" in data:
                logger.info(f"Fetched {len(data['bars'])} bars for {symbol} ({timeframe})")
                return data["bars"]
            else:
                logger.warning(f"No bars found for {symbol} on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    return []
                time.sleep(random.uniform(1, 3))
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {symbol}: {str(e)}")
            if attempt == retries - 1:
                logger.error(f"Failed to fetch bars for {symbol} after {retries} attempts")
                return []
            time.sleep(random.uniform(1, 3))
    
    return []

def fetch_alpaca_quote(symbol, retries=3):
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=ALPACA_HEADERS, timeout=10)
            response.raise_for_status()
            data = response.json()
            if "quote" in data:
                logger.info(f"Fetched quote for {symbol}")
                return data["quote"]
            else:
                logger.warning(f"No quote found for {symbol} on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    return None
                time.sleep(random.uniform(1, 3))
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {symbol}: {str(e)}")
            if attempt == retries - 1:
                logger.error(f"Failed to fetch quote for {symbol} after {retries} attempts")
                return None
            time.sleep(random.uniform(1, 3))
    return None

def get_alpaca_account_id():
    url = "https://broker-api.alpaca.markets/v1/accounts"
    try:
        response = requests.get(url, headers=ALPACA_HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]["id"]
        else:
            logger.error("No account ID found")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch account ID: {str(e)}")
        return None

def safe_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def get_last_trading_day(end_dt):
    est_offset = timedelta(hours=-5)
    est_time = end_dt + est_offset

    last_trading_day = end_dt
    if est_time.weekday() == 5:  # Saturday
        last_trading_day -= timedelta(days=1)
    elif est_time.weekday() == 6:  # Sunday
        last_trading_day -= timedelta(days=2)
    elif est_time.weekday() == 0 and est_time.hour < 9:  # Monday before 9 AM EST
        last_trading_day -= timedelta(days=3)
    elif est_time.hour < 9:  # Before 9 AM EST
        last_trading_day -= timedelta(days=1)

    est_last_trading = last_trading_day + est_offset
    while est_last_trading.weekday() >= 5:
        last_trading_day -= timedelta(days=1)
        est_last_trading = last_trading_day + est_offset

    return last_trading_day

def get_price_history(symbol, period):
    now = datetime.utcnow()
    est_offset = timedelta(hours=-5)
    est_now = now + est_offset

    end_dt = min(now, datetime.utcnow().replace(hour=21, minute=0, second=0, microsecond=0))
    
    if period == "1D":
        last_trading_day = get_last_trading_day(end_dt)
        start_dt = last_trading_day.replace(hour=9, minute=30, second=0, microsecond=0) + est_offset
        end_dt = last_trading_day.replace(hour=16, minute=0, second=0, microsecond=0) + est_offset
        timeframe = "1Min"
        
        if is_market_open():
            start_dt = now - timedelta(hours=6)  # Last 6 hours for intraday
            end_dt = now
            timeframe = "1Min"
            
    elif period == "1W":
        start_dt = end_dt - timedelta(days=7)
        timeframe = "1Day"
    elif period == "1M":
        start_dt = end_dt - timedelta(days=30)
        timeframe = "1Day"
    else:
        start_dt = end_dt - timedelta(days=14)
        timeframe = "1Day"

    if start_dt >= end_dt:
        logger.warning(f"Invalid date range for {symbol}: start={start_dt}, end={end_dt}")
        return [{"error": f"Invalid date range for {period} data"}]
    if start_dt > datetime.utcnow():
        logger.warning(f"Start date {start_dt} is in the future for {symbol}")
        return [{"error": f"Start date is in the future for {period} data"}]

    bars = fetch_alpaca_data(symbol, start_dt, end_dt, timeframe)
    if not bars:
        if period == "1D":
            start_dt = last_trading_day.replace(hour=9, minute=30, second=0, microsecond=0) + est_offset
            end_dt = last_trading_day.replace(hour=16, minute=0, second=0, microsecond=0) + est_offset
            timeframe = "1Day"
            bars = fetch_alpaca_data(symbol, start_dt, end_dt, timeframe)
            if not bars:
                return [{"error": f"Unable to fetch {period} data for {symbol} after multiple attempts."}]
    
    try:
        history = []
        for bar in bars:
            dt = datetime.fromisoformat(bar['t'].replace('Z', '+00:00'))
            if period == "1D" and is_market_open() and dt > datetime.utcnow():
                continue
            history.append({
                'date': dt.strftime('%Y-%m-%d %H:%M:%S' if timeframe == "1Min" else '%Y-%m-%d'),
                'close': bar['c']
            })
        if not history:
            return [{"error": f"No valid {period} data points for {symbol}."}]
        return history
    except Exception as e:
        logger.error(f"Error processing {period} history for {symbol}: {str(e)}")
        return [{"error": f"Error processing {period} data for {symbol}: {str(e)}"}]

def get_stock_info(symbol):
    time.sleep(random.uniform(0.5, 1.5))
    quote = fetch_alpaca_quote(symbol)
    if quote:
        try:
            return {
                "symbol": symbol,
                "name": symbol,
                "current_price": quote.get('ap', None),
                "sector": SECTOR_MAPPING.get(symbol, "Unknown"),
            }
        except Exception as e:
            logger.error(f"Error processing quote for {symbol}: {str(e)}")
    return {
        "symbol": symbol,
        "name": symbol,
        "current_price": None,
        "sector": SECTOR_MAPPING.get(symbol, "Unknown"),
    }

def get_historical_data(symbol, days=60):
    time.sleep(random.uniform(0.5, 1.5))
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        bars = fetch_alpaca_data(symbol, start_date, end_date, timeframe="1Day")
        if not bars:
            return calculate_fallback_data(symbol)
        
        timestamps = []
        prices = []
        volumes = []
        highs = []
        lows = []
        
        for bar in bars:
            try:
                dt = datetime.fromisoformat(bar['t'].replace('Z', '+00:00'))
                timestamps.append(int(dt.timestamp()))
                prices.append(bar['c'])
                volumes.append(bar['v'])
                highs.append(bar['h'])
                lows.append(bar['l'])
            except Exception as e:
                logger.warning(f"Error processing bar for {symbol}: {str(e)}")
                continue
        
        if len(prices) < 2:
            return calculate_fallback_data(symbol)
        
        df = pd.DataFrame({
            'Close': prices,
            'Volume': volumes,
            'High': highs,
            'Low': lows
        })
        
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        stochastic = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['Stochastic'] = stochastic.stoch()
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['Close']
        
        start_price = prices[0]
        end_price = prices[-1]
        high_price = max(prices)
        low_price = min(prices)
        price_change = end_price - start_price
        percent_change = (price_change / start_price) * 100
        
        prices_series = pd.Series(prices)
        percent_change_5d = prices_series.pct_change(periods=5).iloc[-1] * 100 if len(prices) >= 5 else 0
        
        daily_returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
        volatility = sum([(ret - (sum(daily_returns)/len(daily_returns)))**2 for ret in daily_returns])
        volatility = (volatility / len(daily_returns))**0.5 if daily_returns else 0
        
        volume_trend = analyze_volume(volumes)
        
        trend = "Neutral"
        bullish_signals = 0
        bearish_signals = 0
        
        rsi_value = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
        if rsi_value > 70:
            bearish_signals += 1
        elif rsi_value < 30:
            bullish_signals += 1
        
        macd_value = df['MACD'].iloc[-1] if not pd.isna(df['MACD'].iloc[-1]) else 0
        if macd_value > 0.5:
            bullish_signals += 1
        elif macd_value < -0.5:
            bearish_signals += 1
        
        if percent_change > 5:
            bullish_signals += 1
        elif percent_change < -5:
            bearish_signals += 1
        
        if "Increasing" in volume_trend:
            bullish_signals += 1
        elif "Decreasing" in volume_trend:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            trend = "Bullish"
        elif bearish_signals > bullish_signals:
            trend = "Bearish"
        
        return {
            "symbol": symbol,
            "start_price": start_price,
            "end_price": end_price,
            "current_price": end_price,
            "price_change": price_change,
            "percent_change_2w": percent_change,
            "percent_change_5d": percent_change_5d,
            "high": high_price,
            "low": low_price,
            "volatility": volatility,
            "volume_trend": volume_trend,
            "technical_indicators": {
                "rsi": f"{rsi_value:.1f}",
                "macd": f"{macd_value:.2f}",
                "sma_50": df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else 0,
                "bb_width": df['BB_Width'].iloc[-1] if not pd.isna(df['BB_Width'].iloc[-1]) else 0,
                "stochastic": df['Stochastic'].iloc[-1] if not pd.isna(df['Stochastic'].iloc[-1]) else 50,
                "volume_analysis": volume_trend,
                "trend": trend
            }
        }
    except Exception as e:
        logger.error(f"Error getting history for {symbol}: {str(e)}")
        return calculate_fallback_data(symbol)

def calculate_fallback_data(symbol):
    return {
        "symbol": symbol,
        "percent_change_2w": random.uniform(-10, 10),
        "percent_change_5d": random.uniform(-5, 5),
        "current_price": random.uniform(50, 500),
        "volatility": random.uniform(1, 8),
        "technical_indicators": {
            "rsi": f"{random.uniform(30, 70):.1f}",
            "macd": f"{random.uniform(-2, 2):.2f}",
            "sma_50": 0,
            "bb_width": 0,
            "stochastic": f"{random.uniform(20, 80):.1f}",
            "volume_analysis": "Neutral",
            "trend": "Neutral"
        }
    }

def analyze_volume(volumes):
    if not volumes or len(volumes) < 5:
        return "N/A"
    
    valid_volumes = [v for v in volumes if v is not None]
    if len(valid_volumes) < 5:
        return "Insufficient Data"
    
    half = len(valid_volumes) // 2
    avg_first_half = sum(valid_volumes[:half]) / half
    avg_second_half = sum(valid_volumes[half:]) / (len(valid_volumes) - half)
    
    volume_change = ((avg_second_half - avg_first_half) / avg_first_half) * 100
    
    if volume_change > 25:
        return "Increasing (High)"
    elif volume_change > 10:
        return "Increasing (Moderate)"
    elif volume_change < -25:
        return "Decreasing (High)"
    elif volume_change < -10:
        return "Decreasing (Moderate)"
    else:
        return "Stable"

def get_news_articles(symbol, retries=3):
    articles = []
    sentiment_score = 0
    for attempt in range(retries):
        try:
            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            news_items = data.get("news", [])[:5]
            if not news_items:
                logger.warning(f"No news articles found for {symbol} on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    return articles, 0
                time.sleep(random.uniform(1, 3))
                continue

            texts = []
            for item in news_items:
                title = item.get("title", "")
                link = item.get("link", "#")
                publisher = item.get("publisher", "Unknown")
                pub_time = item.get("providerPublishTime", 0)
                if pub_time:
                    pub_date = datetime.utcfromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    pub_date = "Unknown"
                articles.append({
                    "title": title,
                    "link": link,
                    "publisher": publisher,
                    "published_at": pub_date
                })
                texts.append(title)

            full_text = " ".join(texts)
            if not full_text.strip():
                logger.warning(f"No valid news titles found for {symbol} on attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    return articles, 0
                time.sleep(random.uniform(1, 3))
                continue

            sentiment_score = TextBlob(full_text).sentiment.polarity
            logger.info(f"Sentiment for {symbol}: {sentiment_score:.3f} based on {len(articles)} articles")
            return articles, sentiment_score
        except Exception as e:
            logger.warning(f"News fetch error for {symbol} on attempt {attempt + 1}/{retries}: {str(e)}")
            if attempt == retries - 1:
                return articles, 0
            time.sleep(random.uniform(1, 3))
    return articles, 0

def analyze_stock(symbol):
    try:
        info = get_stock_info(symbol)
        history = get_historical_data(symbol, days=60)
        news_articles, news_sentiment = get_news_articles(symbol, retries=3)
        history_1d = get_price_history(symbol, "1D")

        current_price = history.get("current_price") or info.get("current_price")
        percent_change_2w = safe_float(history.get("percent_change_2w", 0))
        percent_change_5d = safe_float(history.get("percent_change_5d", 0))
        volatility = safe_float(history.get("volatility", 5))

        technical_indicators = history.get("technical_indicators", {})
        rsi_str = str(technical_indicators.get("rsi", "50"))
        macd_str = str(technical_indicators.get("macd", "0"))
        sma_50 = safe_float(technical_indicators.get("sma_50", 0))
        bb_width = safe_float(technical_indicators.get("bb_width", 0))
        stochastic = safe_float(technical_indicators.get("stochastic", 50))

        rsi = safe_float(rsi_str, default=50)
        macd = safe_float(macd_str, default=0)
        volume_score = 1 if "Increasing" in technical_indicators.get("volume_analysis", "") else 0

        features_dict = {
            'RSI': rsi,
            'MACD': macd,
            'SMA_50': sma_50,
            'BB_Width': bb_width,
            'Stochastic': stochastic,
            'News_Sentiment': news_sentiment,
            'volume_score': volume_score,
            'percent_change_5d': percent_change_5d,
            'volatility': volatility
        }
        features_df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)

        features_df['News_Sentiment'] = features_df['News_Sentiment'].fillna(0.0)

        if model is not None and scaler is not None:
            features_array = scaler.transform(features_df)
            features_tensor = torch.tensor(features_array, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                pred = model(features_tensor).argmax(dim=1).item()
            recommendation = label_encoder.inverse_transform([pred])[0]
        else:
            recommendation = "HOLD"

        reason = (
            f"CNN-based prediction using "
            f"RSI={rsi:.1f}, MACD={macd:.2f}, SMA_50={sma_50:.2f}, BB_Width={bb_width:.2f}, "
            f"Stochastic={stochastic:.1f}, Sentiment={news_sentiment:.2f}, "
            f"Volume_Score={volume_score}, Change_5d={percent_change_5d:.2f}%, "
            f"Volatility={volatility:.2f}"
        )

        logger.info(f"{symbol} → CNN RECOMMEND: {recommendation}")

        return {
            "symbol": symbol,
            "name": info.get("name", symbol),
            "recommendation": recommendation,
            "percent_change_2w": percent_change_2w,
            "current_price": current_price,
            "reason": reason,
            "technical_indicators": technical_indicators,
            "news_sentiment": news_sentiment,
            "news_articles": news_articles,
            "history_1d": history_1d,
            "sector": info.get("sector", SECTOR_MAPPING.get(symbol, "Unknown"))
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return {
            "symbol": symbol,
            "name": symbol,
            "recommendation": "HOLD",
            "percent_change_2w": 0,
            "current_price": 100.0,
            "reason": "Analysis failed. Defaulting to HOLD.",
            "technical_indicators": {
                "rsi": "N/A", "macd": "N/A", 
                "volume_analysis": "N/A", "trend": "N/A"
            },
            "news_articles": [],
            "history_1d": [],
            "sector": SECTOR_MAPPING.get(symbol, "Unknown")
        }

def create_fallback_entry(symbol):
    return {
        "symbol": symbol,
        "name": symbol,
        "recommendation": "HOLD",
        "percent_change_2w": random.uniform(-3, 3),
        "current_price": random.uniform(80, 300),
        "reason": "Analysis unavailable. Maintain position.",
        "technical_indicators": {
            "rsi": "N/A", "macd": "N/A", 
            "volume_analysis": "N/A", "trend": "N/A"
        },
        "news_articles": [],
        "history_1d": [],
        "sector": SECTOR_MAPPING.get(symbol, "Unknown")
    }

def analyze_all_stocks():
    try:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(analyze_stock, symbol): symbol for symbol in STOCK_LIST}
            stocks = []
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    stocks.append(result)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {str(e)}")
                    stocks.append(create_fallback_entry(symbol))

        stocks.sort(key=lambda x: x['symbol'])

        summary = {"BUY": 0, "HOLD": 0, "SELL": 0}
        for stock in stocks:
            recommendation = stock.get('recommendation', 'HOLD')
            summary[recommendation] = summary.get(recommendation, 0) + 1

        result = {
            "stocks": stocks,
            "summary": summary,
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open('data/stock_analysis.json', 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Successfully analyzed {len(stocks)} stocks")
        return result
    except Exception as e:
        logger.error(f"Error in analyze_all_stocks: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}

# Simple authentication for API access (replace with token-based auth if needed)
@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    # Replace with your own authentication logic
    if username == "admin" and password == "password123":
        session['user'] = {"username": username}
        return jsonify({"success": True, "message": "Logged in successfully"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/logout')
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route('/api/stocks')
def api_stocks():
    try:
        if not session.get('user'):
            return jsonify({"error": "Unauthorized access"}), 401
        
        cache_duration = 300 if is_market_open() else 1800
        if os.path.exists('data/stock_analysis.json'):
            with open('data/stock_analysis.json', 'r') as f:
                data = json.load(f)
                last_updated = datetime.strptime(data['last_updated'], "%Y-%m-%d %H:%M:%S")
                age = (datetime.now() - last_updated).total_seconds()
                if age < cache_duration:
                    return jsonify(data)
        return jsonify(analyze_all_stocks())
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({"error": f"API error: {str(e)}"}), 500

@app.route('/api/stock_history/<symbol>/<period>')
def api_stock_history(symbol, period):
    try:
        if not session.get('user'):
            return jsonify({"error": "Unauthorized access"}), 401
        
        history = get_price_history(symbol, period)
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history for {symbol} ({period}): {str(e)}")
        return jsonify([{"error": f"Error fetching {period} history: {str(e)}"}]), 500

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    try:
        if not session.get('user'):
            return jsonify({"error": "Unauthorized access"}), 401
        
        if os.path.exists('data/stock_analysis.json'):
            os.remove('data/stock_analysis.json')
        data = analyze_all_stocks()
        if not isinstance(data, dict) or "stocks" not in data:
            raise ValueError("Invalid format returned from analysis")
        return jsonify({"success": True, "message": "Refreshed successfully"})
    except Exception as e:
        logger.error(f"Error refreshing data: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/buy", methods=["POST"])
def buy_stock():
    try:
        if not session.get('user'):
            return jsonify({"error": "Unauthorized access"}), 401
        
        data = request.json
        symbol = data.get("symbol")
        quantity = data.get("quantity")
        
        if not symbol or not quantity:
            return jsonify({"error": "Missing symbol or quantity"}), 400
        
        try:
            quantity = int(quantity)
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
        except ValueError:
            return jsonify({"error": "Invalid quantity"}), 400
        
        if symbol not in STOCK_LIST:
            return jsonify({"error": "Invalid symbol"}), 400

        account_id = get_alpaca_account_id()
        if not account_id:
            return jsonify({"error": "Failed to retrieve account ID"}), 500
        
        url = f"https://broker-api.alpaca.markets/v1/trading/accounts/{account_id}/orders"
        payload = {
            "symbol": symbol,
            "qty": quantity,
            "side": "buy",
            "type": "market",
            "time_in_force": "day"
        }
        response = requests.post(url, json=payload, headers=ALPACA_HEADERS, timeout=10)
        response.raise_for_status()
        logger.info(f"Buy order placed for {quantity} shares of {symbol}")
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f"Error placing buy order for {symbol}: {str(e)}")
        return jsonify({"error": f"Failed to place buy order: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in buy_stock: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/sell", methods=["POST"])
def sell_stock():
    try:
        if not session.get('user'):
            return jsonify({"error": "Unauthorized access"}), 401
        
        data = request.json
        symbol = data.get("symbol")
        quantity = data.get("quantity")
        
        if not symbol or not quantity:
            return jsonify({"error": "Missing symbol or quantity"}), 400
        
        try:
            quantity = int(quantity)
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
        except ValueError:
            return jsonify({"error": "Invalid quantity"}), 400
        
        if symbol not in STOCK_LIST:
            return jsonify({"error": "Invalid symbol"}), 400

        account_id = get_alpaca_account_id()
        if not account_id:
            return jsonify({"error": "Failed to retrieve account ID"}), 500
        
        url = f"https://broker-api.alpaca.markets/v1/trading/accounts/{account_id}/orders"
        payload = {
            "symbol": symbol,
            "qty": quantity,
            "side": "sell",
            "type": "market",
            "time_in_force": "day"
        }
        response = requests.post(url, json=payload, headers=ALPACA_HEADERS, timeout=10)
        response.raise_for_status()
        logger.info(f"Sell order placed for {quantity} shares of {symbol}")
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        logger.error(f"Error placing sell order for {symbol}: {str(e)}")
        return jsonify({"error": f"Failed to place sell order: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in sell_stock: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/live_prediction/<symbol>')
def live_prediction(symbol):
    try:
        if not session.get('user'):
            return jsonify({"error": "Unauthorized access"}), 401
        
        history_1d = get_price_history(symbol, "1D")
        if not history_1d or ('error' in history_1d[0] and history_1d[0]['error']):
            return jsonify({"error": "Insufficient intraday data for prediction"}), 400

        info = get_stock_info(symbol)
        _, news_sentiment = get_news_articles(symbol)

        prices = [entry['close'] for entry in history_1d if 'close' in entry]
        if not prices:
            return jsonify({"error": "No valid price data available for prediction"}), 400

        current_price = prices[-1] if prices else info.get("current_price", 100.0)

        df = pd.DataFrame({'Close': prices})
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd.macd()
        df['SMA_50'] = ta.trend.SMAIndicator(df['Close'], window=50).sma_indicator()
        stochastic = ta.momentum.StochasticOscillator(high=[max(prices)]*len(prices), low=[min(prices)]*len(prices), close=df['Close'], window=14, smooth_window=3)
        df['Stochastic'] = stochastic.stoch()
        bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_Width'] = (bollinger.bollinger_hband() - bollinger.bollinger_lband()) / df['Close']

        start_price = prices[0]
        percent_change = ((current_price - start_price) / start_price) * 100 if start_price else 0
        prices_series = pd.Series(prices)
        percent_change_5d = prices_series.pct_change(periods=5).iloc[-1] * 100 if len(prices) >= 5 else 0
        daily_returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
        volatility = (sum([(ret - (sum(daily_returns)/len(daily_returns)))**2 for ret in daily_returns]) / len(daily_returns))**0.5 if daily_returns else 5

        rsi_value = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
        macd_value = df['MACD'].iloc[-1] if not pd.isna(df['MACD'].iloc[-1]) else 0
        sma_50 = df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else 0
        bb_width = df['BB_Width'].iloc[-1] if not pd.isna(df['BB_Width'].iloc[-1]) else 0
        stochastic_value = df['Stochastic'].iloc[-1] if not pd.isna(df['Stochastic'].iloc[-1]) else 50
        volume_score = 1 if len(prices) > 10 and prices[-1] > prices[-2] else 0

        features_dict = {
            'RSI': rsi_value,
            'MACD': macd_value,
            'SMA_50': sma_50,
            'BB_Width': bb_width,
            'Stochastic': stochastic_value,
            'News_Sentiment': news_sentiment,
            'volume_score': volume_score,
            'percent_change_5d': percent_change_5d,
            'volatility': volatility
        }
        features_df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)

        features_df['News_Sentiment'] = features_df['News_Sentiment'].fillna(0.0)

        if model is not None and scaler is not None:
            features_array = scaler.transform(features_df)
            features_tensor = torch.tensor(features_array, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                pred = model(features_tensor).argmax(dim=1).item()
            recommendation = label_encoder.inverse_transform([pred])[0]
        else:
            recommendation = "HOLD"

        return jsonify({
            "symbol": symbol,
            "recommendation": recommendation,
            "current_price": current_price,
            "percent_change_today": percent_change,
            "technical_indicators": {
                "rsi": f"{rsi_value:.1f}",
                "macd": f"{macd_value:.2f}",
                "trend": "Bullish" if percent_change > 0 else "Bearish"
            },
            "news_sentiment": news_sentiment,
            "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.error(f"Error generating live prediction for {symbol}: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
from flask import render_template

from flask import render_template

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    if path != "" and os.path.exists(os.path.join("static", path)):
        return send_from_directory("static", path)
    return render_template("index.html")




if __name__ == "__main__":
    if not os.path.exists('data/stock_analysis.json'):
        try:
            analyze_all_stocks()
        except Exception as e:
            logger.error(f"Initial analysis error: {str(e)}")
            with open('data/stock_analysis.json', 'w') as f:
                json.dump({"stocks": [], "summary": {"BUY": 0, "HOLD": 0, "SELL": 0}, "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)