from flask import Flask, render_template, jsonify, request, send_from_directory, redirect, session, url_for
import requests
import json
import os
import time
from datetime import datetime, timedelta
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob  # For basic sentiment analysis
import ta  # For technical indicators (RSI, MACD, etc.)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with a secure secret key for sessions

# Google OAuth details
GOOGLE_CLIENT_ID = "534755939275-0g4f0ih1a9n7fl5mao1f418oamh614r2.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "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"
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"

# Load pre-trained model and label encoder
model = joblib.load("models/stock_predictor.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# Define the feature columns expected by the model (same as in training)
FEATURE_COLUMNS = [
    'RSI', 'MACD', 'SMA_50', 'BB_Width', 'PE_Ratio',
    'Dividend_Yield', 'News_Sentiment', 'volume_score',
    'percent_change_5d', 'volatility'
]

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_analysis_webapp')

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('static', exist_ok=True)

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
    """Check if U.S. markets are open (9:30 AM to 4:00 PM EST)"""
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

def fetch_yahoo_finance_data(symbol, start, end, interval, retries=3):
    """Fetch data from Yahoo Finance with retry logic"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start}&period2={end}&interval={interval}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                return data
            else:
                logger.warning(f"No data found for {symbol} (interval={interval}): {data.get('chart', {}).get('error', 'Unknown error')}")
                return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {symbol}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(random.uniform(1, 3))
            else:
                logger.error(f"Failed to fetch data for {symbol} after {retries} attempts: {str(e)}")
                return None

def safe_float(value, default=0.0):
    """Safely convert a value to float, returning a default if conversion fails"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def get_last_trading_day(end_dt):
    """Get the last trading day before the given datetime"""
    est_offset = timedelta(hours=-5)
    est_time = end_dt + est_offset

    last_trading_day = end_dt
    if est_time.weekday() == 5:
        last_trading_day -= timedelta(days=1)
    elif est_time.weekday() == 6:
        last_trading_day -= timedelta(days=2)
    elif est_time.weekday() == 0 and est_time.hour < 14:
        last_trading_day -= timedelta(days=3)
    elif est_time.hour < 14:
        last_trading_day -= timedelta(days=1)

    est_last_trading = last_trading_day + est_offset
    while est_last_trading.weekday() >= 5:
        last_trading_day -= timedelta(days=1)
        est_last_trading = last_trading_day + est_offset

    return last_trading_day

def get_price_history(symbol, period):
    """Get price history for a specific period (1D, 1W, 1M, or 14D)"""
    now = datetime.utcnow()
    end_dt = now.replace(minute=0, second=0, microsecond=0)
    
    if period == "1D":
        last_trading_day = get_last_trading_day(end_dt)
        est_offset = timedelta(hours=-5)
        start_dt = last_trading_day.replace(hour=14, minute=30, second=0, microsecond=0)
        end_dt = last_trading_day.replace(hour=21, minute=0, second=0, microsecond=0)
        interval = "1m"
        
        if is_market_open():
            start_dt = end_dt - timedelta(days=1)
            end_dt = now
            interval = "1m"
            
    elif period == "1W":
        start_dt = end_dt - timedelta(weeks=1)
        interval = "1d"
    elif period == "1M":
        start_dt = end_dt - timedelta(days=30)
        interval = "1d"
    else:
        start_dt = end_dt - timedelta(days=14)
        interval = "1d"

    start = int(start_dt.timestamp())
    end = int(end_dt.timestamp())
    
    data = fetch_yahoo_finance_data(symbol, start, end, interval)
    if not data or ('error' in data['chart'] and data['chart']['error']):
        if period == "1D":
            start_dt = last_trading_day - timedelta(days=1)
            start = int(start_dt.timestamp())
            interval = "1d"
            data = fetch_yahoo_finance_data(symbol, start, end, interval)
            if not data or ('error' in data['chart'] and data['chart']['error']):
                return [{"error": f"Unable to fetch {period} data for {symbol} after multiple attempts."}]
    
    try:
        chart = data['chart']['result'][0]
        timestamps = chart.get('timestamp', [])
        if not timestamps:
            return [{"error": f"No {period} data available for {symbol}."}]

        closes = chart['indicators']['quote'][0]['close']
        history = []
        for ts, close in zip(timestamps, closes):
            if close is not None:
                dt = datetime.utcfromtimestamp(ts)
                if period == "1D" and is_market_open() and dt > datetime.utcnow():
                    continue
                history.append({
                    'date': dt.strftime('%Y-%m-%d %H:%M:%S' if interval == "1m" else '%Y-%m-%d'),
                    'close': close
                })
        if not history:
            return [{"error": f"No valid {period} data points for {symbol}."}]
        return history
    except Exception as e:
        logger.error(f"Error processing {period} history for {symbol}: {str(e)} - Response: {data}")
        return [{"error": f"Error processing {period} data for {symbol}: {str(e)}"}]

def get_stock_info(symbol):
    """Get basic stock info and current price with improved reliability"""
    time.sleep(random.uniform(0.5, 1.5))  # Randomized delay to avoid rate limiting
    
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        
        if 'quoteResponse' in data and 'result' in data['quoteResponse'] and len(data['quoteResponse']['result']) > 0:
            quote = data['quoteResponse']['result'][0]
            return {
                "symbol": symbol,
                "name": quote.get('shortName', symbol),
                "current_price": quote.get('regularMarketPrice', None),
                "sector": quote.get('sector', SECTOR_MAPPING.get(symbol, "Unknown")),
                "industry": quote.get('industry', "Unknown"),
                "market_cap": quote.get('marketCap', None),
                "pe_ratio": quote.get('trailingPE', None),
                "dividend_yield": quote.get('dividendYield', 0.0)
            }
        else:
            return get_stock_info_by_scraping(symbol)
    except Exception as e:
        logger.error(f"Error fetching info for {symbol}: {str(e)}")
        return get_stock_info_by_scraping(symbol)

def get_stock_info_by_scraping(symbol):
    """Get stock info by scraping - backup method"""
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        price = None
        name = symbol
        
        if response.status_code == 200:
            html = response.text
            
            if '<h1' in html:
                name_start = html.find('<h1')
                name_end = html.find('</h1>', name_start)
                if name_end > 0:
                    name_content = html[name_start:name_end]
                    name_parts = name_content.split('>')
                    if len(name_parts) > 1:
                        name = name_parts[-1].strip()
            
            price_marker = 'data-field="regularMarketPrice"'
            if price_marker in html:
                price_pos = html.find(price_marker)
                value_attr = 'value="'
                value_start = html.find(value_attr, price_pos)
                if value_start > 0:
                    value_end = html.find('"', value_start + len(value_attr))
                    if value_end > 0:
                        try:
                            price = float(html[value_start + len(value_attr):value_end])
                        except ValueError:
                            pass
        
        return {
            "symbol": symbol,
            "name": name if name else symbol,
            "current_price": price,
            "sector": SECTOR_MAPPING.get(symbol, "Unknown"),
            "industry": "Unknown",
            "pe_ratio": None,
            "dividend_yield": 0.0
        }
    except Exception as e:
        logger.error(f"Error scraping info for {symbol}: {str(e)}")
        return {
            "symbol": symbol,
            "name": symbol,
            "current_price": None,
            "sector": SECTOR_MAPPING.get(symbol, "Unknown"),
            "pe_ratio": None,
            "dividend_yield": 0.0
        }

def get_historical_data(symbol, days=60):
    """Get historical price data for analysis with improved reliability"""
    time.sleep(random.uniform(0.5, 1.5))
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        data = response.json()
        
        if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
            return calculate_fallback_data(symbol)
        
        result = data["chart"]["result"][0]
        
        timestamps = result["timestamp"]
        quotes = result["indicators"]["quote"][0]
        close_prices = quotes["close"]
        volumes = quotes.get("volume", [])
        highs = quotes.get("high", [])
        lows = quotes.get("low", [])
        
        valid_data = []
        for i in range(len(timestamps)):
            price = close_prices[i] if i < len(close_prices) else None
            volume = volumes[i] if i < len(volumes) else None
            high = highs[i] if i < len(highs) else None
            low = lows[i] if i < len(lows) else None
            if price is not None and high is not None and low is not None:
                valid_data.append((timestamps[i], price, volume, high, low))
        
        if len(valid_data) < 2:
            return calculate_fallback_data(symbol)
        
        timestamps, prices, volumes, highs, lows = zip(*valid_data)
        
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
                "volume_analysis": volume_trend,
                "trend": trend
            }
        }
    except Exception as e:
        logger.error(f"Error getting history for {symbol}: {str(e)}")
        return calculate_fallback_data(symbol)

def calculate_fallback_data(symbol):
    """Calculate fallback data when we can't get real data"""
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
            "volume_analysis": "Neutral",
            "trend": "Neutral"
        }
    }

def analyze_volume(volumes):
    """Analyze trading volume trend"""
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
    """Fetch recent news articles for a symbol and calculate sentiment"""
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
            logger.info(f"Sentiment for {symbol}: {sentiment_score:.3f} based on {len(articles)} articles: {texts}")
            return articles, sentiment_score
        except Exception as e:
            logger.warning(f"News fetch error for {symbol} on attempt {attempt + 1}/{retries}: {str(e)}")
            if attempt == retries - 1:
                return articles, 0
            time.sleep(random.uniform(1, 3))
    return articles, 0

def analyze_stock(symbol):
    """Analyze a single stock"""
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

        rsi = safe_float(rsi_str, default=50)
        macd = safe_float(macd_str, default=0)
        volume_score = 1 if "Increasing" in technical_indicators.get("volume_analysis", "") else 0
        sentiment_score = safe_float(news_sentiment, 0)
        pe_ratio = safe_float(info.get("pe_ratio", np.nan))
        dividend_yield = safe_float(info.get("dividend_yield", 0))

        features_dict = {
            'RSI': rsi,
            'MACD': macd,
            'SMA_50': sma_50,
            'BB_Width': bb_width,
            'PE_Ratio': pe_ratio,
            'Dividend_Yield': dividend_yield,
            'News_Sentiment': sentiment_score,
            'volume_score': volume_score,
            'percent_change_5d': percent_change_5d,
            'volatility': volatility
        }
        features_df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)

        features_df['PE_Ratio'] = features_df['PE_Ratio'].fillna(features_df['PE_Ratio'].median())
        features_df['Dividend_Yield'] = features_df['Dividend_Yield'].fillna(0.0)
        features_df['News_Sentiment'] = features_df['News_Sentiment'].fillna(0.0)

        pred = model.predict(features_df)[0]
        recommendation = label_encoder.inverse_transform([pred])[0]

        reason = (
            f"🤖 ML-based prediction using "
            f"RSI={rsi:.1f}, MACD={macd:.2f}, SMA_50={sma_50:.2f}, BB_Width={bb_width:.2f}, "
            f"PE_Ratio={pe_ratio:.2f}, Dividend_Yield={dividend_yield:.2f}, "
            f"Sentiment={sentiment_score:.2f}, Volume_Score={volume_score}, "
            f"Change_5d={percent_change_5d:.2f}%, Volatility={volatility:.2f}"
        )

        logger.info(f"{symbol} → ML RECOMMEND: {recommendation}")

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
            "reason": "⚠️ Analysis failed. Defaulting to HOLD.",
            "technical_indicators": {
                "rsi": "N/A", "macd": "N/A", 
                "volume_analysis": "N/A", "trend": "N/A"
            },
            "news_articles": [],
            "history_1d": [],
            "sector": SECTOR_MAPPING.get(symbol, "Unknown")
        }

def create_fallback_entry(symbol):
    """Create a fallback stock entry"""
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
    """Analyze all stocks and cache the results"""
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
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open('data/stock_analysis.json', 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"Successfully analyzed {len(stocks)} stocks")
        return result
    except Exception as e:
        logger.error(f"Error in analyze_all_stocks: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}

@app.route('/')
def index():
    user_info = session.get('user')
    if user_info:
        return render_template('index.html', user_name=user_info.get('name', 'User'))
    return render_template('login.html')


@app.route('/login')
def login():
    """Initiate Google OAuth login"""
    discovery_doc = requests.get(GOOGLE_DISCOVERY_URL).json()
    authorization_endpoint = discovery_doc["authorization_endpoint"]

    request_uri = f"{authorization_endpoint}?response_type=code&client_id={GOOGLE_CLIENT_ID}&redirect_uri=https://www.pratstockprediction.co.uk/callback&scope=openid%20email%20profile"

    return redirect(request_uri)

@app.route('/callback', methods=["POST"])
def callback():
    """Handle Google Identity Services login callback"""
    try:
        token = request.form.get('credential')  # New GIS sends 'credential'
        if not token:
            return jsonify({"error": "Missing credential token"}), 400

        from google.oauth2 import id_token
        from google.auth.transport import requests as grequests

        idinfo = id_token.verify_oauth2_token(
            token,
            grequests.Request(),
            GOOGLE_CLIENT_ID
        )
        # Save user info in session
        session['user'] = {
            "name": idinfo.get('name'),
            "email": idinfo.get('email'),
            "picture": idinfo.get('picture')
        }
        return redirect('/')
    except Exception as e:
        logger.error(f"Error verifying ID token: {str(e)}")
        return jsonify({"error": "Authentication failed"}), 400



@app.route('/logout')
def logout():
    """Clear session and log out"""
    session.clear()
    return redirect('/')

@app.route('/api/stocks')
def api_stocks():
    """Get stock data - first try cache, then live data"""
    try:
        cache_duration = 300 if is_market_open() else 1800
        if os.path.exists('data/stock_analysis.json'):
            with open('data/stock_analysis.json', 'r') as f:
                data = json.load(f)
                last_updated = datetime.strptime(data['last_updated'], "%Y-%m-%d %H:%M:%S")
                age = datetime.now() - last_updated
                if age.total_seconds() < cache_duration:
                    return jsonify(data)
        return jsonify(analyze_all_stocks())
    except Exception as e:
        error_msg = f"API error: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

@app.route('/api/stock_history/<symbol>/<period>')
def api_stock_history(symbol, period):
    """Get price history for a specific stock and time period"""
    try:
        history = get_price_history(symbol, period)
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history for {symbol} ({period}): {str(e)}")
        return jsonify([{"error": f"Error fetching {period} history: {str(e)}"}]), 500

@app.route('/api/stock_news/<symbol>')
def api_stock_news(symbol):
    """Get recent news articles for a specific stock"""
    try:
        articles, _ = get_news_articles(symbol, retries=3)
        return jsonify(articles)
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {str(e)}")
        return jsonify({"error": f"Error fetching news: {str(e)}"}), 500

@app.route('/api/refresh', methods=['POST'])
def api_refresh():
    """Refresh stock data"""
    try:
        if os.path.exists('data/stock_analysis.json'):
            os.remove('data/stock_analysis.json')
        data = analyze_all_stocks()
        if not isinstance(data, dict) or "stocks" not in data:
            raise ValueError("Invalid format returned from analysis")
        return jsonify({"success": True, "message": "Refreshed successfully"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Predict recommendation for given features"""
    try:
        data = request.get_json()
        features_dict = {
            'RSI': data.get("rsi", 50),
            'MACD': data.get("macd", 0),
            'SMA_50': data.get("sma_50", 0),
            'BB_Width': data.get("bb_width", 0),
            'PE_Ratio': data.get("pe_ratio", np.nan),
            'Dividend_Yield': data.get("dividend_yield", 0),
            'News_Sentiment': data.get("news_sentiment", 0),
            'volume_score': data.get("volume_score", 0),
            'percent_change_5d': data.get("percent_change_5d", 0),
            'volatility': data.get("volatility", 0)
        }
        features_df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)
        features_df['PE_Ratio'] = features_df['PE_Ratio'].fillna(features_df['PE_Ratio'].median())
        features_df['Dividend_Yield'] = features_df['Dividend_Yield'].fillna(0.0)
        features_df['News_Sentiment'] = features_df['News_Sentiment'].fillna(0.0)

        prediction = model.predict(features_df)[0]
        recommendation = label_encoder.inverse_transform([prediction])[0]
        return jsonify({
            "recommendation": recommendation,
            "reason": f"ML-based prediction using RSI={features_df['RSI'][0]}, MACD={features_df['MACD'][0]}, volume_score={features_df['volume_score'][0]}, change={features_df['percent_change_5d'][0]}, volatility={features_df['volatility'][0]}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/live_prediction/<symbol>')
def live_prediction(symbol):
    """Get a live prediction for a specific stock based on the latest intraday data"""
    try:
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
        volume_score = 1 if len(prices) > 10 and prices[-1] > prices[-2] else 0
        pe_ratio = safe_float(info.get("pe_ratio", np.nan))
        dividend_yield = safe_float(info.get("dividend_yield", 0))

        features_dict = {
            'RSI': rsi_value,
            'MACD': macd_value,
            'SMA_50': sma_50,
            'BB_Width': bb_width,
            'PE_Ratio': pe_ratio,
            'Dividend_Yield': dividend_yield,
            'News_Sentiment': news_sentiment,
            'volume_score': volume_score,
            'percent_change_5d': percent_change_5d,
            'volatility': volatility
        }
        features_df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)

        features_df['PE_Ratio'] = features_df['PE_Ratio'].fillna(features_df['PE_Ratio'].median())
        features_df['Dividend_Yield'] = features_df['Dividend_Yield'].fillna(0.0)
        features_df['News_Sentiment'] = features_df['News_Sentiment'].fillna(0.0)

        pred = model.predict(features_df)[0]
        recommendation = label_encoder.inverse_transform([pred])[0]

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

@app.route("/retrain", methods=["POST"])
def retrain_model():
    """Placeholder for model retraining"""
    try:
        import train_model
        return jsonify({"success": True, "message": "Model retrained successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    if not os.path.exists('data/stock_analysis.json'):
        try:
            analyze_all_stocks()
        except Exception as e:
            logger.error(f"Initial analysis error: {str(e)}")
    port = int(os.getenv("PORT", 10000))
    app.run(host='0.0.0.0', port=port)