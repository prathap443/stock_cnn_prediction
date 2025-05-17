# train_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
import ta
from tenacity import retry, stop_after_attempt, wait_fixed
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import os
import joblib

# ---------- CONSTANTS ----------
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- NEWS SENTIMENT FUNCTION ----------
def fetch_news_sentiment(ticker):
    try:
        url = f"https://news.google.com/search?q={ticker}%20stock"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3', limit=5)]
        sentiment = TextBlob(" ".join(headlines)).sentiment.polarity  # Fixed typo: headelines -> headlines
        return round(sentiment, 2)
    except Exception as e:
        print(f"⚠️ News error for {ticker}: {str(e)}")
        return 0.0

# ---------- FETCH STOCK DATA ----------
@retry(stop=stop_after_attempt(5), wait=wait_fixed(2))
def fetch_stock_data(ticker):
    try:
        df = yf.Ticker(ticker).history(period='120d', interval='1d')
        if df.empty:
            print(f"⚠️ Empty data for {ticker}")
            return None

        # Feature engineering
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['Close']
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'], low=df['Low'], close=df['Close'], 
            window=14, smooth_window=3
        )
        df['Stochastic'] = stoch.stoch()
        df['News_Sentiment'] = fetch_news_sentiment(ticker)
        
        return df.reset_index()

    except Exception as e:
        print(f"❌ Failed to process {ticker}: {str(e)}")
        return None

# ---------- FEATURE PROCESSING ----------
def prepare_features(df):
    try:
        # Create labels based on 5-day percentage change
        df['percent_change_5d'] = df['Close'].pct_change(5).fillna(0) * 100
        df['label'] = pd.cut(
            df['percent_change_5d'],
            bins=[-np.inf, -5, 5, np.inf],
            labels=['SELL', 'HOLD', 'BUY']
        )
        
        # Additional features
        df['volume_score'] = df['Volume'].diff().gt(0).astype(int)
        df['volatility'] = df['Close'].rolling(5).std().fillna(0)
        
        feature_columns = [
            'RSI', 'MACD', 'SMA_50', 'BB_Width', 'Stochastic',
            'News_Sentiment', 'volume_score', 'percent_change_5d', 'volatility'
        ]
        
        return df.dropna(subset=feature_columns + ['label'])[['Date', 'Close'] + feature_columns + ['label']]
    
    except Exception as e:
        print(f"⚠️ Feature processing failed: {str(e)}")
        return None

# ---------- NEURAL NETWORK ARCHITECTURE ----------
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

# ---------- DATA PREPARATION ----------
def prepare_training_data(final_df):
    try:
        print("Preparing training data...")
        feature_columns = [
            'RSI', 'MACD', 'SMA_50', 'BB_Width', 'Stochastic',
            'News_Sentiment', 'volume_score', 'percent_change_5d', 'volatility'
        ]
        
        # Feature scaling
        scaler = StandardScaler()
        features = scaler.fit_transform(final_df[feature_columns])
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
        
        # Label encoding
        le = LabelEncoder()
        labels = le.fit_transform(final_df['label'])
        joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        
        return features, labels
    
    except Exception as e:
        print(f"Data preparation failed: {str(e)}")
        return None, None

# ---------- TRAINING FUNCTION ----------
def train_model(features, labels):
    try:
        print("\nStarting model training...")
        # Train/test split
        split = int(0.8 * len(features))
        X_train, X_test = features[:split], features[split:]
        y_train, y_test = labels[:split], labels[split:]
        
        # Model configuration
        model = StockPredictor(
            input_size=X_train.shape[1],
            num_classes=len(np.unique(labels))
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.NLLLoss()
        
        # Training loop
        best_acc = 0
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            # Validation
            with torch.no_grad():
                model.eval()
                test_outputs = model(X_test)
                _, preds = torch.max(test_outputs, 1)
                acc = (preds == y_test).float().mean()
                
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'stock_predictor.pth'))
            
            print(f"Epoch {epoch+1}/100 | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")
        
        # Final evaluation
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'stock_predictor.pth')))
        with torch.no_grad():
            model.eval()
            outputs = model(X_test)
            _, preds = torch.max(outputs, 1)
            print("\nClassification Report:")
            print(classification_report(y_test.numpy(), preds.numpy(), 
                                      target_names=joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl')).classes_))
            
    except Exception as e:
        print(f"Training failed: {str(e)}")

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    all_data = []
    
    # Data collection
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        data = fetch_stock_data(ticker)
        if data is not None:
            processed = prepare_features(data)
            if processed is not None:
                all_data.append(processed)
                print(f"✅ Collected {len(processed)} samples")
                
    if not all_data:
        raise ValueError("No data available for training")
    
    # Combine and prepare data
    final_df = pd.concat(all_data).sort_values('Date').dropna()
    print(f"\nFinal dataset shape: {final_df.shape}")
    
    # Train model
    features, labels = prepare_training_data(final_df)
    if features is not None and labels is not None:
        train_model(features, labels)
    else:
        print("Failed to prepare training data")