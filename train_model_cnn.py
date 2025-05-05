import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import yfinance as yf
import ta
from tenacity import retry, stop_after_attempt, wait_fixed
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import joblib

# ---------- CONSTANTS ----------
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Define models directory
BASE_DIR = os.path.abspath(os.path.join('C:', 'Users', 'prath', 'Downloads', 'stock_gcn_cnn'))
MODELS_DIR = 'models'  # Relative path since working directory is set
print(f"Base directory: {BASE_DIR}")
print(f"Models directory (relative): {MODELS_DIR}")
print(f"Expected models directory (absolute): {os.path.join(BASE_DIR, MODELS_DIR)}")

# Set working directory
os.chdir(BASE_DIR)
print(f"Current working directory: {os.getcwd()}")

# ---------- NEWS SENTIMENT FUNCTION ----------
def fetch_news_sentiment(ticker):
    try:
        url = f"https://news.google.com/search?q={ticker}%20stock"
        response = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.text for h in soup.find_all('h3', limit=5)]
        sentiment = TextBlob(" ".join(headlines)).sentiment.polarity
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

        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }).drop(columns=['Dividends', 'Stock Splits'], errors='ignore')

        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['Close']
        stoch = ta.momentum.StochasticOscillator(
            high=df['High'],
            low=df['Low'], 
            close=df['Close'], 
            window=14,
            smooth_window=3
        )
        df['Stochastic'] = stoch.stoch()
        
        df = df.reset_index()
        df['symbol'] = ticker
        df['News_Sentiment'] = fetch_news_sentiment(ticker)
        
        return df

    except Exception as e:
        print(f"❌ Failed to process {ticker}: {str(e)}")
        return None

# ---------- FEATURE PROCESSING ----------
def prepare_features(df):
    try:
        df['percent_change_5d'] = df.groupby('symbol')['Close'].transform(
            lambda x: x.pct_change(periods=5)
        ) * 100
        
        df['label'] = pd.cut(
            df['percent_change_5d'],
            bins=[-np.inf, -5, 5, np.inf],
            labels=['SELL', 'HOLD', 'BUY']
        )
        
        df['volume_score'] = df['Volume'].diff().gt(0).astype(int)
        df['volatility'] = df.groupby('symbol')['Close'].transform(
            lambda x: x.rolling(5).std()
        )
        
        feature_columns = [
            'RSI', 'MACD', 'SMA_50', 'BB_Width', 'Stochastic',
            'News_Sentiment', 'volume_score', 'percent_change_5d', 'volatility'
        ]
        
        df = df.dropna(subset=feature_columns + ['label'])
        return df[['symbol', 'Date', 'Close'] + feature_columns + ['label']]
    
    except Exception as e:
        print(f"⚠️ Feature processing failed: {str(e)}")
        return None

# ---------- MODEL ARCHITECTURE ----------
class SimplifiedModel(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.gcn1 = GCNConv(num_features, 32)
        self.gcn2 = GCNConv(32, num_classes)
        
    def forward(self, x, edge_index, edge_weight):
        print("Input shape:", x.shape)
        x = F.relu(self.gcn1(x, edge_index, edge_weight))
        print("After GCN1:", x.shape)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.gcn2(x, edge_index, edge_weight)
        print("After GCN2:", x.shape)
        return F.log_softmax(x, dim=1)

# ---------- DATA PREPARATION ----------
def prepare_graph_data(final_df, tickers):
    try:
        print("Starting prepare_graph_data...")
        feature_columns = [
            'RSI', 'MACD', 'SMA_50', 'BB_Width', 'Stochastic',
            'News_Sentiment', 'volume_score', 'percent_change_5d', 'volatility'
        ]
        
        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(final_df[feature_columns])
        os.makedirs(MODELS_DIR, exist_ok=True)
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Successfully saved scaler to {scaler_path}")
        
        # Encode labels
        le = LabelEncoder()
        labels = le.fit_transform(final_df['label'])
        label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
        joblib.dump(le, label_encoder_path)
        print(f"Successfully saved label encoder to {label_encoder_path}")
        
        # Build correlation graph
        returns = final_df.pivot(index='Date', columns='symbol', values='Close').pct_change().dropna()
        corr_matrix = returns.corr().fillna(0)
        edge_threshold = 0.7
        
        edge_index = []
        edge_weight = []
        for i, j in np.ndindex(corr_matrix.shape):
            if i != j and abs(corr_matrix.iloc[i, j]) > edge_threshold:
                edge_index.append([i, j])
                edge_weight.append(corr_matrix.iloc[i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.tensor([[0], [0]], dtype=torch.long)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float) if edge_weight else torch.tensor([0.0], dtype=torch.float)
        
        # Reshape features for temporal processing
        time_steps = final_df.groupby('symbol').size().min()
        num_stocks = len(tickers)
        features = features.reshape(num_stocks, time_steps, -1)
        labels = labels.reshape(num_stocks, time_steps)
        
        features = torch.tensor(features, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        
        print("Finished prepare_graph_data")
        return Data(x=features, edge_index=edge_index, edge_weight=edge_weight, y=labels)
    
    except Exception as e:
        print(f"⚠️ Graph data preparation failed: {str(e)}")
        return None

# ---------- TRAINING PIPELINE ----------
def train_model(data):
    try:
        print("Starting train_model...")
        # Split data by time steps
        train_size = int(0.8 * data.x.shape[1])
        train_data = Data(
            x=data.x[:, :train_size, :].contiguous(),
            edge_index=data.edge_index,
            edge_weight=data.edge_weight,
            y=data.y[:, :train_size].contiguous()
        )
        test_data = Data(
            x=data.x[:, train_size:, :].contiguous(),
            edge_index=data.edge_index,
            edge_weight=data.edge_weight,
            y=data.y[:, train_size:].contiguous()
        )
        
        # Initialize model
        model = SimplifiedModel(
            num_features=train_data.x.shape[2],
            num_classes=len(np.unique(data.y))
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        
        # Training loop
        best_acc = 0
        model_path = os.path.join(MODELS_DIR, 'stock_predictor.pkl')
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            
            # Flatten the temporal dimension
            batch_size, seq_len, num_features = train_data.x.size()
            x_flat = train_data.x.view(-1, num_features)
            y_flat = train_data.y.view(-1)
            
            out = model(x_flat, train_data.edge_index, train_data.edge_weight)
            loss = F.nll_loss(out, y_flat)
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                test_x_flat = test_data.x.view(-1, num_features)
                test_y_flat = test_data.y.view(-1)
                
                pred = model(test_x_flat, test_data.edge_index, test_data.edge_weight).argmax(dim=1)
                acc = (pred == test_y_flat).sum().item() / test_y_flat.size(0)
                scheduler.step(acc)
                
            if acc > best_acc:
                best_acc = acc
                os.makedirs(MODELS_DIR, exist_ok=True)
                joblib.dump(model, model_path)
                print(f"Successfully saved model to {model_path} with accuracy {acc:.4f}")
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}')
        
        # Force save the model at the end of training
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Successfully saved final model to {model_path}")
        
        # Final evaluation
        print(f"Loading model from {model_path} for final evaluation")
        model = joblib.load(model_path)
        print("Successfully loaded model for evaluation")
        
        model.eval()
        with torch.no_grad():
            test_x_flat = test_data.x.view(-1, num_features)
            test_y_flat = test_data.y.view(-1)
            label_encoder_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
            pred = model(test_x_flat, test_data.edge_index, test_data.edge_weight).argmax(dim=1)
            print(classification_report(test_y_flat.numpy(), pred.numpy(), 
                                      target_names=joblib.load(label_encoder_path).classes_))
    
    except Exception as e:
        print(f"⚠️ Training failed: {str(e)}")

# ---------- MAIN EXECUTION ----------
if __name__ == "__main__":
    print(f"Models directory: {MODELS_DIR}")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    dfs = []
    
    for ticker in tickers:
        print(f"⏳ Processing {ticker}...")
        try:
            raw_data = fetch_stock_data(ticker)
            if raw_data is not None:
                processed = prepare_features(raw_data)
                if processed is not None and not processed.empty:
                    dfs.append(processed)
                    print(f"✅ Successfully processed {ticker}")
                else:
                    print(f"⛔ Empty/invalid data for {ticker}")
            else:
                print(f"⛔ Failed to fetch {ticker}")
        except Exception as e:
            print(f"🔥 Critical error processing {ticker}: {str(e)}")
    
    if not dfs:
        raise ValueError("🚫 No valid data available for training")
    
    final_df = pd.concat(dfs).sort_values(['Date', 'symbol'])
    print(f"📊 Final dataset shape: {final_df.shape}")
    
    # Prepare graph data and train model
    graph_data = prepare_graph_data(final_df, tickers)
    if graph_data is not None:
        train_model(graph_data)
    else:
        raise ValueError("🚫 Failed to prepare graph data for training")