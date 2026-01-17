import yfinance as yf
import pandas as pd
from typing import Dict, Any

class MarketDataManager:
    """
    yfinance를 래핑하여 시계열 데이터와 기본 기술적 지표를 제공하는 클래스
    """
    def __init__(self):
        pass

    def get_price_history(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        return df

    def get_financial_summary(self, symbol: str) -> Dict[str, Any]:
        """주요 재무 정보 요약"""
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "sector": info.get("sector"),
            "current_price": info.get("currentPrice")
        }

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """간단한 기술적 지표 추가 (TA-Lib 대체 가능)"""
        df = df.copy()
        # SMA 20 (이동평균선)
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        # RSI 계산 로직 (간소화)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df
