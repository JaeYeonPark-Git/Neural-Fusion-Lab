import matplotlib.pyplot as plt
import pandas as pd
import os
import operator
from typing import TypedDict, Annotated, List, Dict, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 내부 모듈 임포트 (기존 구조 유지)
from .tools import MarketDataManager
from .multimodal import VisionAnalyst
from .graph_rag import GraphRAGEngine

# ==========================================
# 1. State Definition (에이전트 공유 메모리)
# ==========================================
class AgentState(TypedDict):
    stock_symbol: str
    messages: Annotated[List[BaseMessage], operator.add]
    
    # 각 전문가 에이전트의 분석 결과 저장소
    chart_data: Dict[str, str]    # VLM 분석 결과
    quant_data: Dict[str, float]  # 수치적 지표 (RSI, Volatility 등)
    knowledge_data: str           # GraphRAG 리포트
    
    final_decision: str           # Supervisor의 최종 판단

# ==========================================
# 2. Chart Analyst (Vision + Technical)
# ==========================================
class ChartAgent:
    def __init__(self, config):
        self.data_manager = MarketDataManager()
        self.vision_analyst = VisionAnalyst(model_name=config['models']['vision'])
        self.chart_dir = config['paths']['chart_save_dir']
        os.makedirs(self.chart_dir, exist_ok=True)

    def _generate_expert_chart(self, symbol: str) -> str:
        """
        [Upgrade] 단순 주가가 아닌 Bollinger Bands, Volume, RSI를 포함한 멀티 플롯 차트 생성
        """
        df = self.data_manager.get_price_history(symbol)
        df = self.data_manager.add_technical_indicators(df) # RSI, SMA 등 계산 가정
        
        # 캔버스 설정 (3분할: 가격 / 거래량 / RSI)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 1. Price & Bollinger Bands
        ax1.plot(df.index, df['Close'], label='Price', color='black')
        ax1.plot(df.index, df['SMA_20'], label='SMA 20', color='blue', alpha=0.7)
        if 'Upper_Band' in df.columns:
            ax1.fill_between(df.index, df['Upper_Band'], df['Lower_Band'], color='gray', alpha=0.2, label='Bollinger Band')
        ax1.set_title(f"Technical Analysis: {symbol}")
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # 2. Volume
        colors = ['red' if r < 0 else 'green' for r in df['Close'].diff()]
        ax2.bar(df.index, df['Volume'], color=colors, alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # 3. RSI
        if 'RSI' in df.columns:
            ax3.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax3.axhline(70, linestyle='--', color='red', alpha=0.5)
            ax3.axhline(30, linestyle='--', color='green', alpha=0.5)
            ax3.set_ylabel('RSI')
            ax3.grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.chart_dir, f"{symbol}_expert_chart.png")
        plt.savefig(save_path)
        plt.close()
        return save_path

    def analyze(self, state: AgentState):
        symbol = state['stock_symbol']
        print(f"👁️ [ChartAgent] Visualizing market data for {symbol}...")
        
        # 차트 생성
        image_path = self._generate_expert_chart(symbol)
        
        # VLM 분석 요청 (Prompt Engineering 강화)
        context = "Focus on candle patterns, support/resistance levels, and divergence in RSI."
        analysis = self.vision_analyst.analyze_chart(image_path, context=context)
        
        return {
            "chart_data": {"path": image_path, "analysis": analysis},
            "messages": [HumanMessage(content=f"📊 Chart Analyst: \n{analysis}")]
        }

# ==========================================
# 3. Quant Analyst (Numerical & Statistical)
# ==========================================
class QuantAgent:
    """
    [New] 수치적 데이터를 바탕으로 통계적 리스크와 모멘텀을 계산하는 에이전트
    """
    def __init__(self, config):
        self.data_manager = MarketDataManager()
        
    def analyze(self, state: AgentState):
        symbol = state['stock_symbol']
        print(f"🧮 [QuantAgent] Calculating statistical metrics for {symbol}...")
        
        df = self.data_manager.get_price_history(symbol)
        
        # 간단한 퀀트 지표 계산 (실제로는 더 복잡한 로직 가능)
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5) # 연환산 변동성
        recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100 # 1달 수익률
        max_drawdown = (df['Close'] / df['Close'].cummax() - 1).min() * 100
        
        metrics = {
            "volatility_annual": f"{volatility:.2%}",
            "1m_return": f"{recent_return:.2f}%",
            "max_drawdown": f"{max_drawdown:.2f}%"
        }
        
        report = f"Volatility: {metrics['volatility_annual']}, MDD: {metrics['max_drawdown']}"
        
        return {
            "quant_data": metrics,
            "messages": [HumanMessage(content=f"🧮 Quant Analyst: \n{report}")]
        }

# ==========================================
# 4. Knowledge Analyst (GraphRAG)
# ==========================================
class KnowledgeAgent:
    """
    [New] Neo4j 지식 그래프를 탐색하여 공급망/지배구조 리스크를 파악하는 에이전트
    """
    def __init__(self, config):
        # 실제 연결이 없으면 Mock 모드로 동작하도록 처리 가능
        try:
            self.engine = GraphRAGEngine()
        except:
            self.engine = None

    def analyze(self, state: AgentState):
        symbol = state['stock_symbol']
        print(f"🕸️ [KnowledgeAgent] Querying Knowledge Graph for {symbol}...")
        
        if self.engine:
            # 실제 그래프 쿼리 (예: 공급망 리스크 탐색)
            context = self.engine.get_entity_context(symbol)
            insight = f"Analyzed supply chain connections for {symbol}. \nContext: {context[:200]}..."
        else:
            insight = "GraphDB connection not available. (Mocking: No major governance risks found.)"
            
        return {
            "knowledge_data": insight,
            "messages": [HumanMessage(content=f"🕸️ Knowledge Analyst: \n{insight}")]
        }

# ==========================================
# 5. Supervisor (Decision Maker)
# ==========================================
class SupervisorAgent:
    def __init__(self, config):
        self.llm = ChatOpenAI(model=config['models']['supervisor'], temperature=0)
        
    def summarize(self, state: AgentState):
        print("🕵️ [Supervisor] Synthesizing all reports...")
        
        # 이전 에이전트들의 결과를 종합
        prompt = f"""
        You are the Chief Investment Officer (CIO) of Neural Fusion Lab.
        Synthesize the following reports to make a final investment decision for '{state['stock_symbol']}'.
        
        1. [Visual Analysis]: {state.get('chart_data', {}).get('analysis')}
        2. [Quant Metrics]: {state.get('quant_data')}
        3. [Knowledge Graph]: {state.get('knowledge_data')}
        
        output format:
        - Decision: [BUY / SELL / HOLD]
        - Confidence Score: [1-10]
        - Key Rationale: (Summarize within 3 sentences)
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "final_decision": response.content,
            "messages": [HumanMessage(content=response.content)]
        }
