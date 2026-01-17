from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# 1. State 정의: 에이전트 간 공유할 메모리
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    stock_symbol: str
    chart_analysis: str
    news_sentiment: str
    quant_metrics: dict
    final_decision: str

# 2. Vision Agent: 차트 이미지를 보고 패턴 분석 (Multimodal)
class ChartAgent:
    def __init__(self, model_name="gpt-4o"): # or Gemini Pro Vision
        self.llm = ChatOpenAI(model=model_name)
    
    def analyze(self, state: AgentState):
        symbol = state['stock_symbol']
        # 실제로는 여기서 이미지 경로를 로드하거나 캡처하는 로직 필요
        # [Image of candlestick chart for {symbol}]
        chart_image_path = f"data/{symbol}_daily_chart.png" 
        
        prompt = "이 주식 차트의 추세(Trend)와 지지/저항선을 분석해줘."
        # 멀티모달 입력 처리 로직 (생략)
        response = "상승 쐐기형 패턴이 관찰되며 20일 이평선 지지 중."
        
        return {"chart_analysis": response, "messages": [HumanMessage(content=f"Chart Analysis: {response}")]}

# 3. Graph RAG Agent: 뉴스와 지식그래프 분석
class KnowledgeAgent:
    def __init__(self):
        # Neo4j Connection 초기화
        pass
        
    def analyze(self, state: AgentState):
        symbol = state['stock_symbol']
        # Neo4j Cypher Query 실행 -> Supply Chain 리스크 등 파악
        insight = "공급망 내 주요 벤더의 파산 리스크가 감지됨."
        return {"news_sentiment": "Negative", "messages": [HumanMessage(content=f"Knowledge Insight: {insight}")]}

# 4. Supervisor (Decision Maker)
class SupervisorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4-turbo")
        
    def decide(self, state: AgentState):
        # 모든 분석 결과를 종합하여 최종 투자의견 도출
        prompt = f"""
        Chart: {state.get('chart_analysis')}
        News: {state.get('news_sentiment')}
        Quant: {state.get('quant_metrics')}
        
        위 정보를 바탕으로 Buy/Sell/Hold 중 하나를 결정하고 이유를 설명해.
        """
        decision = self.llm.invoke(prompt).content
        return {"final_decision": decision}
