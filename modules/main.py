from langgraph.graph import StateGraph, END
from modules.agents import ChartAgent, KnowledgeAgent, SupervisorAgent, AgentState

def build_graph():
    # 컴포넌트 초기화
    chart_agent = ChartAgent()
    knowledge_agent = KnowledgeAgent()
    supervisor = SupervisorAgent()

    # 그래프 정의
    workflow = StateGraph(AgentState)

    # 노드 추가
    workflow.add_node("chart_reader", chart_agent.analyze)
    workflow.add_node("knowledge_miner", knowledge_agent.analyze)
    workflow.add_node("decision_maker", supervisor.decide)

    # 엣지 연결 (병렬 처리 가능)
    workflow.set_entry_point("chart_reader")
    workflow.add_edge("chart_reader", "knowledge_miner") # 순차 실행 예시
    workflow.add_edge("knowledge_miner", "decision_maker")
    workflow.add_edge("decision_maker", END)

    return workflow.compile()

if __name__ == "__main__":
    app = build_graph()
    initial_state = {"stock_symbol": "NVDA", "messages": []}
    result = app.invoke(initial_state)
    print(f"Final Decision: {result['final_decision']}")
