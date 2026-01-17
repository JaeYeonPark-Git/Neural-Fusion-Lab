import networkx as nx
import matplotlib.pyplot as plt
import random

def generate_financial_network(num_nodes=50, num_edges=100):
    """
    랜덤한 금융 거래 네트워크 생성 (Erdos-Renyi 변형)
    """
    G = nx.DiGraph() # 방향 그래프 (송금: A -> B)
    
    # 노드 추가 (계좌)
    G.add_nodes_from(range(num_nodes))
    
    # 엣지 추가 (거래)
    for _ in range(num_edges):
        u, v = random.sample(range(num_nodes), 2)
        amount = random.randint(1000, 1000000)
        G.add_edge(u, v, amount=amount, type="transfer")
        
    return G

def visualize_graph(G, title="Financial Graph", highlight_nodes=None):
    """
    그래프 시각화 (하이라이트 기능 포함)
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # 기본 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    # 발견된 패턴 하이라이트 (빨간색)
    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color='red', node_size=500)
        
    plt.title(title)
    plt.axis('off')
    plt.show()
