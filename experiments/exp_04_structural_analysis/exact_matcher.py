import networkx as nx
from networkx.algorithms import isomorphism
from .utils import generate_financial_network, visualize_graph

class FraudPatternMatcher:
    """
    [Mathematical Approach]
    Uses VF2 algorithm for Subgraph Isomorphism to detect exact fraud patterns.
    """
    def __init__(self):
        self.market_graph = generate_financial_network(num_nodes=30, num_edges=60)

    def inject_circular_fraud(self, nodes):
        """
        강제로 자전 거래(Circular Trading) 패턴 주입
        Ex) A -> B -> C -> A (돈세탁 의심 거래)
        """
        print(f"⚠️ Injecting Fraud Ring: {nodes}")
        edges = []
        for i in range(len(nodes)):
            u = nodes[i]
            v = nodes[(i + 1) % len(nodes)]
            edges.append((u, v))
            
        self.market_graph.add_edges_from(edges, type="fraud")

    def find_fraud_patterns(self):
        """
        정의된 패턴과 'Isomorphic(동형)'인 부분 그래프를 시장 전체에서 탐색
        """
        # 1. 찾고자 하는 패턴 정의 (삼각 순환 거래)
        pattern = nx.DiGraph()
        pattern.add_edges_from([(0, 1), (1, 2), (2, 0)])
        
        print("\n🔍 Searching for Circular Trading Patterns (Triangle)...")
        
        # 2. VF2 알고리즘 사용 (Subgraph Isomorphism)
        # DiGraphMatcher(큰_그래프, 찾는_패턴)
        matcher = isomorphism.DiGraphMatcher(self.market_graph, pattern)
        
        matches = list(matcher.subgraph_isomorphisms_iter())
        
        unique_suspects = set()
        if matches:
            print(f"🚨 FOUND {len(matches)} suspicious patterns!")
            for i, match in enumerate(matches):
                # match는 {패턴노드: 실제노드} 딕셔너리 반환
                real_nodes = list(match.keys()) # NetworkX 버전에 따라 keys/values 확인 필요
                # 매핑: {GraphNode: PatternNode} 형태임
                
                detected_nodes = list(match.keys())
                print(f"   Match #{i+1}: {detected_nodes}")
                unique_suspects.update(detected_nodes)
                
            # 시각화
            visualize_graph(self.market_graph, title="Detected Fraud Patterns (Exact Match)", highlight_nodes=list(unique_suspects))
        else:
            print("✅ No exact fraud patterns found.")

if __name__ == "__main__":
    matcher = FraudPatternMatcher()
    
    # 3개 노드로 구성된 사기 고리 주입
    matcher.inject_circular_fraud([5, 10, 15]) 
    
    matcher.find_fraud_patterns()
