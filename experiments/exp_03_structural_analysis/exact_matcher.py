import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import isomorphism

class PatternDetector:
    """
    [Classical Algorithm Approach]
    Uses VF2 algorithm to find exact subgraph isomorphism/homomorphism.
    Target: Detecting Circular Trading (Money Laundering) patterns.
    """
    def __init__(self):
        self.market_graph = nx.DiGraph() # Directed Graph for transaction flow

    def generate_dummy_market_data(self):
        """Creates a synthetic transaction network with some random noise."""
        # Normal transactions
        edges = [
            (1, 2), (2, 3), (3, 4), (4, 5),
            (10, 11), (11, 12), (5, 9)
        ]
        self.market_graph.add_edges_from(edges)
        print(f"✅ Market Graph Generated: {self.market_graph.number_of_nodes()} nodes.")

    def inject_fraud_pattern(self):
        """Injects a specific 'Circular Trading' pattern (A->B->C->A)."""
        # Fraud ring: Node 100 -> 101 -> 102 -> 100
        fraud_edges = [(100, 101), (101, 102), (102, 100)]
        self.market_graph.add_edges_from(fraud_edges)
        print("⚠️ Injected Fraud Pattern: 100 -> 101 -> 102 -> 100")

    def detect_pattern(self, pattern_type="circular"):
        """
        Detects if the pattern graph exists within the market graph.
        Mapping: Subgraph Isomorphism
        """
        # Define the pattern to search for
        pattern_graph = nx.DiGraph()
        if pattern_type == "circular":
            # Finding a 3-cycle (Triangle)
            pattern_graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        
        # Use VF2 Algorithm for Subgraph Isomorphism
        # G1: Market (Large), G2: Pattern (Small)
        matcher = isomorphism.DiGraphMatcher(self.market_graph, pattern_graph)
        
        results = list(matcher.subgraph_isomorphisms_iter())
        
        if results:
            print(f"\n🚨 [MATHEMATICAL DETECTION] Found {len(results)} isomorphic subgraphs!")
            for idx, mapping in enumerate(results):
                print(f"   Match #{idx+1} (Market Node -> Pattern Node): {mapping}")
                # 역매핑을 통해 실제 마켓의 노드들을 추출 가능
        else:
            print("\n✅ No fraud patterns detected.")

if __name__ == "__main__":
    detector = PatternDetector()
    detector.generate_dummy_market_data()
    detector.inject_fraud_pattern()
    detector.detect_pattern()
