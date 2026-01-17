import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx

# GIN 모델 정의
class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN)
    Theory: Can distinguish graph structures as powerful as the WL-test.
    Usage: Extract structural embeddings for similarity comparison.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        
        # MLP layers for GIN aggregation
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        )
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
        )
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        # batch가 None일 경우 (단일 그래프) 처리
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)

        # 1. Message Passing
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # 2. Readout (Graph-level Embedding) using Sum Pooling
        # Sum pooling is theoretically better for isomorphism than mean/max
        x = global_add_pool(x, batch)
        
        # 3. Projection
        x = self.lin(x)
        return x

class NeuralGraphMatcher:
    def __init__(self):
        # Feature dimension=1 (Structural only), Hidden=32, Output=16
        self.model = GIN(1, 32, 16)
        self.model.eval() # Inference mode

    def nx_to_pyg(self, G):
        """NetworkX 그래프를 PyTorch Geometric 데이터로 변환"""
        # 노드 피처가 없으므로 모든 노드에 상수 1 부여 (구조만 보겠다는 의미)
        for i in G.nodes():
            G.nodes[i]['x'] = [1.0]
            
        data = from_networkx(G)
        # PyG의 x(feature) 텐서 확인 및 차원 맞춤
        if data.x is None:
             data.x = torch.ones((G.number_of_nodes(), 1))
        else:
             data.x = data.x.view(-1, 1).float()
             
        return data

    def calculate_similarity(self, G1, G2):
        """
        두 그래프의 구조적 유사도(Cosine Similarity) 계산
        """
        data1 = self.nx_to_pyg(G1)
        data2 = self.nx_to_pyg(G2)
        
        with torch.no_grad():
            emb1 = self.model(data1.x, data1.edge_index)
            emb2 = self.model(data2.x, data2.edge_index)
            
            similarity = F.cosine_similarity(emb1, emb2)
            return similarity.item()

if __name__ == "__main__":
    matcher = NeuralGraphMatcher()
    
    # Case 1: 완벽하게 동일한 구조 (Isomorphic)
    G_pattern = nx.cycle_graph(5) # 5각형 고리
    G_suspect = nx.cycle_graph(5) # 5각형 고리
    
    score_iso = matcher.calculate_similarity(G_pattern, G_suspect)
    print(f"🤖 Similarity (Isomorphic): {score_iso:.4f}") # 1.0에 가까워야 함

    # Case 2: 약간 다른 구조 (Non-Isomorphic but similar)
    G_noise = nx.cycle_graph(5)
    G_noise.add_edge(0, 2) # 엣지 하나 추가 (노이즈)
    
    score_noise = matcher.calculate_similarity(G_pattern, G_noise)
    print(f"🤖 Similarity (Noisy): {score_noise:.4f}") # 1.0보다 낮아야 함
    
    # Case 3: 완전히 다른 구조
    G_diff = nx.star_graph(4) # 별 모양
    
    score_diff = matcher.calculate_similarity(G_pattern, G_diff)
    print(f"🤖 Similarity (Different): {score_diff:.4f}") # 훨씬 낮아야 함
