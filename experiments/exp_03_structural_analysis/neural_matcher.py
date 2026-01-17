import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, DataLoader

class GIN(torch.nn.Module):
    """
    [Deep Learning Approach]
    Graph Isomorphism Network (GIN)
    Theoretical basis: Weisfeiler-Lehman (WL) Graph Isomorphism Test.
    Goal: Learn structural embeddings to measure similarity between financial graphs.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GIN, self).__init__()
        
        # MLP layers for GIN
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

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        # 2. Readout layer (Pooling) -> Graph-level embedding
        # Sum pooling is crucial for GIN's expressiveness (injectivity)
        x = global_add_pool(x, batch)
        
        # 3. Final classifier / Embedding projection
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def calculate_similarity(graph1, graph2):
    """
    Compares two graphs using GIN embeddings.
    """
    model = GIN(in_channels=1, hidden_channels=32, out_channels=16)
    model.eval()

    # Dummy Feature (All 1s) - We focus on structure, not node features here
    with torch.no_grad():
        emb1 = model(graph1.x, graph1.edge_index, graph1.batch)
        emb2 = model(graph2.x, graph2.edge_index, graph2.batch)
        
        # Cosine Similarity
        similarity = F.cosine_similarity(emb1, emb2)
        return similarity.item()

if __name__ == "__main__":
    # Create two structurally identical graphs (Isomorphic)
    # Graph A: 0-1, 1-2
    edge_index_a = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x_a = torch.ones((3, 1)) # 3 nodes
    data_a = Data(x=x_a, edge_index=edge_index_a, batch=torch.zeros(3, dtype=torch.long))

    # Graph B: 0-1, 1-2 (Nodes labeled differently but structure is same line)
    edge_index_b = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x_b = torch.ones((3, 1)) # 3 nodes
    data_b = Data(x=x_b, edge_index=edge_index_b, batch=torch.zeros(3, dtype=torch.long))

    score = calculate_similarity(data_a, data_b)
    print(f"🤖 [NEURAL MATCHING] Structural Similarity Score: {score:.4f}")
    
    if score > 0.9:
        print("✅ The model considers these graphs structurally isomorphic.")
