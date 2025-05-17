import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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