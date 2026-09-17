import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import TransformerConv
from typing import List, Optional, Tuple, Any


class Learn_Layer(nn.Module):
    def __init__(self, num_edge_type: int, feature_size: int, hid_size: int,
                 layer_num_heads: int, dropout_rate: float):
        super().__init__()

        self.gat_layers = nn.ModuleList([
            TransformerConv(in_channels=feature_size, out_channels=feature_size, heads=layer_num_heads,
                            dropout=dropout_rate, concat=False)
            for _ in range(num_edge_type)
        ])

        self.gated = nn.Sequential(
            nn.Linear(feature_size + feature_size, feature_size),
            nn.Sigmoid()
        )

        self.linear = nn.Linear(feature_size * num_edge_type, hid_size)
        torch.nn.init.kaiming_normal_(self.linear.weight, nonlinearity='leaky_relu')
        self.Dropout = nn.Dropout(dropout_rate)
        self.Tanh = nn.Tanh()

    def forward(self, learn_features: torch.Tensor, edge_indices: List[torch.Tensor],
                agg: Optional[str] = 'min') -> Tuple[Any, List[Tensor]]:

        transformed_features = [
            self.gat_layers[i](learn_features, edge_indices[i][0].T).flatten(1)
            for i in range(len(edge_indices))
        ]

        gated_weights = [self.gated(torch.cat((transformed, learn_features), dim=1))
                         for transformed in transformed_features]

        gated_features = [
            torch.mul(torch.tanh(transformed), gated) + torch.mul(learn_features, (1 - gated))
            for transformed, gated in zip(transformed_features, gated_weights)
        ]

        fusion_embeddings = gated_features[0]
        for i in range(1, len(gated_features)):
            fusion_embeddings = torch.cat((fusion_embeddings, gated_features[i]), dim=1)
        fusion_features = self.Dropout(self.Tanh(self.linear(fusion_embeddings)))

        return fusion_features, gated_features