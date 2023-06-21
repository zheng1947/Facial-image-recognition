import torch
import torch.nn as nn

class gMLPBlock(nn.Module):
    def __init__(self, in_features, mlp_features, dropout_rate=0.0):
        super(gMLPBlock, self).__init__()

        self.mlp_fc1 = nn.Linear(in_features, mlp_features)
        self.mlp_fc2 = nn.Linear(mlp_features, in_features)
        self.gate_fc1 = nn.Linear(in_features, mlp_features)
        self.gate_fc2 = nn.Linear(mlp_features, in_features)
        self.layer_norm1 = nn.LayerNorm(in_features)
        self.layer_norm2 = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x):
        # Compute input pre-activation
        input_norm = self.layer_norm1(x)
        mlp_activations = self.mlp_fc1(input_norm)
        mlp_activations = self.activation(mlp_activations)
        mlp_activations = self.dropout(mlp_activations)
        mlp_activations = self.mlp_fc2(mlp_activations)
        # Compute gate activations
        gate_activations = self.gate_fc1(input_norm)
        gate_activations = self.activation(gate_activations)
        gate_activations = self.dropout(gate_activations)
        gate_activations = self.gate_fc2(gate_activations)
        # Apply gate
        gated_activations = input_norm * torch.sigmoid(gate_activations)
        # Apply activation
        hidden_activations = self.layer_norm2(gated_activations + mlp_activations)
        return hidden_activations

class gMLP(nn.Module):
    def __init__(self, in_features, out_features, mlp_features, num_blocks, dropout_rate=0.0):
        super(gMLP, self).__init__()
        self.input_fc = nn.Linear(in_features, mlp_features)
        self.output_fc = nn.Linear(mlp_features, out_features)
        self.blocks = nn.ModuleList([gMLPBlock(mlp_features, mlp_features, dropout_rate) for i in range(num_blocks)])
        self.activation = nn.GELU()

    def forward(self, x):
        # Compute input pre-activation
        input_activations = self.input_fc(x)
        # Apply activation
        hidden_activations = self.activation(input_activations)
        # Pass through blocks
        for block in self.blocks:
            hidden_activations = block(hidden_activations)
        # Compute output
        output_activations = self.output_fc(hidden_activations)
        return output_activations
