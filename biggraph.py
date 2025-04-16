#!/usr/bin/env python
"""
Toy PyTorch script with a decently sized operation graph.
This script builds a custom model with several branches and layers,
then performs a forward pass on a random input tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.compiler.reset()


class Branch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Branch, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # A simple branch with two linear layers and non-linearity.
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class ToyNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_branches=3, branch_output_dim=32):
        super(ToyNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        # Create multiple branches
        self.branches = nn.ModuleList([
            Branch(hidden_dim, hidden_dim, branch_output_dim) for _ in range(num_branches)
        ])
        self.fuse = nn.Linear(branch_output_dim * num_branches, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 10)  # final output layer (e.g., 10 classes)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        
        # Each branch computes its output, we then fuse them together.
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Concatenate along the feature dimension:
        fused = torch.cat(branch_outputs, dim=1)
        fused = F.relu(self.fuse(fused))
        
        # A couple more operations: apply dropout and a couple of linear transforms
        fused = F.dropout(fused, p=0.5, training=self.training)
        inter = F.relu(self.output_layer(fused))
        
        # Add a redundant operation for depth: element-wise multiplication and log-softmax
        inter = inter * torch.sigmoid(inter)
        output = F.log_softmax(inter, dim=1)
        
        return output


def main():
    torch.manual_seed(42)

    # Create a model instance
    model = ToyNet(input_dim=128, hidden_dim=64, num_branches=3, branch_output_dim=32)
    model.train()  # set to training mode

    # Create a random input tensor of shape (batch_size, input_dim)
    batch_size = 16
    input_tensor = torch.randn(batch_size, 128)

    # Run a forward pass
    forward = torch.compile(model.__call__)
    output = forward(input_tensor)
    print("Output shape:", output.shape)
    print("Output:", output)


if __name__ == "__main__":
    main()






