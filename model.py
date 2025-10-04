import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, p_dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.linear_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.gelu(self.ln1(self.linear1(x)))
        x = self.dropout(x)
        x = F.gelu(self.ln2(self.linear2(x)))
        x = self.dropout(x)
        return self.linear_out(x)
