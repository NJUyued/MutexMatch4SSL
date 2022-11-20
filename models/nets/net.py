import torch.nn as nn

class ReverseCLS(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ReverseCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out



