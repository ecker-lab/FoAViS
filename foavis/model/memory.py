"""Memory module (paper Sec. 3.3): an LSTM cell applied at every position of the 64 x 64 memory grid."""
import torch
import torch.nn as nn


class MemoryModule(nn.Module):
    """H_t = LSTM(H_{t-1}, F_t) per grid position, with F_t the backprojected backbone features.

    A convolutional LSTM cell; with kernel_size=1 (the paper models) the positions do not interact.
    """

    def __init__(self, input_channels, channels=128, kernel_size=1, grid_size=64):
        super().__init__()
        self.channels, self.grid_size = channels, grid_size
        self.conv = nn.Conv2d(input_channels + channels, 4 * channels, kernel_size, padding=kernel_size // 2,
                              bias=True)

    def init_state(self, batch_size, device):
        shape = (batch_size, self.channels, self.grid_size, self.grid_size)
        return (torch.zeros(shape, device=device, dtype=torch.float32).contiguous(),
                torch.zeros(shape, device=device, dtype=torch.float32).contiguous())

    def forward(self, features, state):
        """features (B, C_in, G, G), state (h, c) -> new state (h, c); h are the memory features."""
        h, c = state
        i, f, o, g = torch.chunk(self.conv(torch.cat([features, h], dim=1)), 4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c
