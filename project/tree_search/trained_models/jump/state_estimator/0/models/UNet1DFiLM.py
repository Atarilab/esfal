import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, num_features, conditioning_dim):
        super(FiLM, self).__init__()
        self.hidden_dim = num_features * 3
        self.fc1 = nn.Linear(conditioning_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, num_features * 2)
    
    def forward(self, x, conditioning):
        h = F.leaky_relu(self.fc1(conditioning))
        gamma_beta = self.fc2(h)
        gamma, beta = torch.split(gamma_beta, gamma_beta.size(1) // 2, dim=1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return gamma * x + beta

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, conditioning_dim, num_features=64):
        super(UNet1D, self).__init__()

        self.encoder1 = nn.Conv1d(in_channels, num_features, kernel_size=3, padding=1, dilation=1)
        self.encoder2 = nn.Conv1d(num_features, num_features * 2, kernel_size=3, padding=2, dilation=2)
        self.encoder3 = nn.Conv1d(num_features * 2, num_features * 4, kernel_size=3, padding=4, dilation=4)
        
        self.conditioning_dim = conditioning_dim
        self.fc1 = nn.Linear(conditioning_dim, num_features)
        self.fc2 = nn.Linear(num_features, num_features)
        self.film1 = FiLM(num_features, num_features)
        self.film2 = FiLM(num_features * 2, num_features)
        self.film3 = FiLM(num_features * 4, num_features)

        self.decoder1 = nn.ConvTranspose1d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.ConvTranspose1d(num_features * 4, num_features, kernel_size=2, stride=2)
        self.decoder3 = nn.Conv1d(num_features * 2, num_features, kernel_size=3, padding=1)
        self.decoder4 = nn.Conv1d(num_features, out_channels, kernel_size=5, padding=2)

        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x_cond):
        x, conditioning = torch.split(x_cond, x_cond.shape[-1] - self.conditioning_dim, dim=-1)
        conditioning = self.fc2(self.fc1(conditioning.squeeze()))
        
        # Encoder
        e1 = self.film1(nn.functional.gelu(self.encoder1(x)), conditioning)
        e2 = self.film2(nn.functional.gelu(self.encoder2(self.pool(e1))), conditioning)
        e3 = self.film3(nn.functional.gelu(self.encoder3(self.pool(e2))), conditioning)
        
        # Decoder with concatenation
        d1 = torch.cat((self.decoder1(e3), e2), dim=1)
        d1 = nn.functional.gelu(d1)
        d2 = torch.cat((self.decoder2(d1), e1), dim=1)
        d2 = nn.functional.gelu(d2)
        d3 = self.decoder3(d2)
        d4 = self.decoder4(d3)
        
        return d4

# Example usage
if __name__ == "__main__":
    # Create a random tensor of shape (batch_size, in_channels, length)
    x = torch.randn(8, 1, 48 + 10)

    # Instantiate and run the model
    model = UNet1D(in_channels=1, out_channels=1, conditioning_dim=10, num_features=64)
    out = model(x)
    print(out.shape)  # Expected output shape: (batch_size, out_channels, length)
