import torch
import torch.nn as nn

class Conv1DAutoencoder(nn.Module):
    """Autoencoder basado en el notebook"""
    def __init__(self, in_channels=3, filters=[16, 32, 64], kernel_size=5):
        super(Conv1DAutoencoder, self).__init__()
        
        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, filters[0], kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(filters[0], filters[1], kernel_size=kernel_size, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(filters[1], filters[2], kernel_size=kernel_size, stride=2, padding=2),
            nn.ReLU()
        )
        
        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(filters[2], filters[1], kernel_size=kernel_size, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(filters[1], filters[0], kernel_size=kernel_size, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
        )
        
        self.final_conv = nn.Sequential(
             nn.ConvTranspose1d(filters[0], in_channels, kernel_size=7, stride=2, padding=3, output_padding=1),
             nn.Sigmoid() # Salida [0, 1] asumiendo normalización MinMax
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = self.final_conv(decoded)
        
        
        if decoded.size(2) != x.size(2):
             decoded = torch.nn.functional.interpolate(decoded, size=x.size(2))
             
        return decoded
