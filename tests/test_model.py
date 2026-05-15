import torch

from src.model import Conv1DAutoencoder


def test_autoencoder_preserves_input_shape():
    model = Conv1DAutoencoder(in_channels=3, filters=[16, 32, 64], kernel_size=5)
    batch = torch.randn(4, 3, 100)

    output = model(batch)

    assert output.shape == batch.shape
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)
