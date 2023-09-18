import torch
from torch import nn


class TransformerAutoencoder(nn.Transformer):
    """
    Adaptation of torch.nn.Transformer class. The sequence to encode is feed in both the sources (src) and the targets (tgt).

    Args:
        parameters (dict): parsed from the config.yaml file that should contain the following fields:
            n_features: number of independent variables of each sequence point.
            n_heads: number of heads of the Multi-Head Attention modules.
            n_encoder_layers: number of layers on the encoder. 1 layer = Multi-Head Attention + Feed Forward.
            n_decoder_layers: number of layers on the decoder. 1 layer = Multi-Head Attention + Feed Forward.
    """
    def __init__(
        self,
        parameters: dict
    ):
        super().__init__(
            d_model=parameters["n_features"],
            nhead=parameters["n_heads"],
            batch_first=True,
            num_encoder_layers=parameters["n_encoder_layers"],
            num_decoder_layers=parameters["n_decoder_layers"]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch_size, seq_len, n_features)
        return super().forward(src=x, tgt=x)