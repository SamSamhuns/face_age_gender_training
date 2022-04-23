import torch
import torch.nn as nn
from base import BaseModel, ViewBlock


class FullyConnectedNet(BaseModel):
    """
    Classic fully connected neural network that downsamples features by half every layer
    """

    def __init__(self, num_blocks: int, num_classes: int,
                 num_initial_feats: int, reduce_in_feats: int,
                 dropout: float, **kwargs):
        super().__init__()

        blocks = []
        # input feature space reduction layer, acts as encoder layer
        # if reduce_feat_num is not None, reduce input features with downsampling instead of residual block
        if reduce_in_feats is not None:
            blocks.append(nn.Linear(num_initial_feats, reduce_in_feats))
            blocks.append(nn.Dropout(p=dropout))
            blocks.append(nn.ReLU())
        else:
            reduce_in_feats = num_initial_feats

        for i in range(num_blocks):
            blocks.extend(self._create_block(reduce_in_feats, dropout))
            reduce_in_feats //= 2

        # last classification layer
        blocks.append(nn.Linear(reduce_in_feats, num_classes))
        self.blocks = nn.Sequential(*blocks)

    def _create_block(self, in_feats: int, dropout: float) -> list:
        block = []
        block.append(nn.Linear(in_feats, in_feats // 2))
        block.append(nn.Dropout(p=dropout))
        block.append(nn.ReLU())
        return block

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.blocks(x)


class FullyConnectedNetBottleneck(BaseModel):
    """
    Classic fully connected neural network that downsamples features by half every layer

    Initial bottleneck layer needs a vector of rank 3
    Note: input must be a vector of rank 3 with shape [batch, num_frames, num_initial_feats]
    """

    def __init__(
            self, num_blocks: int, num_classes: int, num_frames: int,
            num_initial_feats: int, reduce_in_feats: int, dropout: float, **kwargs):
        super().__init__()

        blocks = []
        # input feature space reduction layer, acts as encoder layer
        # if reduce_feat_num is not None, reduce input features with downsampling instead of residual block
        if reduce_in_feats is not None:
            blocks.append(nn.Linear(num_initial_feats, reduce_in_feats))
            blocks.append(nn.Dropout(p=dropout))
            blocks.append(nn.ReLU())
        else:
            reduce_in_feats = num_initial_feats

        blocks.append(ViewBlock([num_frames * reduce_in_feats]))
        reduce_in_feats = num_frames * reduce_in_feats

        for i in range(num_blocks):
            blocks.extend(self._create_block(reduce_in_feats, dropout))
            reduce_in_feats //= 2

        # last classification layer
        blocks.append(nn.Linear(reduce_in_feats, num_classes))
        self.blocks = nn.Sequential(*blocks)

    def _create_block(self, in_feats: int, dropout: float) -> list:
        block = []
        block.append(nn.Linear(in_feats, in_feats // 2))
        block.append(nn.Dropout(p=dropout))
        block.append(nn.ReLU())
        return block

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.blocks(x)
