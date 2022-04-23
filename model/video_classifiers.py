import torch
import torchvision
import torch.nn as nn
from base import BaseModel, ViewBlock


class TimeDistributedImage(nn.Module):
    """
    Applies a modules to each tiemslice of an image tensor array.
    i.e. x = torch.ones([5, 10, 3, 128, 128])
         conv2d = nn.Conv2d(3, 64, (3, 3))
         y = TimeDistributedImage(conv2d)(x)
         print(y.shape)  # (5, 10, 64, 126, 126)

    Note: use smaller batch_size to avoid OOM errors
    """
    def __init__(self, module):
        super(TimeDistributedImage, self).__init__()
        self.module = module

    def forward(self, x):
        # if input is of rank 4, add a batch dimension at the beginning
        if len(x.size()) == 4:
            x = x.unsqueeze(0)
        # Squash batch and timesteps into a single axis
        # (batch * timesteps, *input_size)
        x_reshape = x.contiguous().view(-1, *x.size()[-3:])
        y = self.module(x_reshape)
        if not isinstance(y, torch.Tensor):
            y = y.logits
        # unsquash batch and timesteps
        # reshape Y to (batch, timesteps, *input_size)
        y = y.view(x.size(0), x.size(1), *y.size()[1:])
        return y


class VideoClsf(BaseModel):
    """
    Feature extractor network with classic fully connected neural network that downsamples features by half every layer
    """

    def __init__(self, backbone: str, backbone_out_feat_size: int,
                 num_blocks: int, dropout: float, num_classes: int,
                 reduce_in_feats: int):
        super().__init__()

        blocks = []
        feature_ext = getattr(torchvision.models, backbone)(pretrained=True)
        final_layer = None
        if hasattr(feature_ext, 'classifier'):
            final_layer = 'classifier'
        elif hasattr(feature_ext, 'fc'):
            final_layer = 'fc'
        # replace the final clsf layer to a identity layer in the pretrained models
        setattr(feature_ext, final_layer, nn.Identity())

        # feature extracted is applied to each temporal slice
        blocks.append(TimeDistributedImage(feature_ext))

        # input feature space reduction layer, acts as encoder layer
        # if reduce_feat_num is not None, reduce input features with downsampling instead of residual block
        if reduce_in_feats is not None:
            blocks.append(nn.Linear(backbone_out_feat_size, reduce_in_feats))
            blocks.append(nn.Dropout(p=dropout))
            blocks.append(nn.ReLU())
        else:
            reduce_in_feats = backbone_out_feat_size

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


class VideoClsfBottleneck(BaseModel):
    """
    Feature extractor network with classic fully connected neural network that downsamples features by half every layer

    Initial bottleneck layer needs a vector of rank 3
    Note: input must be a vector of rank 3 with shape [batch, num_frames, num_initial_feats]
    """

    def __init__(self, backbone: str, backbone_out_feat_size: int,
                 num_blocks: int, dropout: float, num_classes: int,
                 num_frames: int, reduce_in_feats: int):
        super().__init__()

        blocks = []
        feature_ext = getattr(torchvision.models, backbone)(pretrained=True)
        final_layer = None
        if hasattr(feature_ext, 'classifier'):
            final_layer = 'classifier'
        elif hasattr(feature_ext, 'fc'):
            final_layer = 'fc'
        # replace the final clsf layer to a identity layer in the pretrained models
        setattr(feature_ext, final_layer, nn.Identity())

        # feature extracted is applied to each temporal slice
        blocks.append(TimeDistributedImage(feature_ext))

        # input feature space reduction layer, acts as encoder layer
        # if reduce_feat_num is not None, reduce input features with downsampling instead of residual block
        if reduce_in_feats is not None:
            blocks.append(nn.Linear(backbone_out_feat_size, reduce_in_feats))
            blocks.append(nn.Dropout(p=dropout))
            blocks.append(nn.ReLU())
        else:
            reduce_in_feats = backbone_out_feat_size

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
