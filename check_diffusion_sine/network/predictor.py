from typing import List

import torch
from espnet_pytorch_library.conformer.encoder import Encoder
from espnet_pytorch_library.nets_utils import make_non_pad_mask
from espnet_pytorch_library.tacotron2.decoder import Postnet
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from check_diffusion_sine.config import NetworkConfig


class Predictor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        block_num: int,
        post_layer_num: int,
    ):
        super().__init__()

        input_size = 1 + 1 + 1  # wave + lf0 + t
        self.pre = torch.nn.Linear(input_size, hidden_size)

        self.encoder = Encoder(
            idim=None,
            attention_dim=hidden_size,
            attention_heads=2,
            linear_units=hidden_size * 4,
            num_blocks=block_num,
            input_layer=None,
            dropout_rate=0.2,
            positional_dropout_rate=0.2,
            attention_dropout_rate=0.2,
            normalize_before=True,
            positionwise_layer_type="conv1d",
            positionwise_conv_kernel_size=3,
            macaron_style=True,
            pos_enc_layer_type="rel_pos",
            selfattention_layer_type="rel_selfattn",
            activation_type="swish",
            use_cnn_module=True,
            cnn_module_kernel=31,
        )

        self.post = torch.nn.Linear(hidden_size, 1)

        if post_layer_num > 0:
            self.postnet = Postnet(
                idim=1,
                odim=1,
                n_layers=post_layer_num,
                n_chans=hidden_size,
                n_filts=5,
                use_batch_norm=True,
                dropout_rate=0.5,
            )
        else:
            self.postnet = None

    def _mask(self, length: Tensor):
        x_masks = make_non_pad_mask(length).to(length.device)
        return x_masks.unsqueeze(-2)

    def forward(
        self,
        wave_list: List[Tensor],  # [(L, 1)]
        lf0_list: List[Tensor],  # [(L, 1)]
        t: Tensor,  # (B, )
    ):
        """
        B: batch size
        L: length
        """
        length_list = [w.shape[0] for w in wave_list]

        length = torch.tensor(length_list, device=wave_list[0].device)
        h = pad_sequence(wave_list, batch_first=True)  # (B, L, ?)

        lf0 = pad_sequence(lf0_list, batch_first=True)  # (B, L, ?)

        t = t.unsqueeze(dim=1).unsqueeze(dim=2)  # (B, 1, ?)
        t_feature = t.expand(t.shape[0], h.shape[1], t.shape[2])  # (B, L, ?)

        h = torch.cat((h, lf0, t_feature), dim=2)  # (B, L, ?)
        h = self.pre(h)

        mask = self._mask(length)
        h, _ = self.encoder(h, mask)

        output1 = self.post(h)
        if self.postnet is not None:
            output2 = output1 + self.postnet(output1.transpose(1, 2)).transpose(1, 2)
        else:
            output2 = output1

        return [output2[i, :l, 0] for i, l in enumerate(length_list)]

    def inference(
        self,
        wave_list: List[Tensor],  # [(L, 1)]
        lf0_list: List[Tensor],  # [(L, 1)]
        t: Tensor,  # (B, )
    ):
        h = self(
            wave_list=wave_list,
            lf0_list=lf0_list,
            t=t,
        )
        return h


def create_predictor(config: NetworkConfig):
    return Predictor(
        hidden_size=config.hidden_size,
        block_num=config.block_num,
        post_layer_num=config.post_layer_num,
    )
