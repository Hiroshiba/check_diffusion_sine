from pathlib import Path
from typing import Any, List, Optional, Union

import numpy
import torch
from torch import Tensor, nn
from typing_extensions import TypedDict

from check_diffusion_sine.config import Config
from check_diffusion_sine.network.predictor import Predictor, create_predictor


class GeneratorOutput(TypedDict):
    wave: Tensor


def to_tensor(array: Union[Tensor, numpy.ndarray, Any]):
    if not isinstance(array, (Tensor, numpy.ndarray)):
        array = numpy.asarray(array)
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    else:
        return array


class Generator(nn.Module):
    def __init__(
        self,
        config: Config,
        predictor: Union[Predictor, Path],
        use_gpu: bool,
    ):
        super().__init__()

        self.config = config
        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        if isinstance(predictor, Path):
            state_dict = torch.load(predictor, map_location=self.device)
            predictor = create_predictor(config.network)
            predictor.load_state_dict(state_dict)
        self.predictor = predictor.eval().to(self.device)

    def forward(
        self,
        noise_wave_list: List[Union[numpy.ndarray, Tensor]],
        lf0_list: List[Union[numpy.ndarray, Tensor]],
        step_num: int,
        return_every_step: bool = False,
    ):
        noise_wave_list = [
            to_tensor(noise_weve).to(self.device) for noise_weve in noise_wave_list
        ]
        lf0_list = [to_tensor(lf0).to(self.device) for lf0 in lf0_list]

        t = torch.linspace(0, 1, steps=step_num, device=self.device)

        wave_list_step = []

        with torch.inference_mode():
            wave_list = [t.clone() for t in noise_wave_list]

            if return_every_step:
                wave_list_step.append([t.clone() for t in wave_list])

            for i in range(step_num):
                k1_list = self.predictor.inference(
                    wave_list=wave_list,
                    lf0_list=lf0_list,
                    t=t[i].expand(len(lf0_list)),
                )
                k2_list = self.predictor.inference(
                    wave_list=[
                        (w + k1.unsqueeze(1) / (2 * step_num))
                        for w, k1 in zip(wave_list, k1_list)
                    ],
                    lf0_list=lf0_list,
                    t=(t[i] + 1 / (2 * step_num)).expand(len(lf0_list)),
                )
                for wave, k1, k2 in zip(wave_list, k1_list, k2_list):
                    wave += (k1 + k2).unsqueeze(1) / (2 * step_num)

                if return_every_step:
                    wave_list_step.append([t.clone() for t in wave_list])

        if not return_every_step:
            return [GeneratorOutput(wave=wave.squeeze(1)) for wave in wave_list]
        else:
            return [
                [GeneratorOutput(wave=wave.squeeze(1)) for wave in wave_list]
                for wave_list in wave_list_step
            ]
