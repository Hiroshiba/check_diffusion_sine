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
        noise_weve_list: List[Union[numpy.ndarray, Tensor]],
        lf0_list: List[Union[numpy.ndarray, Tensor]],
        step_num: int,
    ):
        noise_weve_list = [
            to_tensor(noise_weve).to(self.device) for noise_weve in noise_weve_list
        ]
        lf0_list = [to_tensor(lf0).to(self.device) for lf0 in lf0_list]

        with torch.inference_mode():
            t = torch.linspace(0, 1, steps=step_num, device=self.device)
            wave_list = [t.clone() for t in noise_weve_list]
            for i in range(step_num):
                output_list = self.predictor.inference(
                    wave_list=wave_list,
                    lf0_list=lf0_list,
                    t=t[i].expand(len(lf0_list)),
                )
                for wave, output in zip(wave_list, output_list):
                    wave += output / step_num

        return [GeneratorOutput(wave=output) for output in output_list]
