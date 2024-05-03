from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import Literal, TypedDict

from check_diffusion_sine.dataset import DatasetOutput
from check_diffusion_sine.generator import Generator, GeneratorOutput


class EvaluatorOutput(TypedDict):
    value: Tensor
    data_num: int


class Evaluator(nn.Module):
    judge: Literal["min", "max"] = "min"

    def __init__(self, generator: Generator, step_num: int):
        super().__init__()
        self.generator = generator
        self.step_num = step_num

    def forward(self, data: DatasetOutput) -> EvaluatorOutput:
        output_list: List[GeneratorOutput] = self.generator(
            noise_weve_list=data["noise_wave"],
            lf0_list=data["lf0"],
            step_num=self.step_num,
        )

        target_wave = torch.cat(data["target_wave"]).squeeze(1)

        output_wave = torch.cat([output["wave"] for output in output_list])

        diff_wave = F.mse_loss(output_wave, target_wave)

        value = diff_wave

        return EvaluatorOutput(
            value=value,
            data_num=len(data),
        )
