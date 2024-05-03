from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing_extensions import TypedDict

from check_diffusion_sine.config import ModelConfig
from check_diffusion_sine.dataset import DatasetOutput
from check_diffusion_sine.network.predictor import Predictor


class ModelOutput(TypedDict):
    loss: Tensor
    data_num: int


def reduce_result(results: List[ModelOutput]):
    result: Dict[str, Any] = {}
    sum_data_num = sum([r["data_num"] for r in results])
    for key in set(results[0].keys()) - {"data_num"}:
        values = [r[key] * r["data_num"] for r in results]
        if isinstance(values[0], Tensor):
            result[key] = torch.stack(values).sum() / sum_data_num
        else:
            result[key] = sum(values) / sum_data_num
    return result


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, predictor: Predictor):
        super().__init__()
        self.model_config = model_config
        self.predictor = predictor

    def forward(self, data: DatasetOutput) -> ModelOutput:
        output_list: List[Tensor] = self.predictor(
            wave_list=data["input_wave"],
            lf0_list=data["lf0"],
            t=torch.stack(data["t"]),
        )

        output = torch.cat(output_list)

        target_wave = torch.cat(data["target_wave"]).squeeze(1)
        noise_wave = torch.cat(data["noise_wave"]).squeeze(1)
        diff_wave = target_wave - noise_wave

        loss = F.mse_loss(output, diff_wave)

        return ModelOutput(
            loss=loss,
            data_num=len(data),
        )
