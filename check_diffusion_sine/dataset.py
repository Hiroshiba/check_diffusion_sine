from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence, Union

import random
import numpy
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import TypedDict

from check_diffusion_sine.config import DatasetConfig


def sigmoid(a: Union[float, numpy.ndarray]):
    return 1 / (1 + numpy.exp(-a))


@dataclass
class DatasetInput:
    lf0: float
    sampling_length: int


@dataclass
class LazyDatasetInput:
    lf0: float
    sampling_length: int

    def generate(self):
        return DatasetInput(
            lf0=self.lf0,
            sampling_length=self.sampling_length,
        )


class DatasetOutput(TypedDict):
    input_wave: Tensor
    target_wave: Tensor
    noise_wave: Tensor
    lf0: Tensor
    t: Tensor


def generate_sin_wave(
    lf0,  # 対数f0
    phase,  # 位相、0~1
    length: int,
    sampling_rate: float,
):
    f0 = numpy.exp(lf0)
    return numpy.sin(
        2 * numpy.pi * f0 * numpy.arange(length) / sampling_rate + 2 * numpy.pi * phase
    )


def preprocess(d: DatasetInput, sampling_rate: float, with_diffusion: bool):
    target_wave = generate_sin_wave(
        lf0=d.lf0,
        phase=numpy.random.rand(),
        length=d.sampling_length,
        sampling_rate=sampling_rate,
    ).reshape(-1, 1)

    if with_diffusion:
        t = sigmoid(numpy.random.randn())
    else:
        t = 1
    noise_wave = numpy.random.randn(*target_wave.shape)
    input_wave = noise_wave + t * (target_wave - noise_wave)

    lf0 = numpy.full((d.sampling_length, 1), d.lf0)

    output_data = DatasetOutput(
        input_wave=torch.from_numpy(input_wave).to(torch.float32),
        target_wave=torch.from_numpy(target_wave).to(torch.float32),
        noise_wave=torch.from_numpy(noise_wave).to(torch.float32),
        lf0=torch.tensor(lf0).to(torch.float32),
        t=torch.tensor(t).to(torch.float32),
    )
    return output_data


class FeatureTargetDataset(Dataset):
    def __init__(
        self,
        datas: Sequence[Union[DatasetInput, LazyDatasetInput]],
        config: DatasetConfig,
    ):
        self.datas = datas
        self.preprocessor = partial(
            preprocess,
            sampling_rate=config.sampling_rate,
        )

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        if isinstance(data, LazyDatasetInput):
            data = data.generate()
        return self.preprocessor(data, with_diffusion=True)


def get_datas(config: DatasetConfig, num: int):
    lf0_low = config.lf0_low
    lf0_high = config.lf0_high
    min_sampling_length = config.min_sampling_length
    max_sampling_length = config.max_sampling_length

    datas = [
        LazyDatasetInput(
            lf0=(lf0_high - lf0_low) / num * i + lf0_low,
            sampling_length=(
                numpy.random.randint(min_sampling_length, max_sampling_length)
                if min_sampling_length != max_sampling_length
                else max_sampling_length
            ),
        )
        for i in range(num)
    ]
    return datas


def create_dataset(config: DatasetConfig):
    datas = get_datas(config, num=config.train_num + config.test_num)
    if config.seed is not None:
        random.Random(config.seed).shuffle(datas)

    tests, trains = datas[: config.test_num], datas[config.test_num :]

    valids = get_datas(config, num=config.test_num)

    def dataset_wrapper(datas, is_eval: bool):
        dataset = FeatureTargetDataset(
            datas=datas,
            config=config,
        )
        return dataset

    return {
        "train": dataset_wrapper(trains, is_eval=False),
        "test": dataset_wrapper(tests, is_eval=False),
        "eval": dataset_wrapper(tests, is_eval=True),
        "valid": dataset_wrapper(valids, is_eval=True),
    }
