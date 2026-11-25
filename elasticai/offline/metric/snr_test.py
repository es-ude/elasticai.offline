import unittest
from math import isclose

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from elasticai.offline.data_augmentation import generate_zero_frames

from .snr import (
    calculate_dsnr_tensor,
    calculate_snr,
    calculate_snr_cluster,
    calculate_snr_tensor,
)

seeds = st.integers(min_value=0, max_value=2**32 - 1)


def _build_noise_numpy(scale: float, size: int, rng: np.random.Generator) -> np.ndarray:
    return scale * 4 * (rng.random(size) - 0.5)


def _build_noise_tensor(scale: float, shape: torch.Size, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return scale * 4 * (torch.rand(shape, generator=gen) - 0.5)


class TestMetricSNR(unittest.TestCase):
    def setUp(self):
        fs_ana = 1e3
        t_end = 10 * np.pi
        time = np.linspace(start=0.0, stop=t_end, num=int(t_end * fs_ana), endpoint=True)
        scale_amp = 1.0
        self.signal_numpy = scale_amp * np.sin(time * 10.0)
        self.signal_torch = scale_amp * torch.sin(torch.Tensor(time) * 10.0).unsqueeze(0)
        self.signal_torch = self.signal_torch.repeat(10, 1)

    @given(seed=seeds)
    def test_snr_numpy_plus_60db(self, seed: int):
        rng = np.random.default_rng(seed)
        snr_out = list()
        for _ in range(10):
            noise = _build_noise_numpy(0.00001, self.signal_numpy.size, rng)
            signal = noise + self.signal_numpy
            rslt = calculate_snr(
                data=signal,
                mean=self.signal_numpy,
            )
            snr_out.append(rslt)
        rslt = np.mean(np.array(snr_out))
        self.assertTrue(isclose(rslt, 60.0, abs_tol=0.5))

    @given(seed=seeds)
    def test_snr_numpy_minus_0db(self, seed: int):
        rng = np.random.default_rng(seed)
        snr_out = list()
        for _ in range(10):
            noise = _build_noise_numpy(0.01, self.signal_numpy.size, rng)
            signal = noise + self.signal_numpy
            rslt = calculate_snr(
                data=signal,
                mean=self.signal_numpy,
            )
            snr_out.append(rslt)
        rslt = np.mean(np.array(snr_out))
        self.assertTrue(isclose(rslt, 0.0, abs_tol=0.5))

    @given(seed=seeds)
    def test_snr_numpy_minus_20db(self, seed: int):
        rng = np.random.default_rng(seed)
        snr_out = list()
        for _ in range(10):
            noise = _build_noise_numpy(0.1, self.signal_numpy.size, rng)
            signal = noise + self.signal_numpy
            rslt = calculate_snr(
                data=signal,
                mean=self.signal_numpy,
            )
            snr_out.append(rslt)
        rslt = np.mean(np.array(snr_out))
        self.assertTrue(isclose(rslt, -20.0, abs_tol=0.5))

    @given(seed=seeds)
    def test_snr_tensor_plus_60db(self, seed: int):
        noise = _build_noise_tensor(0.00001, self.signal_torch.shape, seed)
        signal = noise + self.signal_torch
        rslt = calculate_snr_tensor(data=signal, mean=self.signal_torch)
        rslt = rslt.mean().item()
        self.assertTrue(isclose(rslt, 60.0, abs_tol=0.5))

    @given(seed=seeds)
    def test_snr_tensor_minus_0db(self, seed: int):
        noise = _build_noise_tensor(0.01, self.signal_torch.shape, seed)
        signal = noise + self.signal_torch
        rslt = calculate_snr_tensor(data=signal, mean=self.signal_torch)
        rslt = rslt.mean().item()
        self.assertTrue(isclose(rslt, 0.0, abs_tol=0.5))

    @given(seed=seeds)
    def test_snr_tensor_minus_20db(self, seed: int):
        noise = _build_noise_tensor(0.1, self.signal_torch.shape, seed)
        signal = noise + self.signal_torch
        rslt = calculate_snr_tensor(data=signal, mean=self.signal_torch)
        rslt = rslt.mean().item()
        self.assertTrue(isclose(rslt, -20.0, abs_tol=0.5))

    @given(seed=seeds)
    def test_dsnr_tensor(self, seed: int):
        noise0 = _build_noise_tensor(0.1, self.signal_torch.shape, seed)
        noise1 = _build_noise_tensor(0.01, self.signal_torch.shape, seed + 1)
        signal0 = noise0 + self.signal_torch
        signal1 = noise1 + self.signal_torch
        rslt = calculate_dsnr_tensor(data=signal0, pred=signal1, mean=self.signal_torch)
        rslt = rslt.mean().item()
        self.assertTrue(isclose(rslt, +20.0, abs_tol=0.5))

    def test_snr_cluster(self):
        self.frames_zero = generate_zero_frames(
            frame_size=32,
            num_frames=10,
            snr_range=[-20, -10],
            fs=20e3,
            return_int=False,
        )
        dataset = {
            "data": np.concatenate((self.frames_zero[0], self.frames_zero[0] + 4.5), axis=0),
            "label": np.concatenate((self.frames_zero[1], self.frames_zero[1] + 1), axis=0),
            "mean": np.concatenate(
                (
                    self.frames_zero[2].reshape(1, 32),
                    self.frames_zero[2].reshape(1, 32) + 4.5,
                ),
                axis=0,
            ),
        }
        rslt = calculate_snr_cluster(
            frames_in=dataset["data"],
            frames_cl=dataset["label"],
            frames_mean=dataset["mean"],
        )
        self.assertEqual(rslt.shape, (np.unique(dataset["label"]).size, 4))


if __name__ == "__main__":
    unittest.main()
