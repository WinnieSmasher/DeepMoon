from __future__ import annotations

import random

import torch

from deepmoon.data.transforms import RandomFlip, RandomRotate90, RandomShift


class FixedRandom:
    def __init__(self, values: list[float], ints: list[int]):
        self.values = values
        self.ints = ints

    def random(self) -> float:
        return self.values.pop(0)

    def randint(self, start: int, end: int) -> int:
        return self.ints.pop(0)


def test_random_flip_applies_to_image_and_mask() -> None:
    image = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    mask = image.clone()
    transform = RandomFlip(horizontal_prob=1.0, vertical_prob=1.0, rng=random.Random(1))
    image_out, mask_out = transform(image, mask)
    expected = torch.tensor([[[8.0, 7.0, 6.0], [5.0, 4.0, 3.0], [2.0, 1.0, 0.0]]])
    assert torch.equal(image_out, expected)
    assert torch.equal(mask_out, expected)


def test_random_shift_respects_requested_offsets() -> None:
    image = torch.zeros((1, 5, 5), dtype=torch.float32)
    image[:, 2, 2] = 1.0
    mask = image.clone()
    transform = RandomShift(max_pixels=1, rng=FixedRandom(values=[], ints=[1, -1]))
    image_out, mask_out = transform(image, mask)
    assert image_out[:, 1, 3].item() == 1.0
    assert torch.equal(image_out, mask_out)


def test_random_rotate90_rotates_consistently() -> None:
    image = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    mask = image.clone()
    transform = RandomRotate90(probability=1.0, rng=FixedRandom(values=[0.0], ints=[1]))
    image_out, mask_out = transform(image, mask)
    expected = torch.tensor([[[2.0, 4.0], [1.0, 3.0]]])
    assert torch.equal(image_out, expected)
    assert torch.equal(mask_out, expected)
