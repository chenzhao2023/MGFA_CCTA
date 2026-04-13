# config.py
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class TestConfig:
    overlap: List[int]
    patch_size: List[int]
    batch_size: int
    num_workers: int
    class_thrd: float
    back_thrd: int


MODEL_TEST_CONFIGS: Dict[str, TestConfig] = {
    "AGFA": TestConfig(
        overlap=[64, 80, 80],
        patch_size=[128, 160, 160],
        batch_size=1,
        num_workers=1,
        class_thrd=0.5,
        back_thrd=5000,
    ),
    "LCT": TestConfig(
        overlap=[64, 80, 80],
        patch_size=[128, 160, 160],
        batch_size=1,
        num_workers=1,
        class_thrd=0.5,
        back_thrd=5000,
    ),
    "casnet": TestConfig(
        overlap=[64, 80, 80],
        patch_size=[128, 160, 160],
        batch_size=1,
        num_workers=1,
        class_thrd=0.5,
        back_thrd=5000,
    ),
    "MGFA": TestConfig(
        overlap=[64, 80, 80],
        patch_size=[128, 160, 160],
        batch_size=1,
        num_workers=1,
        class_thrd=0.5,
        back_thrd=5000,
    ),
    "densenet": TestConfig(

        overlap=[64, 80, 80],
        patch_size=[128, 160, 160],
        batch_size=1,
        num_workers=1,
        class_thrd=0.5,
        back_thrd=5000,
    ),
    "csnet": TestConfig(

        overlap=[64, 80, 80],
        patch_size=[128, 160, 160],
        batch_size=1,
        num_workers=1,
        class_thrd=0.5,
        back_thrd=5000,
    ),
    "vnet": TestConfig(

        overlap=[64, 80, 80],
        patch_size=[128, 160, 160],
        batch_size=1,
        num_workers=1,
        class_thrd=0.5,
        back_thrd=5000,
    ),
    "unet": TestConfig(

        overlap=[64, 80, 80],
        patch_size=[128, 160, 160],
        batch_size=1,
        num_workers=1,
        class_thrd=0.5,
        back_thrd=5000,
    ),

}


def get_test_config(model_name: str) -> TestConfig:

    if model_name not in MODEL_TEST_CONFIGS:
        raise ValueError(f"Unknown model name '{model_name}'. "
                         f"Available: {list(MODEL_TEST_CONFIGS.keys())}")
    return MODEL_TEST_CONFIGS[model_name]
