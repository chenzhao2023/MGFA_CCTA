# train_config.py
from dataclasses import dataclass
from typing import Dict

class datapro:
    basedir = r'E:\proj\CA_segment'   #项目的根目录
    source_datapath = r'D:\imgcas'    # 原始数据
    cou_dir = r'F:\UltraLight-VM-UNet-main\UltraLight-VM-UNet-main\mergedall\contour' # 轮廓数据
    den_traindir = r'data\train'  #处理后数据存放位置
    den_valdir = r'data\val'     #处理后数据存放位置
    seed = 58790





@dataclass
class TrainConfig:
    model: str
    batch_size: int
    earlystop: int
    num_workers: int
    half: bool
    init_scale: int
    max_norm: float
    scheduler: bool



TRAIN_CONFIGS: Dict[str, TrainConfig] = {
    "MGFA": TrainConfig(
        model="MGFA",
        batch_size=4,
        earlystop=50,
        num_workers=4,
        half=False,
        init_scale=1024,
        max_norm=1.0,
        scheduler=True,
    ),
    "unet": TrainConfig(
        model="unet",
        batch_size=4,
        earlystop=50,
        num_workers=4,
        half=True,
        init_scale=1024,
        max_norm=1.0,
        scheduler=True,
    ),
    "vnet": TrainConfig(
        model="vnet",
        batch_size=2,
        earlystop=50,
        num_workers=4,
        half=False,
        init_scale=1024,
        max_norm=1.0,
        scheduler=True,
    ),
    "densenet": TrainConfig(
        model="densenet",
        batch_size=2,
        earlystop=50,
        num_workers=4,
        half=False,
        init_scale=1024,
        max_norm=1.0,
        scheduler=True,
    ),
    "csnet": TrainConfig(
        model="csnet",
        batch_size=2,
        earlystop=50,
        num_workers=4,
        half=False,
        init_scale=1024,
        max_norm=1.0,
        scheduler=True,
    ),
    "casnet": TrainConfig(
        model="casnet",
        batch_size=4,
        earlystop=50,
        num_workers=4,
        half=False,
        init_scale=1024,
        max_norm=1.0,
        scheduler=True,
    ),
    "LCT": TrainConfig(
        model="LCT",
        batch_size=1,
        earlystop=50,
        num_workers=4,
        half=False,
        init_scale=1024,
        max_norm=1.0,
        scheduler=True,
    ),

}


def get_train_config(model_name: str) -> TrainConfig:
    if model_name not in TRAIN_CONFIGS:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Available: {list(TRAIN_CONFIGS.keys())}"
        )
    return TRAIN_CONFIGS[model_name]
