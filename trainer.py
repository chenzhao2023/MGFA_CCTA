import warnings
warnings.filterwarnings("ignore")
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from models import unet3d, MGFA, DenseVoxelNet, csnet_3d, vnet, casnet_3d, LCT_DR_UNet_new, AGFA
from train import train
from train_cfg import get_train_config
from data_process import dataprocess
from train_cfg import datapro

model_dict = {
    # 'MGFA': MGFA.MGFANet(),
    'unet': unet3d.UNet3D(),
    'vnet': vnet.VNet(),
    'densenet': DenseVoxelNet.DenseVoxelNet(),
    'csnet': csnet_3d.CSNet3D(),
    'casnet': casnet_3d.CASNet3D(),
    'LCT': LCT_DR_UNet_new.LCT_new_DR_UNet_new(),
}

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir', type=str, default=datapro.basedir)  # project dir
parser.add_argument('--source_datapath', type=str, default=datapro.source_datapath)
parser.add_argument('--train_dir', type=str, default=datapro.den_traindir)
parser.add_argument('--val_dir', type=str, default=datapro.den_valdir)
parser.add_argument('--save_dir', type=str, default=r'exp_log')

parser.add_argument('--model', choices=model_dict.keys(), default='') # need model name
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--step', type=int, default=20)
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--earlystop', type=int, default=15)
parser.add_argument('--seed', type=int, default=58790)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda')

parser.add_argument('--scheduler', action='store_true', default=True, help='')
parser.add_argument('--T_0', type=int, default=20)
parser.add_argument('--T_mult', type=int, default=2)
parser.add_argument('--eta_min', type=float, default=1e-6)

parser.add_argument('--overlap', type=int, nargs='+', default=[32, 40, 40])
parser.add_argument('--patch_size', type=int, nargs='+', default=[128, 160, 160])

parser.add_argument('--half', action='store_true', default=True, help='')
parser.add_argument('--init_scale', type=int, default=1024)
parser.add_argument('--max_norm', type=float, default=1.0)

parser.add_argument('--check_point', action='store_true', default=False)
parser.add_argument('--check_point_path', type=str, default=r'')

parser.add_argument('--data_size', type=float, default=1.0)

args = parser.parse_args()

if __name__ == '__main__':

    if not os.path.exists(datapro.den_traindir):
        dataprocess() #处理数据


    for i in model_dict.keys():  # 遍历模型字典 训练所有的
        args.model = str(i)
        print('*' * 60)
        print('Strat training:',args.model)
        print('*' * 60)

        cfg = get_train_config(args.model)

        args.batch_size = cfg.batch_size
        args.earlystop = cfg.earlystop
        args.num_workers = cfg.num_workers
        args.half = cfg.half
        args.init_scale = cfg.init_scale
        args.max_norm = cfg.max_norm
        args.scheduler = cfg.scheduler

        train(args, model_dict)
