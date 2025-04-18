import warnings
warnings.filterwarnings("ignore")
import argparse
import os
from models import unet3d, MGFA
from train import train


model_dict = {
    'MGFA': MGFA.MGFANet(),
    'baseline': unet3d.UNet3D(),
}

parser = argparse.ArgumentParser()

parser.add_argument('--base_dir', type=str, default=r'')
parser.add_argument('--source_datapath', type=str, default=r'')
parser.add_argument('--train_dir', type=str, default=r'../data/train')
parser.add_argument('--val_dir', type=str, default=r'../data/val')
parser.add_argument('--save_dir', type=str, default=r'exp_log')

parser.add_argument('--model', choices=model_dict.keys(), default='MGFA')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--step', type=int, default=20)
parser.add_argument('--epoch', type=int, default=200)
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

    train(args, model_dict)
