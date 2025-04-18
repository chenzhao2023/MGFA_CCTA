import warnings
warnings.filterwarnings("ignore")

from dataset import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from tqdm import tqdm
from metrics import *
import random
from torch.cuda.amp import autocast, GradScaler
from loss import CombinedLoss, CombinedLoss_rba
import logging
import os
import datetime


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose

        self.best_loss = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score

            self.counter = 0






def setup_logger_paths(args):
    model_name = args.model
    base_dir = os.path.join(args.base_dir, args.save_dir)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    log_dir = os.path.join(base_dir, model_name, current_time, "logs")
    writer_dir = os.path.join(base_dir, model_name, current_time, "writer")
    checkpoint_dir = os.path.join(base_dir, model_name, current_time, "checkpoints")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = os.path.join(log_dir, 'training.log')
    setup_logger(log_dir)

    logging.info(f"log file path: {log_dir}")
    logging.info(f"Checkpoint save path: {checkpoint_dir}")

    return log_dir, checkpoint_dir, writer_dir


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def close_logger():
    logger = logging.getLogger()

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    print("Logger closed")

def save_checkpoint(model, optimizer, scheduler, epoch, save_path, best_dice, best=False):

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'best_dice': best_dice
    }
    if not best:
        save_path = os.path.join(save_path, f"checkpoint_{epoch}.pth")
        logging.info(f"Checkpoint saved to {save_path}")
    else:
        save_path = os.path.join(save_path, f"checkpoint_best.pth")
        logging.info(f"Best Dice Checkpoint saved to {save_path}")
    torch.save(checkpoint, save_path)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"model has {total_params} params")



def load_checkpoint(model, optimizer, scheduler, load_path):

    checkpoint = torch.load(load_path)

    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    best_dice = checkpoint['best_dice']
    logging.info(f"Checkpoint loaded from {load_path}, epoch {epoch}，best dice {best_dice}")

    return epoch, best_dice

def freeze_batchnorm_stats(net):
    for m in net.modules():
        for child in m.children():
            if type(child) == nn.BatchNorm3d:
                child.track_running_stats = False
                child.running_mean = None
                child.running_var = None
    for m in net.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.track_running_stats = False
    count = 0
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm3d):
            count += 1
            if count >= 2:
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    return net


def set_seed(random_seed=58790):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"random seed：{random_seed}")


def log_args(args):
    logging.info("training params：")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")


def train(args, models):

    early_stopping = EarlyStopping(patience=30, verbose=True)

    log_dir, checkpoint_dir, writer_dir = setup_logger_paths(args)
    writer = SummaryWriter(log_dir=writer_dir)
    log_args(args)
    set_seed(args.seed)


    device = torch.device(args.device)

    net = models[args.model].to(device)
    net = nn.DataParallel(net)
    count_parameters(net)
    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
    scaler = GradScaler(init_scale=args.init_scale)
    criterion = CombinedLoss().to(device)

    if args.check_point:
        strat_epoch, best_dice = load_checkpoint(net, opt, scheduler, args.check_point_path)
    else:
        strat_epoch=0
        best_dice = 0.

    train_set = VesselsSegmentionDataSetsave_t(args.train_dir, args.data_size)
    val_set = VesselsSegmentionDataSetsave_t(args.val_dir, args.data_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False)


    for epoch in tqdm(range(strat_epoch, args.epoch)):

        net.train()
        running_loss, dice_coefficient_value_avg = \
            train_epoch(args,train_loader,scheduler,device,opt,net,criterion,scaler)

        logging.info('--------------------------------------------------------------------')
        logging.info(f"Epoch {epoch + 1}, Train loss is {running_loss:.5f}, Dice is {dice_coefficient_value_avg:.5f}")
        writer.add_scalar('train_loss', running_loss, epoch + 1)
        writer.add_scalar('train_dice', dice_coefficient_value_avg, epoch + 1)


        net.eval()
        net = freeze_batchnorm_stats(net)

        running_loss_te, dice_coefficient_value_avg_te = \
            eval_epoch(args, val_loader, device, net, criterion)

        logging.info(f"Epoch {epoch + 1}, Val  loss is {running_loss_te:.5f}, Dice is {dice_coefficient_value_avg_te:.5f}")




        if (epoch + 1) % args.step == 0:
            save_checkpoint(net, opt, scheduler, epoch, checkpoint_dir, best_dice)

        writer.add_scalar('eval_loss', running_loss_te, epoch + 1)
        writer.add_scalar('eval_dice', dice_coefficient_value_avg_te, epoch + 1)

        if dice_coefficient_value_avg_te > best_dice:
            best_dice = dice_coefficient_value_avg_te
            save_checkpoint(net, opt, scheduler, epoch, checkpoint_dir, best_dice,True)
        logging.info('--------------------------------------------------------------------')

        early_stopping(running_loss_te, net)
        if early_stopping.early_stop:
            save_checkpoint(net, opt, scheduler, epoch, checkpoint_dir, best_dice)
            print("Early stopping")
            writer.close()
            close_logger()
            break
    close_logger()
    writer.close()



def train_epoch(args, train_loader, scheduler, device, opt, net, criterion, scaler):
    running_loss = 0.
    dice_coefficient_value_avg = 0.
    for i, data in enumerate(train_loader, 0):

        if args.scheduler:
            scheduler.step()

        inputs, labels, name = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        opt.zero_grad()
        if args.half:
            with autocast():
                outputs_tr = net(inputs[:,:1],inputs[:,1:2])

                loss = criterion(outputs_tr, labels)
                outputs_tr = torch.sigmoid(outputs_tr)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=args.max_norm)
            scaler.step(opt)
            scaler.update()
        else:
            outputs_tr = net(inputs[:,:1],inputs[:,1:2])

            loss = criterion(outputs_tr, labels)
            outputs_tr = torch.sigmoid(outputs_tr)
            loss.backward()
            opt.step()


        running_loss += loss.item()
        outputs_tr[outputs_tr > 0.5] = 1
        outputs_tr[outputs_tr < 0.5] = 0
        # outputs_tr = torch.argmax(outputs_tr, dim=1)

        labels = labels.detach()
        out = outputs_tr.detach()

        dice_coefficient_value = compute_dice_coefficient(labels, out)
        dice_coefficient_value_avg += dice_coefficient_value
        # print(loss, dice_coefficient_value)

    running_loss = running_loss / (len(train_loader))
    dice_coefficient_value_avg /= len(train_loader)
    return running_loss, dice_coefficient_value_avg


def eval_epoch(args, test_loader, device, net, criterion):
    running_loss = 0.
    dice_coefficient_value_avg_te = 0.
    with torch.no_grad():
        for i, (inpute, labele, name) in enumerate(test_loader):
            inpute = inpute.to(device)
            labele = labele.to(device)
            if args.half:
                with autocast():
                    outputs_te = net(inpute[:,:1],inpute[:,1:2])

                    loss = criterion(outputs_te, labele.to(torch.float32))
                    outputs_te = torch.sigmoid(outputs_te)
            else:
                outputs_te = net(inpute[:,:1],inpute[:,1:2])

                loss = criterion(outputs_te, labele.to(torch.float32))
                outputs_te = torch.sigmoid(outputs_te)
            running_loss += loss.item()
            outputs_te[outputs_te > 0.5] = 1
            outputs_te[outputs_te < 0.5] = 0
            # outputs_te = torch.argmax(outputs_te, dim=1)
            dice_coefficient_value_te = compute_dice_coefficient(outputs_te, labele)
            dice_coefficient_value_avg_te += dice_coefficient_value_te
    running_loss = running_loss / (len(test_loader))
    dice_coefficient_value_avg_te /= (len(test_loader))

    return running_loss, dice_coefficient_value_avg_te