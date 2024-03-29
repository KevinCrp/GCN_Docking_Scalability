import argparse
import multiprocessing as mp
import os
import os.path as osp

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.strategies import DDPStrategy

import data
from models.model import Model


def get_str_time(sec: float):
    # https://www.tutsmake.com/python-convert-time-in-seconds-to-days-hours-minutes-seconds/
    # convert seconds to day, hour, minutes and seconds
    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    # print day, hour, minutes and seconds
    return "{:d}-{:d}:{:d}:{:d} (d-h:m:s)".format(int(day),
                                                  int(hour),
                                                  int(minutes),
                                                  int(seconds))


def train(batch_size: int,
          lr: float,
          strategy: str,
          accelerator_name: str,
          nb_devices: int,
          num_nodes: int,
          num_epochs: int,
          model_name: str,
          experiments_path):
    pl.utilities.seed.seed_everything(42)
    exp_model_name = 'PERFS'

    logger = pl.loggers.CSVLogger(
        experiments_path, name=exp_model_name)
    print("Version: {}".format(logger.version))

    model = Model(lr=lr,
                  weight_decay=1e-4,
                  loss=F.mse_loss,
                  model_name=model_name)

    nb_param_trainable = model.get_nb_parameters(only_trainable=True)
    nb_param = model.get_nb_parameters(only_trainable=False)
    logger.log_metrics(
        {'nb_param_trainable': torch.tensor(nb_param_trainable)})
    logger.log_metrics({'nb_param': torch.tensor(nb_param)})
    bs = batch_size // (num_nodes *
                        nb_devices) if nb_devices > 0 else batch_size
    print('Batch size per device = ', bs)
    datamodule = data.PDBBindDataModule(root='data',
                                        batch_size=bs,
                                        num_workers=mp.cpu_count(),
                                        only_pocket=True)
    timer = Timer()
    callbacks_list = [timer]

    strategy = None
    devices = None
    if accelerator_name != '' and nb_devices > 0:
        strategy = DDPStrategy(find_unused_parameters=False,
                               accelerator=accelerator_name)
        devices = nb_devices
    trainer = pl.Trainer(devices=devices,
                         strategy=strategy,
                         callbacks=callbacks_list,
                         max_epochs=num_epochs,
                         logger=logger,
                         log_every_n_steps=1,
                         num_sanity_val_steps=0)

    trainer.fit(model, datamodule)
    training_time = timer.time_elapsed("train")

    metrics_path = osp.join(experiments_path, exp_model_name,
                            "version_{}".format(logger.version), 'metrics.csv')

    if trainer.is_global_zero:
        df_metrics = pd.read_csv(metrics_path)
        last_row = df_metrics.iloc[-1:]
        print(last_row)

        print('\n{}'.format(training_time), end='')
        print(',{},{},{}'.format(last_row['ep_end/train_loss'].item(),
                                last_row['ep_end/train_r2_score'].item(),
                                last_row['ep_end/train_pearson'].item()))


def main(args, experiments_path):
    train(args.batch_size,
          args.lr,
          args.strategy,
          args.accelerator,
          args.nb_devices,
          args.num_nodes,
          args.num_epochs,
          args.model_name,
          experiments_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',
                        type=int,
                        help='Total batch size',
                        default=16)
    parser.add_argument('--lr',
                        type=float,
                        help='Learning Rate',
                        default=1e-3)
    parser.add_argument('--strategy',
                        type=str,
                        help='Data Distributed Strategy',
                        default='ddp')
    parser.add_argument('--accelerator',
                        type=str,
                        help='The accelerator class',
                        choices=['gpu', 'tpu', 'ipu'],
                        default='')
    parser.add_argument('--nb_devices',
                        type=int,
                        help='Number of devices to accelerate over',
                        default=0)
    parser.add_argument('--num_nodes',
                        type=int,
                        help='Number of nodes to accelerate over',
                        default=1)
    parser.add_argument('--num_epochs',
                        type=int,
                        help='Number of epochs',
                        default=1)
    parser.add_argument('--model_name',
                        type=str,
                        help='The model name',
                        choices=['MolGCN', 'MolGAT',
                                 'MolAttentiveFP', 'MolAttentiveFP_EA'],
                        required=True)
    args = parser.parse_args()

    experiments_path = 'experiments'
    if not osp.isdir(experiments_path):
        os.mkdir(experiments_path)

    main(args, experiments_path)
