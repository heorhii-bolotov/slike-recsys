import os
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pl_modules import NCFDataModule, NCF

import wandb
from logger import create_logger

logging = create_logger()


def parse_args():
    """
    PYTHONPATH=./src python src/trainer.py --path data --epochs 10 --batch_size 256 --layers 32 32 32 --weight_decay 0.0001 --learner adagrad --verbose True
    """
    parser = ArgumentParser(description='Implicit Recommender')
    parser.add_argument('--path', type=Path, default=Path('../'), help='Input data path for train\\test')
    parser.add_argument('--train_fp', type=Path, default=Path('train.csv'), help='Train file name')
    parser.add_argument('--test_fp', type=Path, default=Path('test.csv'), help='Test file name')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--bs', type=int, default=2048, help='Batch size')
    parser.add_argument('--layers', type=int, nargs='+', default=[32, 48, 32, 16],
                        help='Size of each layer. The first layer is a concatenation of user and item embeddings. So '
                             'layers[0]/2 is the embedding size', )
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Regularization for each layer')
    parser.add_argument('--num_neg_train', type=int, default=200,
                        help='Number of negative instances while training')
    parser.add_argument('--num_neg_test', type=int, default=20,
                        help='Number of negative instances while testing')  # doesn't work
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=.1, help='Dropout prob after each dense layer')
    parser.add_argument('--learner', default='adam', choices=('adagrad', 'adam', 'rmsprop', 'sgd'),
                        help='Specify an optimizer')
    parser.add_argument('--verbose', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Show performance per X iterations')
    parser.add_argument('--seed', type=int, default=33,
                        help='Random state for all')
    parser.add_argument('--ckpt_callback_fp', type=Path, default=Path(datetime.today().strftime("%Y-%m-%d")),
                        help='Checkpoint callback filepath')
    parser.add_argument('--experiment_name', type=str, default=f'{datetime.today().strftime("%Y-%m-%d")}',
                        help='Experiment filepath')
    parser.add_argument('--map_item_id_name_fp', type=Path,
                        default=Path(f'{datetime.today().strftime("%Y-%m-%d")}_map_item_id_name_fp.json'), help='')
    return parser.parse_args()


def print_name_value_type(args):
    for arg in vars(args):
        value = getattr(args, arg)
        logging.debug(f'{arg}, {value}, {type(value)}')


def update_namespace(ns: Namespace, **kwargs) -> Namespace:
    ns = deepcopy(ns)
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


def get_device() -> torch.device:
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def main():
    args = parse_args()
    print_name_value_type(args)  # debug

    device = get_device()
    logging.debug(f"Device available: {device}")

    # set seed
    pl.seed_everything(getattr(args, 'seed'), workers=True)

    # data
    data = NCFDataModule(args)
    data.setup(stage='fit')

    # update namespace for model
    args = update_namespace(args, n_users=data.num_users, n_items=data.num_items, device=device)
    print_name_value_type(args)  # debug

    ncf = NCF(args)

    ckpt_fp = getattr(args, 'ckpt_callback_fp')
    ckpt_fp.absolute().mkdir(exist_ok=True, parents=True)
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss',
        verbose=True,
        dirpath=str(ckpt_fp.absolute()),
        filename=str(ckpt_fp.name),
        save_top_k=1,
        mode='min'
    )

    wandb.login(key=os.getenv('wandb_api'))
    trainer = pl.Trainer(gpus=[0],
                         logger=WandbLogger(getattr(args, 'experiment_name')),
                         callbacks=[ckpt_callback],
                         max_epochs=getattr(args, 'epochs'))
    trainer.fit(model=ncf, datamodule=data)


if __name__ == "__main__":
    main()
