import logging
import math
from argparse import Namespace
from typing import Optional
from pathlib import Path
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import Interactions

from network import MLP


class NCFDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: Optional[str] = None):
        data_path = getattr(self.args, 'path')
        train_dataset = Interactions(Path(data_path / getattr(self.args, 'train_fp')),
                                     map_item_id_name_fp=getattr(self.args, 'map_item_id_name_fp'))
        self.num_users, self.num_items = train_dataset.num_users, train_dataset.num_items
        self.test_dataset = Interactions(Path(data_path / getattr(self.args, 'test_fp')), mode='test',
                                         map_item_id_name_fp=getattr(self.args, 'map_item_id_name_fp'))
        self.test_dataset.set_num_users_items(self.num_users, self.num_items)

        # split train on train val - ugly
        train = train_dataset.inputs.sample(frac=0.8, random_state=getattr(self.args, 'seed'))
        val = train_dataset.inputs.drop(train.index)
        train_dataset.inputs = train
        val_dataset = deepcopy(train_dataset)
        val_dataset.inputs = val
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=getattr(self.args, 'bs'), shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=getattr(self.args, 'bs'), shuffle=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=getattr(self.args, 'bs'))


class NCF(pl.LightningModule):
    def __init__(self, args: Namespace):
        super().__init__()
        self.save_hyperparameters(args)
        self.model = MLP(**self.hparams)
        if getattr(self.hparams, 'verbose'):
            logging.info(self.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        batch = {k: v.to(dtype=torch.long) for k, v in batch.items()}
        logits = self(batch)
        rating = batch['rating'].float().view(logits.size())
        loss = F.binary_cross_entropy(logits, rating)
        self.log('epoch', self.trainer.current_epoch, on_step=False, on_epoch=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = {k: v.to(dtype=torch.long) for k, v in batch.items()}
        logits = self(batch)
        rating = batch['rating'].float().view(logits.size())
        loss = F.binary_cross_entropy(logits, rating)
        self.log('val_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), weight_decay=self.hparams.weight_decay)


if __name__ == '__main__':
    from trainer import parse_args, print_name_value_type

    args = parse_args()
    print_name_value_type(args)  # debug

    # data
    data = NCFDataModule(args)
    data.setup(stage='fit')

    logging.debug(data.num_users)
    logging.debug(data.num_items)
