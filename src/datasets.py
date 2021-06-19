import json
from abc import ABC
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from logger import create_logger

logging = create_logger()


class InteractionsMeta(ABC):

    @staticmethod
    def read_df(fp: Path) -> pd.DataFrame:
        return pd.read_csv(fp)

    @staticmethod
    def get_df_user_item_size(df: pd.DataFrame, user_col='user_id', item_col='product_id') -> Tuple[int, int]:
        # return df[user_col].unique().size, df[item_col].unique().size
        return df[user_col].max() + 1, df[item_col].max() + 1

    @staticmethod
    def get_item_id_name_map(df: pd.DataFrame, item_id_col='product_id', item_name_col='product_name') -> Dict[int, str]:
        return dict(zip(df[item_id_col].unique().tolist(), df[item_name_col].tolist()))

    @staticmethod
    def negative_sample(df: pd.DataFrame, num_negatives: int, user_col='user_id', item_col='product_id'):
        """
                user_id  product_id  rating
            0         1         196       1
            1         1         196       1
            2         1         196       1
            3         1         196       1
            4         1         196       1
            ..      ...         ...     ...
            69        1       43176       0
            70        1       45026       0
            71        1       47801       0
            72        1       49286       0
            73        1       49397       0
        """

        items = df[item_col].unique()

        def sampler(x):
            x_ = np.random.choice(items, num_negatives, replace=True)  # can generate the same item several times
            x_ = np.setdiff1d(x_, np.intersect1d(x, x_))
            return x_

        df_list = []
        for u, i in tqdm(df.groupby(user_col)[item_col]):
            i_neg = sampler(i.values)
            items = np.concatenate((i, i_neg), axis=None)

            pos_idx = len(i)

            df_ = pd.DataFrame({user_col: pd.Series(np.repeat(u, len(items)), dtype='int32'),
                                item_col: pd.Series(items, dtype='int32'),
                                'rating': pd.Series(np.zeros_like(items), dtype='int32')})
            df_['rating'].iloc[:pos_idx] = 1
            df_list.append(df_)

        return pd.concat(df_list, ignore_index=True)


class Interactions(Dataset, InteractionsMeta):
    def set_num_users_items(self, num_users: int, num_items: int) -> None:
        self.num_users, self.num_items = num_users, num_items

    def __init__(self, fp: Path, n_neg=100, mode='train', map_item_id_name_fp: Path = None):
        df = self.read_df(fp)

        # set_item_mapper
        if not map_item_id_name_fp or not map_item_id_name_fp.exists():
            self.map_item_id_name = self.get_item_id_name_map(df)
            with open(map_item_id_name_fp, 'w') as f:
                json.dump(self.map_item_id_name, f)
        else:
            with open(map_item_id_name_fp) as f:
                self.map_item_id_name = json.load(f, object_hook=lambda x: {int(k): v for k, v in x.items()})

        if mode == 'train':
            self.num_users, self.num_items = self.get_df_user_item_size(df)
            self.inputs: pd.DataFrame = self.negative_sample(df, n_neg)
            # df['rating'] = 1
            # self.inputs: pd.DataFrame = (df[['user_id', 'product_id', 'rating']]
            #                              .astype({'user_id': np.int32,
            #                                       'product_id': np.int32,
            #                                       'rating': np.int32})
            #                              .copy())
            logging.info(f'Samples from {df.shape} to {self.inputs.shape}')
        else:
            df['rating'] = 1
            self.inputs: pd.DataFrame = (df[['user_id', 'product_id', 'rating']]
                                         .astype({'user_id': np.int32,
                                                  'product_id': np.int32,
                                                  'rating': np.int32})
                                         .copy())
            logging.info(f'Samples from {df.shape} to {self.inputs.shape}')
            # set_num_users_items from train df
        del df

    def __len__(self):
        """Denotes the total number of rating in test set"""
        return self.inputs.shape[0]

    def __getitem__(self, index: int):
        """Generates one sample of data"""
        # get the train data
        user_id, item_id, rating = self.inputs.iloc[index].tolist()
        return {"user_id": user_id, "item_id": item_id, "rating": rating}


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_dataset = Interactions(Path('../test.csv'), mode='test')
    dl = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    for b in dl:
        logging.debug(b.keys())
        logging.debug(b)
        logging.debug([v.dtype for v in b.values()])
        break
