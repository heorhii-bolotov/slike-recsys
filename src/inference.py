import pandas as pd

from pl_modules import NCF
from trainer import parse_args, print_name_value_type, update_namespace, get_device

from logger import create_logger

logging = create_logger()


def predict():
    pass


if __name__ == '__main__':
    args = parse_args()
    print_name_value_type(args)  # debug

    device = get_device()
    logging.debug(f"Device available: {device}")

    from pathlib import Path

    ckpt_fp = '2021-05-15/2021-05-15.ckpt'
    ncf = NCF.load_from_checkpoint(checkpoint_path=ckpt_fp).to(device)
    ncf.eval()
    ncf.freeze()

    import torch
    from torch.utils.data import DataLoader
    from datasets import Interactions
    import numpy as np

    test_dataset = Interactions(Path('../test.csv'), mode='test',
                                map_item_id_name_fp=getattr(args, 'map_item_id_name_fp'))

    dl = DataLoader(test_dataset, batch_size=5000, shuffle=False, num_workers=4)
    res = {
        'user_id': [], 'logits': [], 'item_id': [], 'item_name': [],
    }
    for feed_dict in dl:
        feed_dict = {k: v.to(dtype=torch.long, device=device)
                     for k, v in feed_dict.items()
                     if v is not None}

        res['user_id'].append(feed_dict['user_id'].cpu().detach().numpy())
        res['logits'].append(ncf(feed_dict).cpu().detach().numpy().ravel())
        res['item_id'].append(feed_dict['item_id'].cpu().detach().numpy())

    res['user_id'] = np.concatenate(res['user_id'])
    res['logits'] = np.concatenate(res['logits'])
    res['item_id'] = np.concatenate(res['item_id'])
    res['item_name'] = np.vectorize(test_dataset.map_item_id_name.get)(res['item_id'])

    res = pd.DataFrame(res)
    logging.info(res.head())
    logging.debug(res[['logits', 'item_name']].columns)
    for u, df in res.groupby('user_id'):
        df = df.nlargest(20, 'logits')
        logging.debug(df[['logits', 'item_name']].values)
