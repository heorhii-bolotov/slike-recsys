import torch
from torch import nn
import torch.nn.functional as F

from logger import create_logger

logging = create_logger()


class MLP(nn.Module):
    def __init__(self, n_users, n_items, layers=(16, 8), dropout=False, device='cpu', **kwargs):
        """
        Simple Feedforward network with Embeddings for users and items
        """
        super().__init__()
        assert layers[0] % 2 == 0, 'layers[0] must be an even number'
        self.__alias__ = f'MLP {layers}'
        self.__dropout__ = dropout

        # user and item embedding layers
        embedding_dim = int(layers[0] / 2)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # list of weight matrices
        self.fc_layers = nn.ModuleList()
        # hidden dense layers
        for _, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))
        # final prediction layer
        self.output_layer = nn.Linear(layers[-1], 1)
        self.device = device

        self._init()

        logging.debug(n_users)
        logging.debug(n_items)

    def forward(self, feed_dict):
        users = feed_dict['user_id']
        items = feed_dict['item_id']
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        # concatenate user and item embeddings to form input
        x = torch.cat([user_embedding, item_embedding], 1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            x = self.fc_layers[idx](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.__dropout__, training=self.training)
        logit = self.output_layer(x)
        rating = torch.sigmoid(logit)
        return rating

    def predict(self, feed_dict):
        # return the score, inputs and outputs are numpy arrays
        feed_dict = {k: v.to(dtype=torch.long, device=self.device)
                     for k, v in feed_dict.items()
                     if v is not None}
        output_scores = self(feed_dict)
        return output_scores.cpu().detach().numpy()

    def get_alias(self):
        return self.__alias__

    def _init(self):
        """
        Setup embeddings and hidden layers with reasonable initial values.
        """

        def init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.user_embedding.weight.data.uniform_(-0.05, 0.05)
        self.item_embedding.weight.data.uniform_(-0.05, 0.05)
        self.fc_layers.apply(init)
        init(self.output_layer)
