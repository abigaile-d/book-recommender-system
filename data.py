import os
import pandas as pd
import numpy as np
from gdown import download

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets.utils import extract_archive

class GoodReadsRatingsDataset(TensorDataset):
    download_url = 'https://drive.google.com/uc?id=116G-epHLGoHxbuDRpwc5nh7IUvuf4cye'
    zip_filename = 'GoodReadsRatingsPart.tgz'
    filename = dict()
    filename['train'] = 'book_{}_train.csv'  # ratings and unread
    filename['test'] = 'book_{}_test.csv'  # ratings and unread

    def __init__(self, root, mode='train', explicit=True):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.file_path = file_path = os.path.join(root, self.filename[mode])
        self._download_data()

        # load data
        tmp_df1 = pd.read_csv(file_path.format('ratings'), sep=',', header=0, quotechar='"')
        tmp_df2 = pd.read_csv(file_path.format('unreads'), sep=',', header=0, quotechar='"')
        df = pd.concat([tmp_df1, tmp_df2], ignore_index=True)
        tmp_df1, tmp_df2 = None, None
        df.sort_values(['encoded_user_id', 'encoded_book_id'], inplace=True)

        self.user_ids = df.encoded_user_id.values
        
        user_ids = torch.LongTensor(df.encoded_user_id.values)
        item_ids = torch.LongTensor(df.encoded_book_id.values)
        if explicit:
            ratings = torch.FloatTensor(df.rating.values / 5.0)
        else:
            df.loc[df.rating < 3, 'rating'] = 0
            df.loc[df.rating >= 3, 'rating'] = 1
            ratings = torch.FloatTensor(df.rating.values)

        self.n_users = user_ids.unique().shape[0]
        self.n_items = item_ids.unique().shape[0]
        self.n_ratings = ratings.shape[0]
        print("Dataset: {}, User count: {}, Book count: {}, Ratings count: {}".format(mode, self.n_users, self.n_items, self.n_ratings))

        super(GoodReadsRatingsDataset, self).__init__(user_ids, item_ids, ratings)
    

    def _download_data(self):
        if os.path.isfile(self.file_path.format('ratings')):
            return

        download(self.download_url, os.path.join(self.root, self.zip_filename), quiet=False)
        extract_archive(os.path.join(self.root, self.zip_filename))


    def get_user_record(self, random=True, user_id=0):
        if random:
            curr_user = np.random.choice(self.user_ids, 1)[0]
        else:
            curr_user = user_id
        curr_user_indices = np.where(self.user_ids == curr_user)[0]

        return curr_user_indices
