import os
import pandas as pd
from gdown import download

import torch
from torch.utils.data import TensorDataset
from torchvision.datasets.utils import extract_archive

class GoodReadsRatingsDataset(TensorDataset):
    download_url = 'https://drive.google.com/uc?id=1zPQU1LuQ_6qmoNIvEpXfMJ-W8jh6S6WQ'
    zip_filename = 'GoodReadsRatingsPart.tgz'
    filename = dict()
    filename['train'] = 'book_ratings_train.csv'
    filename['test'] = 'book_ratings_test.csv'

    def __init__(self, root, mode='train'):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

        self.file_path = file_path = os.path.join(root, self.filename[mode])
        self._download_data()

        # load data
        df = pd.read_csv(file_path, sep=',', header=0, quotechar='"')
        # df['rating'] = df['rating'].astype(float)
        # df['rating'] = (df['rating'] - 1.0) / 4.0

        user_ids = torch.LongTensor(df.encoded_user_id.values)
        item_ids = torch.LongTensor(df.encoded_book_id.values)
        ratings = torch.FloatTensor(df.rating.values)

        self.n_users = user_ids.unique().shape[0]
        self.n_items = item_ids.unique().shape[0]
        self.n_ratings = ratings.shape[0]
        print("Dataset: {}, User count: {}, Book count: {}, Ratings count: {}".format(mode, self.n_users, self.n_items, self.n_ratings))

        super(GoodReadsRatingsDataset, self).__init__(user_ids, item_ids, ratings)
    

    def _download_data(self):
        if os.path.isfile(self.file_path):
            return

        download(self.download_url, os.path.join(self.root, self.zip_filename), quiet=False)
        extract_archive(os.path.join(self.root, self.zip_filename))
