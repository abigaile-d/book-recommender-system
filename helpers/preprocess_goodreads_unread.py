import os
import datetime
import pandas as pd
import numpy as np

from sklearn import preprocessing

# gzip -cd goodreads_interactions_fantasy_paranormal.json.gz | dd ibs=1024 count=250000 > tmp.json
# sed '$d' tmp.json > goodreads_ratings_part.json
# rm tmp.json

root = '../data'
filename = 'goodreads_ratings_{}.json'

file_ratings = os.path.join(root, 'book_ratings_{}.csv')
file_unreads = os.path.join(root, 'book_unreads_{}.csv')
file_titles = os.path.join(root, 'book_titles.csv')

print("Loading json files to dataframe...")

for mode in ('train', 'test'):
    print("Processing:", mode)
    ratings_df = pd.read_csv(file_ratings.format(mode), sep=',', header=0, quotechar='"')
    ratings_df.set_index(['encoded_user_id'], inplace=True)

    titles_df = pd.read_csv(file_titles, sep=',', header=0, quotechar='"')
    titles_df.set_index(['encoded_book_id'], inplace=True)

    for user_id in ratings_df.index.unique():
        read_books = ratings_df.loc[user_id, 'encoded_book_id']
        if not isinstance(read_books, pd.Series):
            read_books = [read_books]
        unread_books = titles_df.loc[~titles_df.index.isin(read_books)]
        unread_books = unread_books.sample(n=len(read_books))
        # print(unread_books)
        
        ratings_df.loc[user_id, ['encoded_book_id']] = unread_books.index.values
        ratings_df.loc[user_id, ['book_id']] = unread_books['book_id'].values
        ratings_df.loc[user_id, ['rating']] = 0.0

    print(ratings_df)
    print("Saving to:", file_unreads.format(mode))
    ratings_df.reset_index(inplace=True)
    ratings_df[['book_id', 'user_id', 'encoded_book_id', 'encoded_user_id', 'rating', 'datetime_read']].to_csv(file_unreads.format(mode), sep=',', header=True, index=False)