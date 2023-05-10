"""
The purpose of this helper code is to create a random sample list of books 
that each user has not yet read. 
This list of books is then added to the dataset with a rating of 0.
"""

import os
import datetime
import pandas as pd
import numpy as np

from sklearn import preprocessing


root = '../data'
filename = 'goodreads_ratings_{}.json'

file_ratings = os.path.join(root, 'book_ratings_{}.csv')
file_unreads = os.path.join(root, 'book_unreads_{}.csv')
file_titles = os.path.join(root, 'book_titles.csv')
multiplier = 1

print("Loading ratings csv to dataframe...")
df_list = []
for mode in ('train', 'test'):
    print("Processing:", mode)
    # load list of books that the user has already read
    tmp_df = pd.read_csv(file_ratings.format(mode), sep=',', header=0, quotechar='"')
    tmp_df['mode'] = mode
    df_list.append(tmp_df)
all_ratings_df = pd.concat(df_list, ignore_index=True)
all_ratings_df.set_index(['encoded_user_id'], inplace=True)
df_list = None
print(all_ratings_df)

# load list of all books
titles_df = pd.read_csv(file_titles, sep=',', header=0, quotechar='"')
titles_df.set_index(['encoded_book_id'], inplace=True)

print("Creating list of unread books...")
for mode in ('train', 'test'):
    print("Processing:", mode)

    # load list of books that the user has already read
    # ratings_df = pd.read_csv(file_ratings.format(mode), sep=',', header=0, quotechar='"')
    # ratings_df.set_index(['encoded_user_id'], inplace=True)
    ratings_df = all_ratings_df.loc[all_ratings_df['mode'] == mode]

    if mode == 'test':
        unreads_df_train = unreads_df[['encoded_user_id', 'encoded_book_id']].copy()
        unreads_df_train.set_index('encoded_user_id', inplace=True)

    unreads_df = ratings_df.copy()
    unreads_df[['book_id', 'encoded_book_id', 'rating']] = 0
    unreads_df['datetime_read'] = None
    if multiplier > 1:
        unreads_df = pd.concat([unreads_df]*multiplier)
    unreads_df.sort_index(inplace=True)
    print(ratings_df)
    print(unreads_df)
    
    # find list of all books that the user has not read, 
    # and get a random sample with size the same as the books that the user has already read
    for user_id in ratings_df.index.unique():
        # if train, get list of previously read books from train.csv
        if mode == 'train':
            read_books = ratings_df.loc[user_id, 'encoded_book_id']
            if not isinstance(read_books, pd.Series):
                read_books = [read_books]
            sample_size = len(read_books)
        # if test, get list of all previous read books, and all unread books not included in training
        else:
            read_books = all_ratings_df.loc[user_id, 'encoded_book_id'].tolist()
            if isinstance(read_books, int):
                read_books = [read_books]
            unread_books_train = unreads_df_train.loc[user_id, 'encoded_book_id'].tolist()
            if isinstance(unread_books_train, int):
                unread_books_train = [unread_books_train]
            sample_size = len(read_books) - int(len(unread_books_train)/multiplier)
            read_books.extend(unread_books_train)

        unread_books = titles_df.loc[~titles_df.index.isin(read_books)]
        # unread_books_pop = unread_books.sample(n=int(sample_size*multiplier*0.35), weights=unread_books['count'])
        unread_books = unread_books.sample(n=int(sample_size*multiplier))
        # unread_books = pd.concat([unread_books_pop, unread_books_non]).drop_duplicates()
        # unread_books = unread_books.sample(n=sample_size*multiplier)
        
        unreads_df.loc[user_id, ['encoded_book_id']] = unread_books.index.values
        unreads_df.loc[user_id, ['book_id']] = unread_books['book_id'].values

    # save to csv
    print(unreads_df)
    print("Saving to:", file_unreads.format(mode))
    unreads_df.reset_index(inplace=True)
    unreads_df[['book_id', 'user_id', 'encoded_book_id', 'encoded_user_id', 'rating', 'datetime_read']].to_csv(file_unreads.format(mode), sep=',', header=True, index=False)