"""
The purpose of this helper code is to convert Goodreads ratings data into a CSV format 
and partition it into training and testing datasets. 
The testing set is specifically designed to include only those books that a user has read 
at a later time than the books in the training set. This approach is essential 
to prevent data leaks and ensure the accuracy of the book recommender system.
"""


import os
import datetime
import pandas as pd
import numpy as np

from sklearn import preprocessing

# read only part of the downloaded file due to storage limitations
# gzip -cd goodreads_interactions_fantasy_paranormal.json.gz | dd ibs=1024 count=250000 > tmp.json
# sed '$d' tmp.json > goodreads_ratings_part.json
# rm tmp.json

root = '../data'
filename = 'goodreads_ratings_{}.json'

file_train = os.path.join(root, 'book_ratings_train.csv')
file_test = os.path.join(root, 'book_ratings_test.csv')

print("Loading json files to dataframe...")

# load json file
df_list = []
for genre in ('fantasy_paranormal', 'romance'):
    file_path = os.path.join(root, filename.format(genre))
    print(file_path)
    tmp_df = pd.read_json(path_or_buf=file_path, lines=True)
    df_list.append(tmp_df)
df = pd.concat(df_list, ignore_index=True)
print(df)

# ###
print("Cleaning up dataframe...")

# cleanup: only include interactions where the book was read
df = df.loc[df['is_read']]
df = df[['user_id', 'book_id', 'rating', 'read_at', 'date_added']]
df = df.loc[df['rating'] > 0]

# cleanup: process empty read_at dates
df['datetime_read'] = df['read_at']
df.loc[df['read_at'] == '', 'datetime_read'] = df.loc[df['read_at'] == '', 'date_added']
tmp_date_splits = df['datetime_read'].str.split(' ', expand = True)
df = df.loc[tmp_date_splits.iloc[:, -1].astype(int) <= datetime.date.today().year]
tmp_date_splits = None

# cleanup: remove unnecessary columns, and convert columns to proper dtype
df = df[['user_id', 'book_id', 'rating', 'datetime_read']]
df['datetime_read'] = pd.to_datetime(df['datetime_read'], format='%a %b %d %H:%M:%S %z %Y')

# ###
print("Assigning to train and test...")

# sort by users & review date to assign into train or test data
df.sort_values(by=['user_id', 'datetime_read'], ascending=False, inplace=True)
df.reset_index(inplace=True)

# count how many reviews per user and rank based on date
counts = df['user_id'].value_counts()
df['count'] = counts.loc[df['user_id']].values
df['perc'] = df.groupby((df['user_id'] != df['user_id'].shift(1)).cumsum(), as_index=False).cumcount()+1
df['perc'] = df['perc'] / df['count']
print(df['perc'])

# assign newer reviews to test data
df['test'] = ((df['count'] <= 10) & (df['perc'] <= 0.4)) | ((df['count'] > 10) & (df['perc'] <= 0.3))

print("\nTrain/Test split before filtering:")
print(df['test'].value_counts())

# books in test that are not in train set
new_books = np.setdiff1d(df.loc[df['test']]['book_id'].unique(), df.loc[df['test'] == False]['book_id'].unique())
df = df.loc[~df['book_id'].isin(new_books)]

# ###
# create final df and save

# convert user and book ids into a value from 0 to max_count
label_encoder = preprocessing.LabelEncoder()
book_ids = label_encoder.fit_transform(df.book_id.values)
user_ids = label_encoder.fit_transform(df.user_id.values)
df['encoded_user_id'] = user_ids
df['encoded_book_id'] = book_ids
df.sort_values(['test', 'encoded_user_id'], inplace=True)

print("\nTrain/Test split after filtering:")
print(df['test'].value_counts())

print("\nRatings distribution:")
print(df['rating'].value_counts())

print("\nUnique users:", np.unique(user_ids).shape)
print("Unique books:", np.unique(df['book_id']).shape)

# save to csv
print("\nSaving files:")
df.loc[df['test'] == False, ['book_id', 'user_id', 'encoded_book_id', 'encoded_user_id', 'rating', 'datetime_read']].to_csv(file_train, sep=',', header=True, index=False)
df.loc[df['test'], ['book_id', 'user_id', 'encoded_book_id', 'encoded_user_id', 'rating', 'datetime_read']].to_csv(file_test, sep=',', header=True, index=False)
