import os
import datetime
import pandas as pd
import numpy as np

from sklearn import preprocessing

# gzip -cd goodreads_interactions_fantasy_paranormal.json.gz | dd ibs=1024 count=1000000 > tmp.json
# sed '$d' tmp.json > goodreads_ratings_part.json
# rm tmp.json

root = '../data'
filename = 'goodreads_ratings_part.json'

file_path = os.path.join(root, filename)
file_train = os.path.join(root, 'book_ratings_train.csv')
file_test = os.path.join(root, 'book_ratings_test.csv')

print("Loading json file to dataframe...")

# load json file
df = pd.read_json(path_or_buf=file_path, lines=True)

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
print("Assigning to train and test...")

# sort by users & review date to assign into train or test data
df.sort_values(by=['user_id', 'datetime_read'], ascending=False, inplace=True)
df.reset_index(inplace=True)

# count how many reviews per user and rank based on date
counts = df['user_id'].value_counts()
df['count'] = counts.loc[df['user_id']].values
df['perc'] = df.groupby((df['user_id'] != df['user_id'].shift(1)).cumsum(), as_index=False).cumcount()+1
df['perc'] = df['perc'] / df['count']

# assign newer reviews to test data
df['test'] = (df['perc'] < 0.34) | ((df['count'] > 10) & (df['perc'] < 0.2))
print(df.head(15))
print(df.tail(15))

print("Train/Test split:")
print(df['test'].value_counts())

# books in test that are not in train set
new_books = np.setdiff1d(df.loc[df['test']]['book_id'].unique(), df.loc[df['test'] == False]['book_id'].unique())
df = df.loc[~df['book_id'].isin(new_books)]

# convert user and book ids into a value from 0 to max_count
label_encoder = preprocessing.LabelEncoder()
book_ids = label_encoder.fit_transform(df.book_id.values)
user_ids = label_encoder.fit_transform(df.user_id.values)
df['encoded_user_id'] = user_ids
df['encoded_book_id'] = book_ids

print("Train/Test split:")
print(df['test'].value_counts())
print("Ratings distribution:")
print(df['rating'].value_counts())

print("Saving files:")
df.loc[df['test'] == False, ['book_id', 'user_id', 'encoded_book_id', 'encoded_user_id', 'rating', 'datetime_read']].to_csv(file_train, sep=',', header=True, index=False)
df.loc[df['test'], ['book_id', 'user_id', 'encoded_book_id', 'encoded_user_id', 'rating', 'datetime_read']].to_csv(file_test, sep=',', header=True, index=False)
