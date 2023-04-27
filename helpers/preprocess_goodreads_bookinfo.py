"""
The purpose of this helper code is to assign titles to all the books contained in the dataset.
"""

import os
import json
import pandas as pd
import numpy as np

root = '../data'
titles_filename = 'goodreads_books_{}.json'
ratings_filename = 'book_ratings_train.csv'

titles_file_path = os.path.join(root, titles_filename)
ratings_file_path = os.path.join(root, ratings_filename)
save_path = os.path.join(root, 'book_titles.csv')

# load book_ratings_train.csv to get list of all books in the dataset 
print("Loading ratings csv file to dataframe...")
ratings_df = pd.read_csv(ratings_file_path, header=0)
ratings_df = ratings_df[['book_id', 'encoded_book_id']]
ratings_df.drop_duplicates(subset=['encoded_book_id'], keep='first', inplace=True)
ratings_df.set_index('book_id', inplace=True)


print("Loading book info json file to dataframe...")

# initialize a dataframe with all book_id's from dataset as index
titles_df = pd.DataFrame(index=ratings_df.index, columns=['encoded_book_id', 'work_id', 'title'])
titles_df.sort_index(inplace=True)

# read reference json file containing book titles
# load each book that are in the dataset to the df, and discard the other books
for genre in ('fantasy_paranormal', 'romance'):
    print("Loading:", genre)
    with open(titles_file_path.format(genre)) as f:
        for line in f:
            line_json = json.loads(line)
            book_id = int(line_json['book_id'])
            if book_id in titles_df.index:
                titles_df.loc[book_id, 'work_id'] = line_json['work_id']
                titles_df.loc[book_id, 'title'] = line_json['title']
                titles_df.loc[book_id, 'encoded_book_id'] = ratings_df.loc[book_id, 'encoded_book_id']

# save to csv file
titles_df.reset_index(inplace=True)
titles_df.to_csv(save_path, sep=',', header=True, index=False)
print(titles_df)
