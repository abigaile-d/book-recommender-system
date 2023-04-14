import os
import pandas as pd
import numpy as np

root = '../data'
titles_filename = 'goodreads_book_works.json'
ratings_filename = 'book_ratings_train.csv'

titles_file_path = os.path.join(root, titles_filename)
ratings_file_path = os.path.join(root, ratings_filename)
save_path = os.path.join(root, 'book_titles.csv')


print("Loading json file to dataframe...")

# load json file
titles_df = pd.read_json(path_or_buf=titles_file_path, lines=True)
titles_df = titles_df[['best_book_id', 'original_title']]
titles_df.columns = ['book_id', 'original_title']

ratings_df = pd.read_csv(ratings_file_path, header=0)
ratings_df = ratings_df[['book_id', 'encoded_book_id']]
ratings_df.drop_duplicates(subset=['book_id'], inplace=True)

merged_df = ratings_df.merge(titles_df, how='left')
merged_df.to_csv(save_path, sep=',', header=True, index=False)
print(merged_df)
