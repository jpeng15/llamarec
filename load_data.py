import pickle

import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def remove_immediate_repeats(self, df):
        df_next = df.shift()
        is_not_repeat = (df['uid'] != df_next['uid']) | (df['sid'] != df_next['sid'])
        df = df[is_not_repeat]
        return df

    def densify_index(self, df):
        umap = {u: i for i, u in enumerate(set(df['uid']), start=1)}
        smap = {s: i for i, s in enumerate(set(df['sid']), start=1)}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def filter_triplets(self, df):
        # print('Filtering triplets')
        # if self.min_sc > 0:
        #     item_sizes = df.groupby('sid').size()
        #     good_items = item_sizes.index[item_sizes >= self.min_sc]
        #     df = df[df['sid'].isin(good_items)]

        # if self.min_uc > 0:
        #     user_sizes = df.groupby('uid').size()
        #     good_users = user_sizes.index[user_sizes >= self.min_uc]
        #     df = df[df['uid'].isin(good_users)]
        return df

    def split_df(self, df, user_count):
        user_group = df.groupby('uid')
        user2items = user_group.progress_apply(
            lambda d: list(d.sort_values(by=['timestamp', 'sid'])['sid']))
        train, val, test = {}, {}, {}
        for i in range(user_count):
            user = i + 1
            items = user2items[user]
            train[user], val[user], test[user] = items[:-2], items[-2:-1], items[-1:]
        return train, val, test


    def load_ratings(self):
        ratings = pd.read_csv(self.data_path, sep='::', engine='python', 
                              names=['uid', 'sid', 'rating', 'timestamp'])
        return ratings
    
    def preprocess(self):
        df = self.load_ratings()
        df = self.remove_immediate_repeats(df)
        df = self.filter_triplets(df)
        df, umap, smap = self.densify_index(df)

        train, val, test = self.split_df(df, len(umap))

        dataset = {
          'train': train,
          'val': val,
          'test': test,
          'umap': umap,
          'smap': smap,
        }

        with open('movielens-preprocessed', 'wb') as f:
          pickle.dump(dataset, f)