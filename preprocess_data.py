import pandas as pd

class MovieLensLoader:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, sep=',', engine='python')
        # ratings_np = self.df['rating'].values.astype('float32')
        # users_np = self.df['userId'].values.astype('int64')
        # movies_np = self.df['movieId'].values.astype('int64')
        self.user_map = {id: i for i, id in enumerate(self.df.userId.unique())}
        self.movie_map = {id: i for i, id in enumerate(self.df.movieId.unique())}
        self.df['user_idx'] = self.df['userId'].map(self.user_map)
        self.df['movie_idx'] = self.df['movieId'].map(self.movie_map)