import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=50):
        super().__init__()
        # User and Movie latent vectors
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.movie_factors = nn.Embedding(n_movies, n_factors)
        
        # Biases to account for systematic variations
        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights (Xavier/Glorot is standard for embeddings)
        nn.init.xavier_uniform_(self.user_factors.weight)
        nn.init.xavier_uniform_(self.movie_factors.weight)

    def forward(self, user, movie):
        # Dot product of latent vectors
        dot = (self.user_factors(user) * self.movie_factors(movie)).sum(dim=1)
        # Add biases
        res = dot + self.user_bias(user).squeeze() + self.movie_bias(movie).squeeze() + self.global_bias
        return res