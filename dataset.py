import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import shutil
import glob
# from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

class HeteroGraph(HeteroData):
    def __init__(self, file, cfg, edge_index_train=None):
        super(HeteroGraph, self).__init__()  # Initialize HeteroData
        self.file = file
        self.cfg = cfg
        self.edge_index_train = edge_index_train
        self.create_df()

        

    def create_df(self):

        ratings_df = np.load(self.file)
        users_ids, items_ids = np.where(~np.isnan(ratings_df))
        ratings = ratings_df[users_ids, items_ids]
        ratings_df = pd.DataFrame(
            np.stack((users_ids, items_ids, ratings), 1),
            columns=['user_id', 'item_id', 'rating']) 
        self.num_users = len(ratings_df["user_id"].unique())
        self.num_movies = len(ratings_df["item_id"].unique())
        edge_index = torch.stack([
            torch.tensor(ratings_df['user_id'].values), 
            torch.tensor(ratings_df['item_id'].values)]
            , dim=0).long()

        assert edge_index.shape == (2, len(ratings_df)) 
        
        user_features = torch.eye(self.num_users)
        movie_features = torch.eye(self.num_movies)

        if self.cfg.input_node_embedding == "one_hot":

            self["user"].x = user_features # [num_users, num_features_users]
            self["movie"].x = movie_features # [num_users, num_features_movies]

            self['user'].x = torch.eye(self.num_users)
            self['movie'].x = torch.eye(self.num_movies)
        
        elif self.cfg.input_node_embedding == "random":

            self["user"].node_id = torch.arange(self.num_users)
            self["movie"].node_id = torch.arange(self.num_movies)
        # self["user"].x = user_features # [num_users, num_features_users]
        # self["movie"].x = movie_features # [num_users, num_features_movies]

        # self['user'].x = torch.eye(self.num_users)
        # self['movie'].x = torch.eye(self.num_movies)
        # num_features_users = 1  # or any desired dimension
        # num_features_movies = 1  # or any desired dimension
        # self['user'].x = torch.ones(self.num_users, num_features_users)
        # self['movie'].x = torch.ones(self.num_movies, num_features_movies)


        # Add the rating edges:
        if self.__dict__.get('edgex_index_train') is not None:
            self['user', 'rates', 'movie'].edge_index = self.edge_index_train # [2, num_ratings_train]
        else:
            self['user', 'rates', 'movie'].edge_index = edge_index  # [2, num_ratings]

        # self['user', 'rates', 'movie'].edge_attr =  edge_attr 
        

        # Add the rating labels:
        rating = torch.from_numpy(ratings_df['rating'].values).to(torch.float)
        self['user', 'rates', 'movie'].edge_label = rating  # [num_ratings]

        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        self = T.ToUndirected()(self)

        # With the above transformation we also got reversed labels for the edges.
        # We are going to remove them:
        del self['movie', 'rev_rates', 'user'].edge_label

                
        assert self['user'].num_nodes == self.num_users
        assert self['user', 'rates', 'movie'].num_edges == len(ratings_df)
        # assert data['movie'].num_features == 404

        self['user', 'movie'].edge_label_index = self['user', 'movie'].edge_index
        # self, _, _ = T.RandomLinkSplit(
        #     num_val=0,
        #     num_test=0,
        #     neg_sampling_ratio=0.0,
        #     edge_types=[('user', 'rates', 'movie')],
        #     rev_edge_types=[('movie', 'rev_rates', 'user')],
        #     )(self)
    

        

