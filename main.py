import yaml
import torch
from cProfile import Profile
import torch.nn.functional as F
from dataset import HeteroGraph
from model import Model
import numpy as np


from torch.utils.tensorboard import SummaryWriter
import os

class Config(dict):
    def __init__(self, config):
        self._conf = config
 
    def __getattr__(self, name):
        if self._conf.get(name) is not None:
            return self._conf[name]
        return None
    
# def graph_to_tensor(x_dict, edge_label_index, estimates):
#     size = x_dict['user'].shape[0], x_dict['movie'].shape[0]
#     result = torch.zeros(*size).to(edge_label_index.device)
#     result[*edge_label_index] = estimates
#     return result

def round_tensor(matrix, delta=0):
    rounded_matrix = torch.round(matrix * 2) / 2  # Round to the nearest half
    mask = torch.abs(matrix - rounded_matrix) < delta
    rounded_matrix[mask] = torch.round(matrix[mask])  # Round to the nearest whole number when within delta
    return rounded_matrix
  
def main(cfg, table=None):
    cfg = Config(cfg)
    writer = SummaryWriter() #log_dir='runs/lr_'+str(cfg.lr)+'_in_'+str(cfg.in_channels)+'_out_'+str(cfg.out_channels)+'_epochs_'+str(cfg.epochs)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    if device == 'cuda:0':
        torch.cuda.set_device(device)
    if cfg.mode == 'train':
        train_data = HeteroGraph(cfg.train_path, cfg)
        #print(train_data.x_dict)
        #print(train_data.edge_index_dict)
        #print(train_data["user"].num_nodes)
        test_data = HeteroGraph(cfg.test_path, cfg, train_data['user', 'rates', 'movie'].edge_index)
    elif cfg.mode == 'prof':
        assert table is not None
        train_data = HeteroGraph(table, cfg)
    
    model = Model(cfg, data=train_data).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def predict_all_link(data, model):

        all_user_ids = torch.arange(data.num_users)
        all_movie_ids = torch.arange(data.num_movies)
        
        # Create a mesh grid for all user-movie pairs
        user_ids, movie_ids = torch.meshgrid(all_user_ids, all_movie_ids, indexing='xy')
        #print(user_ids.shape, movie_ids.shape)

        # Stack them together to create the edge_label_index
        edge_label_index = torch.stack((user_ids.reshape(-1), movie_ids.reshape(-1)), dim=0)
        #print("data", data.x_dict)
        data = data.to(device)
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict,
                    edge_label_index)
        ratings_matrix = pred.view(data.num_users, data.num_movies)
        return ratings_matrix
    
    # def RMSE_ratings(R, R_hat):
    #         nz_mask = R > 0
    #         T = len(R[nz_mask])
    #         return np.sqrt(np.sum((R[nz_mask] - R_hat[nz_mask])**2)/T)

    # def test_(R_hat):
    #         print("test")
    #         R = np.load(cfg.train_path)
    #         print("Test ",RMSE_ratings(R, R_hat))

    def train():
        model.train()
        optimizer.zero_grad()
        #print("ee", train_data.node_id_dict.keys())
        if cfg.input_node_embedding == "random":
            pred = model(train_data.node_id_dict, train_data.edge_index_dict,
                        train_data['user', 'movie'].edge_label_index,
                        # train_data.edge_types
                        )
        elif cfg.input_node_embedding == "one_hot":
            pred = model(train_data.x_dict, train_data.edge_index_dict,
                        train_data['user', 'movie'].edge_label_index,
                        # train_data.edge_types
                        )

        target = 2*train_data['user', 'movie'].edge_label -1
        target = target.long()
        #print(target, target.shape, target.dtype)
        #print(pred, pred.shape, pred.dtype)
        loss = F.nll_loss(pred, target)
        # loss = F.mse_loss(pred, target)
        loss.backward()
        #print(loss)
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(data):
        data = data.to(device)
        model.eval()

        if cfg.input_node_embedding == "random":
            pred = model(data.node_id_dict, data.edge_index_dict,
                    data['user', 'movie'].edge_label_index)

        elif cfg.input_node_embedding == "one_hot":
            pred = model(data.x_dict, data.edge_index_dict,
                    data['user', 'movie'].edge_label_index)
        # pred = pred.clamp(min=0, max=5)
        target = 2*data['user', 'movie'].edge_label - 1
        target = target.long()
        # loss = F.nll_loss(pred, target)
        loss = F.mse_loss(pred, target).sqrt()
        return float(loss)


    for epoch in range(1, cfg.epochs+1):
        train_data = train_data.to(device)
        loss = train()
        train_rmse = test(train_data)
        if cfg.mode == 'train':
            val_rmse = test(test_data)
            writer.add_scalars("rmse", {"train": train_rmse, "val": val_rmse}, epoch)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
            f'Val: {val_rmse:.4f}')    
        elif cfg.mode == 'prof':
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}')
 
        
    print("Training ended.")

    if cfg.save:
        torch.save(model.state_dict(), 'model.pt')
        print('Model saved.')

    matrix_pred = predict_all_link(train_data, model)
    if cfg.round_output:
        matrix_pred = round_tensor(matrix_pred, cfg.round_delta)

    return matrix_pred
        
if __name__ == '__main__':
    with open('config.yml') as f:
        cfg = yaml.safe_load(f)
    table = main(cfg)
    print(len(table[table != 0]), table.shape)