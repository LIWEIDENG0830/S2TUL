import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, RGCNConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# class BiLSTMConcatGlobalTUL(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.n_all_trajs = args.n_all_trajs
#         self.hidden_size = args.hidden_size
#         self.loc_embedding = nn.Embedding(args.n_vocabs, args.hidden_size)
#         self.traj_embedding = nn.Embedding(args.n_all_trajs, args.hidden_size)
#         # self.global_graph_conv_one = GATConv(args.hidden_size*2, args.hidden_size, add_self_loops=False)
#         # self.global_graph_conv_two = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
#         self.lstm = nn.LSTM(args.hidden_size, args.hidden_size, batch_first=True)
#         self.global_graph_conv_one = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
#         self.global_graph_conv_two = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
#         self.dropout_layer = nn.Dropout(p=args.dropout)
#         self.predictor = nn.Sequential(
#             # nn.Linear(args.hidden_size, args.hidden_size),
#             # nn.ReLU(),
#             nn.Linear(args.hidden_size*2, args.n_users)
#         )
#
#     def forward(self, padded_trajs, trajs_len, TrajTrajGraph, idxes):
#         padded_trajs = self.loc_embedding(padded_trajs)
#         pack_padded_trajs = pack_padded_sequence(padded_trajs, trajs_len, batch_first=True, enforce_sorted=False)
#         _, (seq_output, _) = self.lstm(pack_padded_trajs)
#         seq_output = seq_output.transpose(0,1).reshape(padded_trajs.shape[0], -1)
#         # Traj_emb_zero = self.dropout_layer(Traj_emb_zero)
#         # GCN
#         # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
#         # Traj_emb_one = self.dropout_layer(Traj_emb_one)
#         # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
#         # Traj_emb_two = self.dropout_layer(Traj_emb_two)
#         # GAT
#         Traj_emb_zero = self.traj_embedding.weight
#         Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
#         # Traj_emb_one = self.dropout_layer(Traj_emb_one)
#         Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0])
#         # concat
#         concat_output = torch.cat([Traj_emb_two[idxes], seq_output[idxes]], dim=-1)
#         Traj_emb_two = self.dropout_layer(Traj_emb_two)
#         # Traj_emb_two = torch.cat([Traj_emb_one, Traj_emb_two], dim=-1)
#         predictions = self.predictor(concat_output)
#         return predictions


# class BiLSTMConcatGlobalTUL(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.n_all_trajs = args.n_all_trajs
#         self.hidden_size = args.hidden_size
#         self.loc_embedding = nn.Embedding(args.n_vocabs, args.hidden_size)
#         self.traj_embedding = nn.Embedding(args.n_all_trajs, args.hidden_size)
#         # self.global_graph_conv_one = GATConv(args.hidden_size*2, args.hidden_size, add_self_loops=False)
#         # self.global_graph_conv_two = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
#         self.lstm = nn.LSTM(args.hidden_size, args.hidden_size, batch_first=True)
#         # self.global_graph_conv_one = GCNConv(args.hidden_size * 2, args.hidden_size, add_self_loops=False)
#         # self.global_graph_conv_two = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
#         self.global_graph_conv_one = RGCNConv(args.hidden_size * 2, args.hidden_size, num_relations=2)
#         self.global_graph_conv_two = RGCNConv(args.hidden_size, args.hidden_size, num_relations=2)
#         self.dropout_layer = nn.Dropout(p=args.dropout)
#         self.predictor = nn.Sequential(
#             # nn.Linear(args.hidden_size, args.hidden_size),
#             # nn.ReLU(),
#             nn.Linear(args.hidden_size, args.n_users)
#         )
#
#     def forward(self, padded_trajs, trajs_len, TrajTrajGraph):
#         padded_trajs = self.loc_embedding(padded_trajs)
#         pack_padded_trajs = pack_padded_sequence(padded_trajs, trajs_len, batch_first=True, enforce_sorted=False)
#         _, (seq_output, _) = self.lstm(pack_padded_trajs)
#         seq_output = seq_output.transpose(0,1).reshape(padded_trajs.shape[0], -1)
#         # Traj_emb_zero = self.dropout_layer(Traj_emb_zero)
#         # GCN
#         # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
#         # Traj_emb_one = self.dropout_layer(Traj_emb_one)
#         # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
#         # Traj_emb_two = self.dropout_layer(Traj_emb_two)
#         # GAT
#         Traj_emb_zero = torch.cat([self.traj_embedding.weight, seq_output], dim=-1)
#         Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
#         # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
#         # Traj_emb_one = self.dropout_layer(Traj_emb_one)
#         Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
#         # concat
#         Traj_emb_two = self.dropout_layer(Traj_emb_two)
#         # concat_output = torch.cat([Traj_emb_two[idxes], seq_output[idxes]], dim=-1)
#         # Traj_emb_two = self.dropout_layer(Traj_emb_two)
#         # Traj_emb_two = torch.cat([Traj_emb_one, Traj_emb_two], dim=-1)
#         predictions = self.predictor(Traj_emb_two)
#         return predictions


class BiLSTMConcatGlobalTUL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_all_trajs = args.n_all_trajs
        self.hidden_size = args.hidden_size
        self.loc_embedding = nn.Embedding(args.n_vocabs, args.hidden_size)
        self.traj_embedding = nn.Embedding(args.n_all_trajs, args.hidden_size)
        # self.global_graph_conv_one = GATConv(args.hidden_size*2, args.hidden_size, add_self_loops=False)
        # self.global_graph_conv_two = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.lstm = nn.LSTM(args.hidden_size, args.hidden_size, batch_first=True)
        # self.global_graph_conv_one = GCNConv(args.hidden_size * 2, args.hidden_size, add_self_loops=False)
        # self.global_graph_conv_two = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.global_graph_conv_one = RGCNConv(args.hidden_size, args.hidden_size, num_relations=2)
        self.global_graph_conv_two = RGCNConv(args.hidden_size, args.hidden_size, num_relations=2)
        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.predictor = nn.Sequential(
            # nn.Linear(args.hidden_size, args.hidden_size),
            # nn.ReLU(),
            nn.Linear(args.hidden_size*2, args.n_users)
        )

    def forward(self, padded_trajs, trajs_len, TrajTrajGraph):
        padded_trajs = self.loc_embedding(padded_trajs)
        pack_padded_trajs = pack_padded_sequence(padded_trajs, trajs_len, batch_first=True, enforce_sorted=False)
        _, (seq_output, _) = self.lstm(pack_padded_trajs)
        seq_output = seq_output.transpose(0,1).reshape(padded_trajs.shape[0], -1)
        # Traj_emb_zero = self.dropout_layer(Traj_emb_zero)
        # GCN
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
        # Traj_emb_one = self.dropout_layer(Traj_emb_one)
        # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
        # Traj_emb_two = self.dropout_layer(Traj_emb_two)
        # GAT
        Traj_emb_zero = self.traj_embedding.weight
        Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
        # Traj_emb_one = self.dropout_layer(Traj_emb_one)
        Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
        # concat
        Traj_emb_two = self.dropout_layer(Traj_emb_two)
        # concat_output = torch.cat([Traj_emb_two[idxes], seq_output[idxes]], dim=-1)
        Traj_emb_two = torch.cat([Traj_emb_two, seq_output], dim=-1)
        # Traj_emb_two = self.dropout_layer(Traj_emb_two)
        # Traj_emb_two = torch.cat([Traj_emb_one, Traj_emb_two], dim=-1)
        predictions = self.predictor(Traj_emb_two)
        return predictions


class BiLSTMGlobalTUL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_all_trajs = args.n_all_trajs
        self.hidden_size = args.hidden_size
        self.embedding_layer = nn.Embedding(args.n_vocabs, args.hidden_size)
        # self.global_graph_conv_one = GATConv(args.hidden_size*2, args.hidden_size, add_self_loops=False)
        # self.global_graph_conv_two = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.lstm = nn.LSTM(args.hidden_size, args.hidden_size, batch_first=True, bidirectional=False)
        # self.global_graph_conv_one = GCNConv(args.hidden_size*2, args.hidden_size, add_self_loops=False)
        # self.global_graph_conv_two = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.global_graph_conv_one = RGCNConv(args.hidden_size, args.hidden_size, num_relations=2)
        self.global_graph_conv_two = RGCNConv(args.hidden_size, args.hidden_size, num_relations=2)
        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.predictor = nn.Sequential(
            # nn.Linear(args.hidden_size, args.hidden_size),
            # nn.ReLU(),
            nn.Linear(args.hidden_size, args.n_users)
        )

    def forward(self, padded_trajs, trajs_len, TrajTrajGraph):
        padded_trajs = self.embedding_layer(padded_trajs)
        pack_padded_trajs = pack_padded_sequence(padded_trajs, trajs_len, batch_first=True, enforce_sorted=False)
        _, (Traj_emb_zero, _) = self.lstm(pack_padded_trajs)
        Traj_emb_zero = Traj_emb_zero.transpose(0,1).reshape(padded_trajs.shape[0], -1)
        # Traj_emb_zero = self.dropout_layer(Traj_emb_zero)
        # GCN
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
        # Traj_emb_one = self.dropout_layer(Traj_emb_one)
        # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
        # Traj_emb_two = self.dropout_layer(Traj_emb_two)
        # GAT
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
        # # Traj_emb_one = self.dropout_layer(Traj_emb_one)
        # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0])
        Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
        # Traj_emb_one = self.dropout_layer(Traj_emb_one)
        Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
        Traj_emb_two = self.dropout_layer(Traj_emb_two)
        # Traj_emb_two = torch.cat([Traj_emb_one, Traj_emb_two], dim=-1)
        predictions = self.predictor(Traj_emb_two)
        return predictions


class LSTMGlobalTUL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_all_trajs = args.n_all_trajs
        self.hidden_size = args.hidden_size
        self.embedding_layer = nn.Embedding(args.n_vocabs, args.hidden_size)
        # self.global_graph_conv_one = GATConv(args.hidden_size*2, args.hidden_size, add_self_loops=False)
        # self.global_graph_conv_two = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.lstm = nn.LSTM(args.hidden_size, args.hidden_size, batch_first=True, bidirectional=False)
        # self.global_graph_conv_one = GCNConv(args.hidden_size*2, args.hidden_size, add_self_loops=False)
        # self.global_graph_conv_two = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.global_graph_conv_one = RGCNConv(args.hidden_size, args.hidden_size, num_relations=2)
        self.global_graph_conv_two = RGCNConv(args.hidden_size, args.hidden_size, num_relations=2)
        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.predictor = nn.Sequential(
            # nn.Linear(args.hidden_size, args.hidden_size),
            # nn.ReLU(),
            nn.Linear(args.hidden_size, args.n_users)
        )

    def forward(self, padded_trajs, trajs_len, TrajTrajGraph):
        padded_trajs = self.embedding_layer(padded_trajs)
        pack_padded_trajs = pack_padded_sequence(padded_trajs, trajs_len, batch_first=True, enforce_sorted=False)
        _, (Traj_emb_zero, _) = self.lstm(pack_padded_trajs)
        Traj_emb_zero = Traj_emb_zero.transpose(0,1).reshape(padded_trajs.shape[0], -1)
        # Traj_emb_zero = self.dropout_layer(Traj_emb_zero)
        # GCN
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
        # Traj_emb_one = self.dropout_layer(Traj_emb_one)
        # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
        # Traj_emb_two = self.dropout_layer(Traj_emb_two)
        # GAT
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
        # # Traj_emb_one = self.dropout_layer(Traj_emb_one)
        # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0])
        Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
        # Traj_emb_one = self.dropout_layer(Traj_emb_one)
        Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
        # Traj_emb_two = self.dropout_layer(Traj_emb_two)
        # Traj_emb_two = torch.cat([Traj_emb_one, Traj_emb_two], dim=-1)
        predictions = self.predictor(Traj_emb_two)
        return predictions



# class LSTMGlobalTUL(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.n_all_trajs = args.n_all_trajs
#         self.hidden_size = args.hidden_size
#         self.embedding_layer = nn.Embedding(args.n_vocabs, args.hidden_size)
#         self.global_graph_conv_one = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
#         self.global_graph_conv_two = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
#         self.lstm = nn.LSTM(args.hidden_size, args.hidden_size, batch_first=True)
#         # self.global_graph_conv_one = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
#         # self.global_graph_conv_two = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
#         self.dropout_layer = nn.Dropout(p=args.dropout)
#         self.predictor = nn.Sequential(
#             # nn.Linear(args.hidden_size*3, args.hidden_size),
#             # nn.ReLU(),
#             nn.Linear(args.hidden_size*3, args.n_users)
#         )
#
#     def forward(self, padded_trajs, trajs_len, TrajTrajGraph):
#         padded_trajs = self.embedding_layer(padded_trajs)
#         pack_padded_trajs = pack_padded_sequence(padded_trajs, trajs_len, batch_first=True, enforce_sorted=False)
#         _, (Traj_emb_zero, _) = self.lstm(pack_padded_trajs)
#         Traj_emb_zero = Traj_emb_zero.transpose(0,1).reshape(padded_trajs.shape[0], -1)
#         # Traj_emb_zero = self.dropout_layer(Traj_emb_zero)
#         # GCN
#         # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
#         # Traj_emb_one = self.dropout_layer(Traj_emb_one)
#         # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
#         # Traj_emb_two = self.dropout_layer(Traj_emb_two)
#         # GAT
#         Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
#         # Traj_emb_one = self.dropout_layer(Traj_emb_one)
#         Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0])
#         # Traj_emb_two = self.dropout_layer(Traj_emb_two)
#         Traj_emb_two = torch.cat([Traj_emb_two, Traj_emb_one, Traj_emb_zero], dim=-1)
#         predictions = self.predictor(Traj_emb_two)
#         return predictions


class GlobalTUL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_all_trajs = args.n_all_trajs
        self.hidden_size = args.hidden_size
        # self.embedding_layer = nn.Embedding(args.n_vocabs, args.hidden_size)
        # self.POI_to_traj_conv = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.embedding_layer = nn.Embedding(args.n_all_trajs, args.hidden_size)
        # self.global_graph_conv_one = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        # self.global_graph_conv_two = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.global_graph_conv_one = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.global_graph_conv_two = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.predictor = nn.Sequential(
            # nn.Linear(args.hidden_size, args.hidden_size),
            # nn.ReLU(),
            nn.Linear(args.hidden_size, args.n_users)
        )

    def forward(self, TrajTrajGraph):
        Traj_emb_zero = self.embedding_layer.weight
        # n_all_trajs, n_all_trajs
        Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
        Traj_emb_one = self.dropout_layer(Traj_emb_one)
        Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_weight=TrajTrajGraph[1])
        # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0])
        Traj_emb_two = self.dropout_layer(Traj_emb_two)
        # Traj_emb = Traj_emb_zero + Traj_emb_one + Traj_emb_two
        Traj_emb = Traj_emb_two
        # Traj_emb = Traj_emb_one
        predictions = self.predictor(Traj_emb)
        return predictions

class GlobalTUL_withSpatio(nn.Module):
    def __init__(self, args, n_relations=2):
        super().__init__()
        self.n_all_trajs = args.n_all_trajs
        self.hidden_size = args.hidden_size
        # self.embedding_layer = nn.Embedding(args.n_vocabs, args.hidden_size)
        # self.POI_to_traj_conv = GCNConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.embedding_layer = nn.Embedding(args.n_all_trajs, args.hidden_size)
        # self.global_graph_conv_one = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        # self.global_graph_conv_two = GATConv(args.hidden_size, args.hidden_size, add_self_loops=False)
        self.global_graph_conv_one = RGCNConv(args.hidden_size, args.hidden_size, num_relations=n_relations)
        self.global_graph_conv_two = RGCNConv(args.hidden_size, args.hidden_size, num_relations=n_relations)
        self.dropout_layer = nn.Dropout(p=args.dropout)
        self.predictor = nn.Sequential(
            # nn.Linear(args.hidden_size, args.hidden_size),
            # nn.ReLU(),
            nn.Linear(args.hidden_size, args.n_users)
        )

    def forward(self, TrajTrajGraph):
        Traj_emb_zero = self.embedding_layer.weight
        # n_all_trajs, n_all_trajs
        Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
        # Traj_emb_one = self.global_graph_conv_one(x=Traj_emb_zero, edge_index=TrajTrajGraph[0])
        # Traj_emb_one = self.dropout_layer(Traj_emb_one)
        Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0], edge_type=TrajTrajGraph[2])
        # Traj_emb_two = self.global_graph_conv_two(x=Traj_emb_one, edge_index=TrajTrajGraph[0])
        Traj_emb_two = self.dropout_layer(Traj_emb_two)
        # Traj_emb = Traj_emb_zero + Traj_emb_one + Traj_emb_two
        Traj_emb = Traj_emb_two
        # Traj_emb = Traj_emb_one
        predictions = self.predictor(Traj_emb)
        return predictions


class TULLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.loc_embedding = nn.Embedding(args.n_vocabs, args.hidden_size)
        self.lstm = nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size, batch_first=True, bidirectional=False)
        self.predictor = nn.Linear(args.hidden_size, args.n_users)

    def load_pretrained(self, weights, freeze=True):
        self.loc_embedding = nn.Embedding.from_pretrained(torch.tensor(weights, dtype=torch.float), freeze=freeze)

    def forward(self, trajs, trajs_len):
        trajs_emb = self.loc_embedding(trajs)
        trajs_emb = pack_padded_sequence(trajs_emb, trajs_len, batch_first=True, enforce_sorted=False)
        _, (trajs_emb, _) = self.lstm(trajs_emb)
        trajs_emb = trajs_emb.transpose(0,1).reshape(trajs.shape[0], -1)
        return self.predictor(trajs_emb)

class TULGRU(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.loc_embedding = nn.Embedding(args.n_vocabs, args.hidden_size)
        self.gru = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, batch_first=True, bidirectional=False)
        self.predictor = nn.Linear(args.hidden_size, args.n_users)

    def load_pretrained(self, weights, freeze=True):
        self.loc_embedding = nn.Embedding.from_pretrained(torch.tensor(weights, dtype=torch.float), freeze=freeze)

    def forward(self, trajs, trajs_len):
        trajs_emb = self.loc_embedding(trajs)
        trajs_emb = pack_padded_sequence(trajs_emb, trajs_len, batch_first=True, enforce_sorted=False)
        _, trajs_emb = self.gru(trajs_emb)
        trajs_emb = trajs_emb.transpose(0,1).reshape(trajs.shape[0], -1)
        return self.predictor(trajs_emb)

class TULBiLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.loc_embedding = nn.Embedding(args.n_vocabs, args.hidden_size)
        self.lstm = nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size, batch_first=True, bidirectional=True)
        self.predictor = nn.Linear(args.hidden_size*2, args.n_users)

    def load_pretrained(self, weights, freeze=True):
        self.loc_embedding = nn.Embedding.from_pretrained(torch.tensor(weights, dtype=torch.float), freeze=freeze)

    def forward(self, trajs, trajs_len):
        trajs_emb = self.loc_embedding(trajs)
        trajs_emb = pack_padded_sequence(trajs_emb, trajs_len, batch_first=True, enforce_sorted=False)
        _, (trajs_emb, _) = self.lstm(trajs_emb)
        trajs_emb = trajs_emb.transpose(0,1).reshape(trajs.shape[0], -1)
        return self.predictor(trajs_emb)