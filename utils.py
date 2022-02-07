import math
import pickle
import torch
import shutil
import numpy as np
from pathlib import Path
from prettytable import PrettyTable
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import networkx as nx
from networkx.convert_matrix import from_scipy_sparse_matrix
from sklearn.metrics.pairwise import pairwise_distances
from datetime import datetime
import copy


def load_embeddings(args, dataset):
    with open(args.pretrain_loc_path) as f:
        n_words, dims = list(map(int,f.readline().strip().split()))
        print("Load Embeddings! {} words! {} dims!".format(n_words, dims))
        embeddings = np.zeros((n_words, dims), dtype=np.float)
        w2idx = dict()
        for idx, line in enumerate(f):
            line = line.strip().split()
            embeddings[idx] = list(map(float,line[1:]))
            w2idx[int(line[0])] = idx
    dataset.vocabs = w2idx
    return embeddings

def print_args(args):
    x = PrettyTable()
    x.field_names = ["Parameters", "Values"]
    for parm, value in vars(args).items():
        x.add_row([parm, value])
    print(x)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # Computes Macro@F
        pred = output.argmax(-1).squeeze().cpu().numpy()
        macrof = f1_score(target.cpu().numpy(), pred, average="macro")
        res.append(macrof*100)
        macrop = precision_score(target.cpu().numpy(), pred, average="macro")
        res.append(macrop*100)
        macror = recall_score(target.cpu().numpy(), pred, average="macro")
        res.append(macror*100)
        return res


def construct_transition_graph(dataset):
    A = np.zeros((len(dataset.vocabs), len(dataset.vocabs)))
    for traj in dataset.train_trajs:
        for i in range(len(traj)-1):
            p1, p2 = traj[i], traj[i+1]
            A[p1,p2] += 1
    A = coo_matrix(A)
    return A.data, np.stack([A.row, A.col], 0)


def Jaccard_similarity(A, B):
    '''
    :param A: a set that contains some items
    :param B: a set that contains some times
    :return: jaccard similarity between A and B
    '''
    return len(set(A) & set(B)) / len(set(A) | set(B))


def construct_global_graph(dataset, args=None):
    train_trajs, val_trajs, test_trajs = dataset.train_trajs, dataset.val_trajs, dataset.test_trajs
    whole_trajs = train_trajs + val_trajs + test_trajs
    # N_trajs, N_vocabs
    x, y = [], []
    for idx, traj in enumerate(whole_trajs):
        for POI in traj:
            x.append(idx)
            y.append(POI)
    TV_A = csr_matrix((np.ones(len(x), dtype=np.float), (x, y)))
    # N_trajs, N_trajs
    TT_A = TV_A.dot(TV_A.T)
    # TV_A = TV_A.tocoo()
    TT_A = TT_A.tocoo()
    # nx_graph = from_scipy_sparse_matrix(TT_A, create_using=nx.Graph)
    # node_color = dataset.train_uid_list + dataset.val_uid_list + dataset.test_uid_list
    # plt.figure()
    # nx.draw_networkx(nx_graph, node_color=node_color)
    # plt.savefig("./analysis/Global_Graph.pdf")
    #return np.stack([TV_A.row+len(dataset.vocabs), TV_A.col], axis=0), (np.stack([TT_A.row, TT_A.col], axis=0), TT_A.data)
    return (np.stack([TT_A.row, TT_A.col], axis=0), TT_A.data)


def construct_global_graph_with_spatioinfo(dataset, args):
    train_trajs, val_trajs, test_trajs = dataset.train_trajs, dataset.val_trajs, dataset.test_trajs
    whole_trajs = train_trajs + val_trajs + test_trajs
    # build vocab dict which is used to convert POIs to vocabs
    vocabs = copy.deepcopy(dataset.vocabs)
    dataset.build_vocabs(dataset.val_trajs, vocabs)
    dataset.build_vocabs(dataset.test_trajs, vocabs)
    reversed_vocabs = dict([(i, v) for v, i in vocabs.items()])
    # N_trajs, N_vocabs
    x, y = [], []
    for idx, traj in enumerate(whole_trajs):
        for POI in traj:
            x.append(idx)
            # converting POIs to vocabs
            y.append(vocabs[POI])
    TV_A = csr_matrix((np.ones(len(x), dtype=np.float), (x, y)))

    # neighbor information, N_vocabs, N_vocabs
    POI_max = max(y)
    POIlatlon = []
    for i in range(POI_max+1):
        i = reversed_vocabs[i]
        if i not in dataset.vidx_to_latlon:
            POIlatlon.append([0, 0])
        else:
            POIlatlon.append(lonlat2meters(dataset.vidx_to_latlon[i][1], dataset.vidx_to_latlon[i][0]))
    POIneighbor_dist = pairwise_distances(POIlatlon, n_jobs=4)
    POIneighbor_graph = np.where(POIneighbor_dist<args.spatio, 1, 0)
    # Mask the diagnoal; Themself should not their own neighbor, which will fuse the two type of information, i.e., spatial info and repeative info.
    POIneighbor_graph = POIneighbor_graph - np.diag(np.diag(POIneighbor_graph))
    POI_A = csr_matrix(POIneighbor_graph)
    TNV_A = TV_A.dot(POI_A)
    # N_trajs, N_trajs, TNV_A[i, j] == 1, only when they are similar in spatial information.
    # TNV_A = TNV_A.dot(TNV_A.T)
    TNV_A = TNV_A.dot(TV_A.T)
    TNV_A = TNV_A.tocoo()

    # N_trajs, N_trajs
    TT_A = TV_A.dot(TV_A.T)
    TT_A = TT_A.tocoo()
    # return (np.stack([TT_A.row, TT_A.col], axis=0), TT_A.data), (np.stack([TNV_A.row, TNV_A.col], axis=0), TNV_A.data)
    TTNV_rows = np.concatenate([TNV_A.row, TT_A.row])
    TTNV_cols = np.concatenate([TNV_A.col, TT_A.col])
    TTNV_data = np.concatenate([TNV_A.data, TT_A.data])
    edge_type = [0]*len(TNV_A.row) + [1]*len(TT_A.row)
    return (np.stack([TTNV_rows, TTNV_cols], axis=0), TTNV_data, edge_type)


def construct_global_graph_with_spatiotemporalinfo(dataset, args):
    train_trajs, val_trajs, test_trajs = dataset.train_trajs, dataset.val_trajs, dataset.test_trajs
    train_trajs_time, val_trajs_time, test_trajs_time = dataset.train_timestamps_list, dataset.val_timestamps_list, dataset.test_timestamps_list
    whole_trajs = train_trajs + val_trajs + test_trajs
    whole_trajs_time = train_trajs_time + val_trajs_time + test_trajs_time
    # build vocab dict which is used to convert POIs to vocabs
    vocabs = copy.deepcopy(dataset.vocabs)
    dataset.build_vocabs(dataset.val_trajs, vocabs)
    dataset.build_vocabs(dataset.test_trajs, vocabs)
    reversed_vocabs = dict([(i, v) for v, i in vocabs.items()])
    # N_trajs, N_vocabs
    x, y = [], []
    for idx, traj in enumerate(whole_trajs):
        for POI in traj:
            x.append(idx)
            # converting POIs to vocabs
            y.append(vocabs[POI])
    TV_A = csr_matrix((np.ones(len(x), dtype=np.float), (x, y)))

    # neighbor information, N_vocabs, N_vocabs
    POI_max = max(y)
    POIlatlon = []
    for i in range(POI_max + 1):
        i = reversed_vocabs[i]
        if i not in dataset.vidx_to_latlon:
            POIlatlon.append([0, 0])
        else:
            POIlatlon.append(lonlat2meters(dataset.vidx_to_latlon[i][1], dataset.vidx_to_latlon[i][0]))
    POIneighbor_dist = pairwise_distances(POIlatlon, n_jobs=4)
    POIneighbor_graph = np.where(POIneighbor_dist < args.spatio, 1, 0)
    # Mask the diagnoal; Themself should not their own neighbor, which will fuse the two type of information, i.e., spatial info and repeative info.
    POIneighbor_graph = POIneighbor_graph - np.diag(np.diag(POIneighbor_graph))
    POI_A = csr_matrix(POIneighbor_graph)
    TNV_A = TV_A.dot(POI_A)
    # N_trajs, N_trajs, TNV_A[i, j] == 1, only when they are similar in spatial information.
    # TNV_A = TNV_A.dot(TNV_A.T)
    TNV_A = TNV_A.dot(TV_A.T)
    TNV_A = TNV_A.tocoo()

    # verify two trajectories that are similar in spatial information, whether are similar in temporal information
    row, col, data = [], [], []
    for i, j in zip(TNV_A.row, TNV_A.col):
        traj_i_time, traj_j_time = whole_trajs_time[i], whole_trajs_time[j]
        traj_i, traj_j = whole_trajs[i], whole_trajs[j]
        flag = 0
        counter = 0
        for idx1, ti_t in enumerate(traj_i_time):
            if flag == 1:
                break
            for idx2, tj_t in enumerate(traj_j_time):
                ti_t_dt = datetime.fromtimestamp(ti_t)
                tj_t_dt = datetime.fromtimestamp(tj_t)
                ti_t_daytime = ti_t_dt.hour + ti_t_dt.minute/60
                tj_t_daytime = tj_t_dt.hour + tj_t_dt.minute/60
                max_daytime = max([ti_t_daytime, tj_t_daytime])
                min_daytime = min([ti_t_daytime, tj_t_daytime])
                loc1_meters = lonlat2meters(dataset.vidx_to_latlon[traj_i[idx1]][1], dataset.vidx_to_latlon[traj_i[idx1]][0])
                loc2_meters = lonlat2meters(dataset.vidx_to_latlon[traj_j[idx2]][1], dataset.vidx_to_latlon[traj_j[idx2]][0])
                if np.sqrt(np.power(loc1_meters[0]-loc2_meters[0],2)+np.power(loc1_meters[1]-loc2_meters[1],2)) >= args.spatio:
                    continue
                if max_daytime-min_daytime < args.temporal or min_daytime-max_daytime+24 < args.temporal:
                    flag = 1
                    counter += 1
        if flag == 1:
            row.append(i)
            col.append(j)
            data.append(counter)

    # N_trajs, N_trajs
    TT_A = TV_A.dot(TV_A.T)
    TT_A = TT_A.tocoo()
    TTNV_rows = np.concatenate([np.array(row, dtype=int), TT_A.row])
    TTNV_cols = np.concatenate([np.array(col, dtype=int), TT_A.col])
    TTNV_data = np.concatenate([np.array(data, dtype=float), TT_A.data])
    edge_type = [0] * len(row) + [1] * len(TT_A.row)
    datapath = Path(args.datadir)
    savepath = Path("./results_tr/") / "SpatioTemporal_{}_{}_stfactor_{}.txt".format(datapath.parts[2], datapath.parts[4], args.temporal)
    with open(savepath, "w") as f:
        print("repeat: {}, spatial: {}, spatiotemporal: {}".format(len(TT_A.row), len(TNV_A.row), len(row)), file=f)
    return (np.stack([TTNV_rows, TTNV_cols], axis=0), TTNV_data, edge_type)


def construct_two_global_graph(dataset):
    train_trajs, val_trajs, test_trajs = dataset.train_trajs, dataset.val_trajs, dataset.test_trajs
    whole_trajs = train_trajs + val_trajs + test_trajs
    # N_trajs, N_vocabs
    x, y = [], []
    for idx, traj in enumerate(whole_trajs):
        for POI in traj:
            x.append(idx)
            y.append(POI)
    # N_trajs, N_vocabs
    TV_A = csr_matrix((np.ones(len(x), dtype=np.float), (x, y)))
    # N_trajs, N_trajs
    TT_A = TV_A.dot(TV_A.T)
    # TV_A = TV_A.tocoo()
    TT_A = TT_A.tocoo()
    # nx_graph = from_scipy_sparse_matrix(TT_A, create_using=nx.Graph)
    # node_color = dataset.train_uid_list + dataset.val_uid_list + dataset.test_uid_list
    # plt.figure()
    # nx.draw_networkx(nx_graph, node_color=node_color)
    # plt.savefig("./analysis/Global_Graph.pdf")
    #return np.stack([TV_A.row+len(dataset.vocabs), TV_A.col], axis=0), (np.stack([TT_A.row, TT_A.col], axis=0), TT_A.data)
    return (np.stack([TT_A.row, TT_A.col], axis=0), TT_A.data)


# def lonlat2meters(lon, lat):
#     semimajoraxis = 6378137.0
#     east = lon * 0.017453292519943295
#     north = lat * 0.017453292519943295
#     t = math.sin(north)
#     # x, y
#     try:
#         return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))
#     except:
#         import pdb; pdb.set_trace()

def lonlat2meters(lon, lat):
    x = lon * 20037508.34 / 180
    try:
        y = math.log(math.tan((90 - 1e-10 + lat) * math.pi / 360)) / (math.pi / 180)
    except:
        y = math.log(math.tan((90 + 1e-10 + lat) * math.pi / 360)) / (math.pi / 180)
    y = y * 20037508.34 / 180
    return x, y


def construct_neighbor_graph(dataset, args):
    if Path(args.Ngraph_path.format(args.threshold)).exists():
        return pickle.load(Path(args.Ngraph_path.format(args.threshold)).open("rb"))
    else:
        idx_to_xy = dict()
        for vocab in dataset.vocabs:
            if vocab not in dataset.vidx_to_latlon:
                continue
            lat, lon = dataset.vidx_to_latlon[vocab]
            x, y = lonlat2meters(lon, lat)
            idx_to_xy[dataset.vocabs[vocab]] = np.array([x, y])
        # construct an empty graph firstly
        A = np.zeros((len(dataset.vocabs), len(dataset.vocabs)))
        for p1 in idx_to_xy:
            for p2 in idx_to_xy:
                p1_xy, p2_xy = idx_to_xy[p1], idx_to_xy[p2]
                if np.sqrt(np.power(p1_xy-p2_xy, 2).sum()) < args.threshold:
                    A[p1,p2] = 1
        A = coo_matrix(A)
        pickle.dump([A.data, np.stack([A.row, A.col], 0)], Path(args.Ngraph_path.format(args.threshold)).open("wb"))
        return A.data, np.stack([A.row, A.col], 0)


def greedy_search(predictions, timestamps_list):
    for timestamps in timestamps_list:
        # 3600*6 is the time interval to split trajectories
        #timestamps.append(timestamps[-1]+3600*6)
        timestamps.append(timestamps[-1])
    def is_overlap(user_ts_list, trajs_ts):
        for user_ts in user_ts_list:
            if (user_ts[-1]-user_ts[0])+(trajs_ts[-1]-trajs_ts[0]) > max(user_ts+trajs_ts)-min(user_ts+trajs_ts):
                return True
        return False
    # predictions: N, n_users
    pred_labels = [-1] * predictions.shape[0]
    pred_list = []
    user_timestamp_dict = dict([(i,[]) for i in range(predictions.shape[1])])
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            pred_list.append([predictions[i,j], i, j])
    for p, i, j in tqdm(sorted(pred_list, key=lambda x:x[0], reverse=True)):
        # if current trajectory have been allocated.
        if pred_labels[i] != -1:
            continue
        # if overlap
        if is_overlap(user_timestamp_dict[j], timestamps_list[i]):
            continue
        else:
            pred_labels[i] = j
            user_timestamp_dict[j].append(timestamps_list[i])
    return pred_labels

if __name__ == "__main__":
    lon, lat = 104.073694, 30.697218
    print(lonlat2meters(lon, lat))
    print(lonlat2meters_v2(lon, lat))
