import torch
from torch.utils.data.dataloader import DataLoader
from transformers import HfArgumentParser, set_seed
from config import FoursquareConfig
from utils import print_args, construct_global_graph, accuracy, greedy_search, construct_global_graph_with_spatiotemporalinfo
from dataset import FoursquareDataset, BatchedDataset
from models import LSTMGlobalTUL, BiLSTMGlobalTUL, BiLSTMConcatGlobalTUL
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import copy


if __name__ == "__main__":

    parser = HfArgumentParser(FoursquareConfig)
    args = parser.parse_args_into_dataclasses()[0]

    set_seed(args.seed)
    device = torch.device(args.device)

    dataset = FoursquareDataset(args)
    #TrajTrajgraph_spatio = construct_global_graph_with_spatioinfo(dataset, args)            # N_trajs, N_vocabs
    TrajTrajgraph_spatio = construct_global_graph_with_spatiotemporalinfo(dataset, args)            # N_trajs, N_vocabs
    dataset.convert_all_trajs_to_vocab()
    # construct padded sequence
    trajs = []
    trajs_len = []
    for traj in dataset.train_trajs+dataset.val_trajs+dataset.test_trajs:
        trajs.append(torch.tensor(traj))
        trajs_len.append(len(traj))
    padded_trajs = pad_sequence(trajs, batch_first=True)

    # train_dataset = BatchedDataset(dataset.train_idx, dataset.train_uid_list)
    train_dataset = BatchedDataset(dataset.m_train_idx, [dataset.train_uid_list[i] for i in dataset.m_train_idx])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    args.n_vocabs = len(dataset.vocabs)
    args.n_users = len(dataset.users)
    args.n_all_trajs = dataset.n_all_trajs
    print_args(args)

    # Globalmodel = BiLSTMGlobalTUL(args)
    # Globalmodel = LSTMGlobalTUL(args)
    Globalmodel = BiLSTMConcatGlobalTUL(args)
    Globalmodel.to(device)
    #TrajTrajGraph = [torch.tensor(TrajTrajGraph[0], dtype=torch.long).to(device), torch.tensor(TrajTrajGraph[1], dtype=torch.float).to(device)]
    TrajTrajgraph_spatio = [torch.tensor(TrajTrajgraph_spatio[0], dtype=torch.long).to(device),
                            torch.tensor(TrajTrajgraph_spatio[1], dtype=torch.float).to(device),
                            torch.tensor(TrajTrajgraph_spatio[2], dtype=torch.long).to(device)]
    padded_trajs = padded_trajs.to(device)

    optimizer = torch.optim.Adam(Globalmodel.parameters(), args.lr, weight_decay=args.weight_decay)
    best_predictions = None
    best_acc = -1

    for epoch in range(args.epochs):
        Globalmodel.train()
        all_train_loss = 0
        for batch in train_dataloader:
            idxes, labels = [b.to(device) for b in batch]
            predictions = Globalmodel(padded_trajs, trajs_len, TrajTrajgraph_spatio)
            train_predictions = predictions[idxes]
            train_loss = F.cross_entropy(train_predictions, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            all_train_loss += train_loss.item()

        with torch.no_grad():
            Globalmodel.eval()
            predictions = Globalmodel(padded_trajs, trajs_len, TrajTrajgraph_spatio)
            val_label = torch.tensor(dataset.val_uid_list, dtype=torch.long).to(device)
            val_predictions = predictions[len(dataset.train_uid_list):len(dataset.train_uid_list) + len(dataset.val_uid_list)]
            val_loss = F.cross_entropy(val_predictions, val_label)
            val_acc_list = accuracy(val_predictions, val_label, topk=(1, 5, 10,))

            train_predictions = predictions[:len(dataset.train_uid_list)]
            train_label = torch.tensor(dataset.train_uid_list, dtype=torch.long).to(device)
            train_acc_list = accuracy(train_predictions, train_label, topk=(1, 5, 10,))

        print("Epoch: {}, TrainingLoss: {:.4f}, ValidationLoss: {:.4f}".format(epoch, all_train_loss, val_loss))
        print("Training: ACC@1: {:4f}, ACC@5: {:4f}, ACC@10: {:4f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}".format(
                train_acc_list[0].item(), train_acc_list[1].item(), train_acc_list[2].item(), train_acc_list[3],
                train_acc_list[4], train_acc_list[5]))
        print("Validation: ACC@1: {:4f}, ACC@5: {:4f}, ACC@10: {:4f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}".format(
                val_acc_list[0].item(), val_acc_list[1].item(), val_acc_list[2].item(), val_acc_list[3],
                val_acc_list[4], val_acc_list[5]))
        with torch.no_grad():
            Globalmodel.eval()
            predictions = Globalmodel(padded_trajs, trajs_len, TrajTrajgraph_spatio)
            test_label = torch.tensor(dataset.test_uid_list, dtype=torch.long).to(device)
            test_predictions = predictions[-len(dataset.test_uid_list):]
            acc_list = accuracy(test_predictions, test_label, topk=(1, 5, 10,))
            print("ACC@1: {:4f}, ACC@5: {:4f}, ACC@10: {:4f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}".format(
                acc_list[0].item(), acc_list[1].item(), acc_list[2].item(), acc_list[3], acc_list[4], acc_list[5]))

            if val_acc_list[0] > best_acc:
                best_predictions = test_predictions
                best_acc = val_acc_list[0]

    with torch.no_grad():
        # Globalmodel.eval()
        # predictions = Globalmodel(padded_trajs, trajs_len, TrajTrajgraph_spatio)
        test_label = torch.tensor(dataset.test_uid_list, dtype=torch.long).to(device)
        # test_predictions = predictions[-len(dataset.test_uid_list):]
        test_predictions = best_predictions
        acc_list = accuracy(test_predictions, test_label, topk=(1, 5, 10,))
        print("ACC@1: {:4f}, ACC@5: {:4f}, ACC@10: {:4f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}".format(
            acc_list[0].item(), acc_list[1].item(), acc_list[2].item(), acc_list[3], acc_list[4], acc_list[5]))
        preds_list = greedy_search(test_predictions.detach().cpu().numpy(), dataset.test_timestamps_list)
        gs_acc1 = 0
        for p, l in zip(preds_list, dataset.test_uid_list):
            if p == l:
                gs_acc1 += 1
        gs_acc1 = gs_acc1 / len(preds_list)
        print("ACC@1: {:4f}".format(gs_acc1))

    datapath = Path(args.datadir)
    #savepath = Path("./results/") / "SpatioTemporalSequential_{}_{}.txt".format(datapath.parts[2], datapath.parts[4])
    savepath = Path("./robustness/") / "SpatioTemporalSequential_{}_{}_m_{}.txt".format(datapath.parts[2], datapath.parts[4], args.m)
    with open(savepath, "w") as fsave:
        print("Acc@1: {:.5f}, Acc@5: {:.5f}, Acc@10: {:.5f}, Macro@F: {:4f}, Macro@P: {:4f}, Macro@R: {:4f}, GSACC@1: {:.5f}".format(
                acc_list[0].item(), acc_list[1].item(), acc_list[2].item(), acc_list[3], acc_list[4], acc_list[5], gs_acc1), file=fsave)
