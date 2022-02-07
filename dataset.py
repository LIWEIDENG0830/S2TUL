import torch
import datetime
import numpy as np
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence


class FoursquareDataset:
    def __init__(self, args):
        self.args = args
        self.m = self.args.m

        # timezone of NewYork
        #self.tz = datetime.timezone(-datetime.timedelta(hours=4))
        self.tz = None

        self.datadir = Path(args.datadir)
        self.test_trajs_path = self.datadir / "test_trajs.txt"
        self.test_trajs_time_path = self.datadir / "test_trajs_time.txt"
        self.val_trajs_path = self.datadir / "val_trajs.txt"
        self.val_trajs_time_path = self.datadir / "val_trajs_time.txt"
        self.train_trajs_path = self.datadir / "train_trajs.txt"
        self.train_trajs_time_path = self.datadir / "train_trajs_time.txt"
        self.vidx_to_latlon_path = self.datadir / "vidx_to_latlon.txt"

        self.test_uid_list, self.test_trajs = self.read_trajs_file(self.test_trajs_path)
        self.test_timestamps_list, self.test_weekdays_list, self.test_hours_list = self.read_trajs_time_file(self.test_trajs_time_path)
        self.val_uid_list, self.val_trajs = self.read_trajs_file(self.val_trajs_path)
        self.val_timestamps_list, self.val_weekdays_list, self.val_hours_list = self.read_trajs_time_file(self.val_trajs_time_path)
        self.train_uid_list, self.train_trajs = self.read_trajs_file(self.train_trajs_path)
        self.train_timestamps_list, self.train_weekdays_list, self.train_hours_list = self.read_trajs_time_file(self.train_trajs_time_path)

        idxes = list(range(len(self.train_trajs)+len(self.val_trajs)+len(self.test_trajs)))
        self.train_idx = idxes[:len(self.train_trajs)]
        self.val_idx = idxes[len(self.train_trajs):len(self.train_trajs)+len(self.val_trajs)]
        self.test_idx = idxes[-len(self.test_trajs):]
        self.n_all_trajs = len(self.train_trajs)+len(self.val_trajs)+len(self.test_trajs)

        # latlon for constructing Ngraph
        self.vidx_to_latlon = self.read_vidx_to_latlon(self.vidx_to_latlon_path)

        # user set for linking
        self.users = dict([(uid, idx) for idx, uid in enumerate(set(self.train_uid_list))])

        # vocabs
        self.vocabs = {"PAD":0, "UNK": 1}
        self.build_vocabs(self.train_trajs, self.vocabs)

        #self.train_trajs = self.convert_trajs_to_vocab(self.train_trajs)
        #self.val_trajs = self.convert_trajs_to_vocab(self.val_trajs)
        #self.test_trajs = self.convert_trajs_to_vocab(self.test_trajs)

        self.train_uid_list = [self.users[uid] for uid in self.train_uid_list]
        self.val_uid_list = [self.users[uid] for uid in self.val_uid_list]
        self.test_uid_list = [self.users[uid] for uid in self.test_uid_list]

        self.m_train_idx = self.obtain_training_idxes(args.m)

    def obtain_training_trajs_by_m_train_idx(self):
        train_trajs, train_uid_list = [], []
        for idx in self.m_train_idx:
            train_trajs.append(self.train_trajs[idx])
            train_uid_list.append(self.train_uid_list[idx])
        return train_trajs, train_uid_list

    def obtain_training_idxes(self, m):
        assert 0 < m and m <= 1
        # uid_to_idxes = dict()
        # for idx, uid in enumerate(self.train_uid_list):
        #     if uid not in uid_to_idxes:
        #         uid_to_idxes[uid] = []
        #     uid_to_idxes[uid].append(idx)
        # idxes = []
        # for uid in uid_to_idxes:
        #     idx_list = uid_to_idxes[uid]
        #     idxes.extend(idx_list[:int(len(idx_list)*m)])
        np.random.seed(0)
        train_idxs = list(range(len(self.train_trajs)))
        np.random.shuffle(train_idxs)
        train_end = int(self.m * len(self.train_trajs))
        idxes = train_idxs[:train_end]
        #self.train_trajs = [self.train_trajs[i] for i in train_idxs[:train_end]]
        #self.train_uid_list = [self.train_uid_list[i] for i in train_idxs[:train_end]]
        return idxes

    def convert_all_trajs_to_vocab(self):
        self.train_trajs, train_unknown_keys = self.convert_trajs_to_vocab(self.train_trajs)
        self.val_trajs, val_unknown_keys = self.convert_trajs_to_vocab(self.val_trajs)
        self.test_trajs, test_unknown_keys = self.convert_trajs_to_vocab(self.test_trajs)
        # import pdb; pdb.set_trace()

    def convert_trajs_to_vocab(self, trajs):
        converted_trajs = []
        unknown_keys = []
        for traj in trajs:
            converted_traj = []
            for point in traj:
                if point not in self.vocabs:
                    converted_traj.append(self.vocabs["UNK"])
                    unknown_keys.append(point)
                else:
                    converted_traj.append(self.vocabs[point])
            converted_trajs.append(converted_traj)
        return converted_trajs, unknown_keys

    def build_vocabs(self, trajs, vocabs):
        for traj in trajs:
            for point in traj:
                if point not in vocabs:
                    vocabs[point] = len(vocabs)

    def read_trajs_file(self, filepath):
        uid_list = []
        trajs_list = []
        with open(filepath) as f:
            for traj in f:
                traj = traj.strip().split(" ")
                uid_list.append(int(traj[0]))
                trajs_list.append(list(map(int, traj[1:])))
        return uid_list, trajs_list

    def read_trajs_time_file(self, filepath):
        timestamps_list = []
        weekdays_list = []
        hours_list = []
        with open(filepath) as f:
            for traj_time in f:
                traj_time = traj_time.strip().split(" ")
                timestamps = list(map(float, traj_time))
                datetime_timestamps = [datetime.datetime.fromtimestamp(timestamp, tz=self.tz) for timestamp in map(float, traj_time)]
                weekdays = np.array([timestamp.weekday() for timestamp in datetime_timestamps])
                # zero for week day, one for weekends
                weekdays[weekdays < 5] = 0
                weekdays[weekdays > 4] = 1
                # time slot is half hour
                hours = np.array([timestamp.hour*2 for timestamp in datetime_timestamps])
                timestamps_list.append(timestamps)
                weekdays_list.append(weekdays.tolist())
                hours_list.append(hours.tolist())
        return timestamps_list, weekdays_list, hours_list

    def read_vidx_to_latlon(self, filepath):
        vidx_to_latlon = dict()
        with open(filepath) as f:
            for line in f:
                vidx, lat, lon = line.strip().split(" ")
                vidx_to_latlon[int(vidx)] = [float(lat), float(lon)]
        return vidx_to_latlon

class BatchedDataset:
    def __init__(self, idxes, labels):
        self.idxes = idxes
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.idxes[i], self.labels[i]

class BatchedFusionDataset:
    def __init__(self, trajs, idxes, labels):
        self.trajs = trajs
        self.idxes = idxes
        self.labels = labels

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, i):
        return torch.tensor(self.trajs[i], dtype=torch.long), len(self.trajs[i]), self.idxes[i], self.labels[i]

    def collate_fun(self, data):
        trajs = [d[0] for d in data]
        trajs_len = [d[1] for d in data]
        idxes = [d[2] for d in data]
        labels = [d[3] for d in data]
        pad_trajs = pad_sequence(trajs, batch_first=True)
        return pad_trajs, torch.tensor(trajs_len, dtype=torch.long), torch.tensor(idxes, dtype=torch.long), \
               torch.tensor(labels, dtype=torch.long)