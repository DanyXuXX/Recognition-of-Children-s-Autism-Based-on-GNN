
import os
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import torch
from scipy import io
from torch.utils.data import Subset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader, Data
from einops import repeat
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def K_Fold(folds, dataset, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset))):
        test_indices.append(index)

    return test_indices


class FSDataset:
    def __init__(self, args, seed):
        self.fc = []
        self.y = []
        root = args.path
        self.class_dict = {"HC": 0, "ASD": 1}

        label_list = os.listdir(root) # ['HC', 'ASD']
        # label_list.sort()
        FC_dir = "RegionSeries.mat"
        threshold = 0.2
        for label_files in label_list:
            list = os.listdir(os.path.join(root, label_files))
            # list.sort()
            label = torch.LongTensor([self.class_dict[label_files]]) # 0/1
            for files in list:
                subj_dir = os.path.join(root, label_files, files)
                subj_mat=np.loadtxt(subj_dir)[:176,:90]
                # isnan = (True in np.isnan(subj_mat_fc))
                # if isnan==True:
                print("reading data " + subj_dir)
                subj_mat_adj = np.corrcoef(np.transpose(subj_mat))
                subj_mat_adj = subj_mat_adj - np.diag(np.diag(subj_mat_adj))
                #take the upper triangle and compute the threshold
                subj_adj_up=subj_mat_adj[np.triu_indices(90,k=1)]
                subj_adj_list = subj_adj_up.reshape((-1))
                threindex = int(threshold * subj_adj_list.shape[0])
                thremax = subj_adj_list[subj_adj_list.argsort()[-1 * threindex-1]] # retrieve top threindex elements
                #avoiding Nan
                subj_adj_t = np.zeros((90, 90))
                subj_adj_t[subj_mat_adj > thremax] = 1
                subj_mat_adj=subj_adj_t
                # subj_mat_adj_list = subj_mat_adj.reshape((-1))
                # threindex = int(threshold * subj_mat_adj_list.shape[0])
                # thremax = subj_mat_adj_list[subj_mat_adj_list.argsort()[-1 * threindex]]
                # subj_mat_adj[subj_mat_adj < thremax] = 0
                # subj_mat_adj[subj_mat_adj >= thremax] = 1
                fcedge_index, _ = dense_to_sparse(torch.from_numpy(subj_mat_adj.astype(np.int16)))

                subj_mat_list = subj_mat.reshape((-1))
                subj_mat_new = (subj_mat - min(subj_mat_list)) / ( max(subj_mat_list) - min(subj_mat_list))
                # subj_mat_fc_new = (subj_mat_fc - np.mean(subj_mat_fc, axis=0, keepdims=True)) / np.std(subj_mat_fc, axis=0, keepdims=True)
                # get Adjacency Matrix
                subj_mat_new = np.transpose(subj_mat_new)

                rowsum = np.array(subj_mat_adj.sum(1))
                N = np.diag(rowsum)
                # get Degree Matrix
                degree_C_BOLD=np.concatenate((N,subj_mat_new),1)
                # BOLD_C_degree=np.concatenate((subj_mat_fc_new,N),1)
                ###one-hot###
                # subj_mat_fc_new=np.eye(args.num_nodes)
                self.fc.append(
                    Data(x=torch.from_numpy(degree_C_BOLD).float(), edge_index=fcedge_index, y=torch.tensor(label)))

                self.y.append(label)

        self.choose_data = self.fc
        self.k_fold = args.repetitions
        self.k_fold_split = K_Fold(self.k_fold, self.choose_data, seed)
        self.batch_size = args.batch_size

    def kfold_split(self, test_index):
        assert test_index < self.k_fold
        test_split = self.k_fold_split[test_index]

        train_mask = np.ones(len(self.choose_data))
        train_mask[test_split] = 0
        train_split = train_mask.nonzero()[0]

        train_subset = Subset(self.choose_data, train_split.tolist())
        test_subset = Subset(self.choose_data, test_split.tolist())

        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
