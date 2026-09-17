import os
import torch
import scipy.io
import hdf5storage
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale
from func.loadRelation import load_relation_knn

class dataset_loader(Dataset):
    def __init__(self, name, args, load_from_file=True):
        self.name = name
        self.batch_size = args.batch_size
        data_dir = os.path.join("./training_data", args.dataset_name)
        exist_k = os.path.exists(os.path.join(data_dir, "data_" + str(args.k) + ".pth"))

        if load_from_file and os.path.exists(data_dir) and exist_k:
            print("Loading Existing adjacency matrix.")
            load_data = torch.load(os.path.join(data_dir, "data_" + str(args.k) + ".pth"), weights_only=False)
            features = load_data["features"]
            label_list = load_data["label_list"]
            relation_list = load_data["relation_list"]

        else:
            print("Generating adjacency matrix.")
            load_path = args.load_path
            features, label_list = loadMatData(os.path.join(load_path, args.dataset_name))
            features = feature_normalization(features)

            relation_list, adj_list = load_relation_knn(features, k=args.k)

            os.makedirs(data_dir, exist_ok=True)
            save_data = {'features': features, 'label_list': label_list, 'relation_list': relation_list}
            torch.save(save_data, os.path.join(data_dir, "data_" + str(args.k) + ".pth"))

        self.label_list = torch.Tensor(label_list).float()
        self.relation_list = relation_list

        view_features = []
        for i in range(features.shape[1]):
            view_features.append(torch.from_numpy(features[0, i]))
        self.view_features = view_features

        p_labeled, p_unlabeled = generate_partition(label_list, args.ratio, args.seed)
        self.p_labeled = torch.Tensor(p_labeled).float()
        self.p_unlabeled = torch.Tensor(p_unlabeled).float()

        if self.name == "train":
            self.length = len(self.p_labeled) // self.batch_size if len(self.p_labeled) > 0 else 1
        elif self.name == "test":
            self.length = len(self.p_unlabeled) // self.batch_size if len(self.p_unlabeled) > 0 else 1
        else:
            raise ValueError(f"Dataset name '{self.name}' not recognized. Use 'train' or 'test'.")


        if self.length == 0:
            self.length = 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.name == "train":
            return (self.view_features, self.relation_list, self.p_labeled, self.label_list)
        elif self.name == "test":
            return (self.view_features, self.relation_list, self.p_unlabeled, self.label_list)
        else:
            raise ValueError(f"Dataset name '{self.name}' not recognized. Use 'train' or 'test'.")


def loadMatData(data_name):
    try:
        data = scipy.io.loadmat(data_name)
    except:
        data = hdf5storage.loadmat(data_name)

    try:
        features = data['X']
        label_list = data['Y']
    except:
        features = data['fea']
        label_list = data['gt']
    label_list = label_list.flatten()
    label_list = label_list - 1
    return features, label_list


def feature_normalization(features, normalization_type='normalize'):
    for idx, fea in enumerate(features[0]):
        if normalization_type == 'minmax_scale':
            features[0][idx] = minmax_scale(fea)
        elif normalization_type == 'maxabs_scale':
            features[0][idx] = maxabs_scale(fea)
        elif normalization_type == 'normalize':
            features[0][idx] = normalize(fea)
        elif normalization_type == 'robust_scale':
            features[0][idx] = robust_scale(fea)
        elif normalization_type == 'scale':
            features[0][idx] = scale(fea)
        else:
            print("Please enter a correct normalization type!")
    return features


def generate_partition(labels, ratio=0.1, seed=22):
    each_class_num = count_each_class_num(labels)

    labeled_each_class_num = {label: max(round(count * ratio), 1)
                              for label, count in each_class_num.items()}

    index = list(range(len(labels)))

    if seed >= 0:
        random.seed(seed)
        random.shuffle(index)

    shuffled_labels = labels[index]

    p_labeled, p_unlabeled = [], []
    for idx, label in enumerate(shuffled_labels):
        if labeled_each_class_num[label] > 0:
            labeled_each_class_num[label] -= 1
            p_labeled.append(index[idx])
        else:
            p_unlabeled.append(index[idx])

    return p_labeled, p_unlabeled


def count_each_class_num(labels):
    count_dict = {}
    for label in labels:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict