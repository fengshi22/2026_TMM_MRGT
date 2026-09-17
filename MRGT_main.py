import argparse
import pytorch_lightning as pl
import torch
import numpy as np

from torch.utils.data import DataLoader
from func.mvdataset_load import dataset_loader
from model.MRGT_model import MRGT
import yaml
from pathlib import Path


def RGTMV_training(args):
    set_seed(args.seed)

    train_dataset = dataset_loader(name='train', args=args, load_from_file=False)
    test_dataset = dataset_loader(name='test', args=args, load_from_file=True)

    num_edge_type = len(train_dataset.relation_list)
    label_num = len(torch.unique(train_dataset.label_list))
    feature_data_in = [train_dataset.view_features[i].shape[1] for i in range(len(train_dataset.view_features))]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = MRGT(num_edge_type=num_edge_type,
                 label_num=label_num,
                 features_data_in=feature_data_in,
                 features_data_out=args.features_data_out,
                 hid_size=args.hid_size,
                 dropout_rate=args.dropout_rate,
                 num_heads=args.num_heads,
                 batch_size=args.batch_size,
                 alpha=args.alpha,
                 beta=args.beta)

    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=args.epoch_num, precision=16)

    print("Training begin!")

    trainer.fit(model, train_loader)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=True)

    return test_result


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(dataset_name: str):
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Dataset {dataset_name} not found in config")

    return config["datasets"][dataset_name]


if __name__ == "__main__":
    dataset_list = {
        1: "20newsgroups",
        2: "ALOI",
        3: "Hdigit",
        4: "HW",
        5: "Mfeat",
        6: "prokaryotic",
        7: "UCI",
        8: "NoisyMNIST_15000",
    }

    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument("--dataset-name", type=str, default='20newsgroups')

    known_args, _ = parser.parse_known_args()
    params = load_config(known_args.dataset_name)

    parser.add_argument("--seed", type=int, default=22, help="Random seed for network training.")
    parser.add_argument("--ratio", type=float, default=0.1, help="Ratio for labeled training.")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout_rate")

    parser.add_argument("--load_path", type=str, default='./data/',
                        help="dataset path")

    parser.add_argument("--batch-size", type=int, default=1, help="Number of batch size.")
    parser.add_argument("--epoch-num", type=int, default=params['epoch_num'],
                        help="Number of training epochs.")
    parser.add_argument("--k", type=int, default=params['k'], help="k of KNN graph.")
    parser.add_argument("--num-heads", nargs='+', default=params['num_heads'], help="num_heads")
    parser.add_argument("--features_data_out", type=int, default=params['features_data_out'],
                        help="Number of features_data_out.")
    parser.add_argument("--hid_size", type=int, default=params['hid_size'],
                        help="Number of HAN_hid_out(64).")
    parser.add_argument("--alpha", type=float, default=params['alpha'], help="Weight of loss.")
    parser.add_argument("--beta", type=float, default=.5, help="Weight of loss.")

    args = parser.parse_args()

    RGTMV_training(args)

