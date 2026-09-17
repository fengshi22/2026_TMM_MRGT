import pytorch_lightning as pl
import torch
from torch import nn
from sklearn import metrics
from model.MRGT_Layer import Learn_Layer


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def cosine_similarity_loss(self, A, B):
        A_flat = A.view(A.size(0), -1)
        B_flat = B.view(B.size(0), -1)
        cos = nn.CosineSimilarity(dim=1)
        return 1 - cos(A_flat, B_flat).mean()


class MRGT(pl.LightningModule):
    def __init__(self, num_edge_type, label_num, hid_size,
                 features_data_in, features_data_out,
                 batch_size, num_heads, dropout_rate, alpha, beta):
        super().__init__()
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.previous_representation = None
        self.alpha = alpha
        self.beta = beta

        count = 0
        if len(features_data_in) == 1:
            count = int(features_data_in[0])
        else:
            for i in range(len(features_data_in)):
                setattr(self, f'in_linear{i}', nn.Linear(features_data_in[i],
                                                         int(features_data_in[i] * 2 / 3), bias=True))
                count += int(features_data_in[i] * 2 / 3)

        self.linear = nn.Linear(count, features_data_out)
        self.RGT_layers = self._build_rgt_layers(num_edge_type, features_data_out, hid_size, num_heads)
        self.output1 = nn.Linear(hid_size, int(hid_size / 2))
        self.output2 = nn.Linear(int(hid_size / 2), label_num)

        self._init_weights(features_data_in)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.CELoss = nn.CrossEntropyLoss()
        self.MLoss = MyLoss()
        self.ReLU = nn.LeakyReLU()

    def _init_weights(self, features_data_in):
        if len(features_data_in) > 1:
            for i in range(len(features_data_in)):
                layer = getattr(self, f'in_linear{i}')
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')

        torch.nn.init.kaiming_normal_(self.linear.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.output1.weight, nonlinearity="leaky_relu")
        torch.nn.init.kaiming_normal_(self.output2.weight, nonlinearity="leaky_relu")

    def _build_rgt_layers(self, num_edge_type, features_data_out, hid_size, num_heads):
        RGT_layers = nn.ModuleList()
        RGT_layers.append(Learn_Layer(num_edge_type=num_edge_type, feature_size=features_data_out,
                                      hid_size=hid_size, layer_num_heads=num_heads[0],
                                      dropout_rate=self.dropout_rate))
        for l in range(1, len(num_heads)):
            RGT_layers.append(Learn_Layer(num_edge_type=num_edge_type, feature_size=hid_size,
                                          hid_size=hid_size, layer_num_heads=num_heads[l],
                                          dropout_rate=self.dropout_rate))
        return RGT_layers

    def forward_features(self, view_features):
        if len(view_features) == 1:
            learn_features = self.dropout(self.ReLU(self.linear(view_features[0][0].float())))
        else:
            new_features_list = []
            for i in range(len(view_features)):
                linear_layer = getattr(self, f'in_linear{i}')
                processed = linear_layer(view_features[i][0].float())
                processed = self.ReLU(processed)
                processed = self.dropout(processed)
                new_features_list.append(processed)
            cat_features = torch.cat(new_features_list, dim=1)
            learn_features = self.dropout(self.ReLU(self.linear(cat_features)))
        return learn_features

    def apply_rgt_layers(self, learn_features, relation):
        for RGT in self.RGT_layers:
            learn_features, _ = RGT(learn_features, relation)  # 忽略view_learn_features
        learned_features = self.dropout(self.ReLU(self.output1(learn_features)))
        return learned_features

    def training_step(self, train_batch, batch_idx):
        view_features, relation, p_labeled, label_list = train_batch
        p_labeled = p_labeled[0]
        label_list = label_list[0]

        learn_features = self.forward_features(view_features)
        learned_features = self.apply_rgt_layers(learn_features, relation)

        batch_id = p_labeled.tolist()
        pred = self.output2(learned_features[batch_id])
        label = torch.LongTensor(label_list[batch_id].tolist())
        loss1 = self.CELoss(pred.cuda(), label.cuda())

        if self.previous_representation is not None:
            loss2 = self.MLoss.cosine_similarity_loss(self.previous_representation, learned_features)
        else:
            loss2 = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        loss4 = torch.norm(learned_features[batch_id], p=2) / learned_features[batch_id].numel()
        total_loss = loss1 + self.alpha * loss2 + self.beta * loss4

        self.previous_representation = learned_features.detach().clone()

        preds = torch.argmax(pred, dim=1)
        ACC, P, R, MAF1, MIF1 = get_evaluation_results(label.cpu().detach().numpy(), preds.cpu().detach().numpy())

        self.log("train_accuracy", ACC, on_step=False, on_epoch=True)
        self.log("train_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train_precision", P, on_step=False, on_epoch=True)
        self.log("train_recall", R, on_step=False, on_epoch=True)
        self.log("train_MAF1", MAF1, on_step=False, on_epoch=True)
        self.log("train_MIF1", MIF1, on_step=False, on_epoch=True)

        return total_loss


    def test_step(self, test_batch, batch_idx):
        view_features, relation, p_unlabeled, label_list = test_batch
        p_unlabeled = p_unlabeled[0]
        label_list = label_list[0]

        learn_features = self.forward_features(view_features)
        for RGT in self.RGT_layers:
            learn_features, _ = RGT(learn_features, relation)
        learned_features = self.ReLU(self.output1(learn_features))

        batch_id = p_unlabeled.tolist()
        pred = self.output2(learned_features[batch_id])
        label = torch.LongTensor(label_list[batch_id].tolist())
        preds = torch.argmax(pred, dim=1)
        ACC, P, R, MAF1, MIF1 = get_evaluation_results(label.cpu().detach().numpy(), preds.cpu().detach().numpy())

        self.log("test_accuracy", ACC, on_step=False, on_epoch=True)
        self.log("test_precision", P, on_step=False, on_epoch=True)
        self.log("test_recall", R, on_step=False, on_epoch=True)
        self.log("test_MAF1", MAF1, on_step=False, on_epoch=True)
        self.log("test_MIF1", MIF1, on_step=False, on_epoch=True)

        return {"test_accuracy": ACC, "test_precision": P, "test_recall": R, "test_MAF1": MAF1, "test_MIF1": MIF1}


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=3e-5, amsgrad=False)
        return optimizer


def get_evaluation_results(labels_true, labels_pred):
    ACC = metrics.accuracy_score(labels_true, labels_pred)
    P = metrics.precision_score(labels_true, labels_pred, average='macro', zero_division=0)
    R = metrics.recall_score(labels_true, labels_pred, average='macro', zero_division=0)
    MAF1 = metrics.f1_score(labels_true, labels_pred, average='macro', zero_division=0)
    MIF1 = metrics.f1_score(labels_true, labels_pred, average='micro', zero_division=0)
    return ACC, P, R, MAF1, MIF1






