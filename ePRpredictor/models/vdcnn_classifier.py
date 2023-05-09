import torch
import torch.nn as nn

import transformers
import torch.nn.functional as F

from .VDCNN import VDCNN

class VDCNNClassifer(torch.nn.Module):
    def __init__(self, args, n_classes=2):
        super(VDCNNClassifer, self).__init__()
        self.vdcnn_desc = VDCNN(n_classes = 768)
        self.vdcnn_diff = VDCNN(n_classes = 768)

        self.args = args

        self.drop = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(768)
        self.relu =  nn.ReLU()

        textual_feature_len = 768
        self.fc_desc = nn.Linear(768, textual_feature_len)
        self.fc_diff = nn.Linear(768, textual_feature_len)
        mid = 30

        self.fc = nn.Linear(textual_feature_len * 2, mid)
        self.feature_bn = nn.BatchNorm1d(46)
        self.feature_layer1 = nn.Linear(46, 768)
        self.feature_fc = nn.Linear(768, 768 // 16)

        self.pr_layer = nn.Linear(768*2 + 768 // 16, 30)
        self.pr_layer2 = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, h, features, true_label, gen_feature=False):

        h1 = h[:, 0]
        h2 = h[:, 1]
        
        h1 = self.vdcnn_desc(h1)
        h2 = self.vdcnn_diff(h2)

        h1 = self.fc_desc(h1)
        h2 = self.fc_diff(h2)

        cla_h = self.relu(torch.cat([h1, h2], 1))

        cla_f = self.feature_layer1(features)
        cla_f = self.feature_fc(cla_f)

        if self.args.phase == 1:
            # In phase 1, we don't train bert.
            cla_h = torch.zeros_like(cla_h).to(cla_h.device)

        cla = self.pr_layer(torch.cat([cla_h, cla_f], 1))
        if gen_feature:
            return cla

        cla = self.pr_layer2(cla)
        loss = nn.BCEWithLogitsLoss()(input=cla.squeeze(-1), target=true_label.float())

        cla = self.sigmoid(cla)

        return {'loss':loss, 'pred':cla}