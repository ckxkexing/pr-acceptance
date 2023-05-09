import torch
import torch.nn as nn
from transformers import (RobertaTokenizer, RobertaConfig, 
                        RobertaModel, RobertaForSequenceClassification, AdamW)
import transformers
import torch.nn.functional as F

class CodeBERTClassifer(torch.nn.Module):
    def __init__(self, args, text_tokenizer_len = None, diff_tokenizer_len = None, n_classes=2):
        super(CodeBERTClassifer, self).__init__()
        config = RobertaConfig.from_pretrained(args.desc_backbone)
        config.num_labels = n_classes
        # self.transformer = RobertaForSequenceClassification.from_pretrained(backbone, config = config)
        self.transformer = RobertaModel.from_pretrained(args.desc_backbone, config = config)
        self.transformer2 = RobertaModel.from_pretrained(args.diff_backbone, config = config)
        
        self.args = args

        if text_tokenizer_len is not None:
            self.transformer.resize_token_embeddings(text_tokenizer_len)
        if diff_tokenizer_len is not None:
            self.transformer2.resize_token_embeddings(diff_tokenizer_len)
        for param in self.transformer.parameters() :
            param.requires_grad = True
        for param in self.transformer2.parameters():
            param.requires_grad = True

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

        # In phase 2, we don't change feature layer.
        # if self.args.phase == 2:
        #     for param in self.feature_layer1.parameters():
        #         param.requires_grad = False
        #     for param in self.feature_layer1.parameters():
        #         param.requires_grad = False

        self.pr_layer = nn.Linear(768*2 + 768 // 16, 30)
        self.pr_layer2 = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids, mask, token_type_ids, features, true_label, gen_feature=False):
        # print(features.shape) [:, 44]
        # For Ablation Experiments
        args = self.args
        if args.hide != 'hide_desc':
            h = self.transformer(ids[:, 0], attention_mask=mask[:, 0], token_type_ids=token_type_ids[:, 0])
            h = h['pooler_output']

        if args.hide != 'hide_diff' :
            h2 = self.transformer2(ids[:, 1], attention_mask=mask[:, 1], token_type_ids=token_type_ids[:, 1])
            h2 = h2['pooler_output']   

        if args.hide == 'hide_desc':
            h = h2
        if args.hide == 'hide_diff':
            h2 = h

        h = self.fc_desc(h)
        h2 = self.fc_diff(h2)
        cla_h = self.relu(torch.cat([h, h2], 1))
        # cla = self.fc(cla_h)    # 30

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