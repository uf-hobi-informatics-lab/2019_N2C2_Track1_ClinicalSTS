# -*- coding: utf-8 -*-
from transformers.modeling_utils import SequenceSummary
import torch
from torch import nn

from transformers import (
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    XLNetConfig,
    XLNetModel,
    XLNetPreTrainedModel,
    XLNetTokenizer,
    RobertaConfig,
    RobertaModel,
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
)


class BertGetPoolFeatures(BertPreTrainedModel):
    def __init__(self, config):
        super(BertGetPoolFeatures, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            position_ids=position_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            head_mask=head_mask)

        pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        return pooled_output


class XLNetGetPoolFeature(XLNetPreTrainedModel):
    def __init__(self, config):
        super(XLNetGetPoolFeature, self).__init__(config)
        self.num_labels = config.num_labels

        self.transformer = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, input_mask=None, attention_mask=None,
                mems=None, perm_mask=None, target_mapping=None,
                labels=None, head_mask=None):
        transformer_outputs = self.transformer(input_ids,
                                               token_type_ids=token_type_ids,
                                               input_mask=input_mask,
                                               attention_mask=attention_mask,
                                               mems=mems,
                                               perm_mask=perm_mask,
                                               target_mapping=target_mapping,
                                               head_mask=head_mask)
        output = transformer_outputs[0]
        output = self.sequence_summary(output)

        return output


class RobertaGetPoolFeature(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaGetPoolFeature, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        head = sequence_output[:, 0, :]  # get the weights of the first token
        head = self.dropout(head)
        head = self.dense(head)
        head = torch.tanh(head)

        return head


class Ensemble(nn.Module):
    def __init__(self, args, config1, config2, model1, model2, num_labels):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels

        dim = config1.hidden_size + config2.hidden_size
        self.linear1 = nn.Linear(dim, dim // 2)
        self.classifier = nn.Linear(dim // 2, num_labels)

    def forward(self, input1, input2, labels=None):
        output1 = self.model1(**input1)
        output2 = self.model2(**input2)
        weights = torch.cat((output1, output2), 1)

        weights = self.linear1(weights)
        weights = self.dropout(weights)
        logits = self.classifier(weights)

        output = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            output = (loss,) + output

        return output


class EnsembleAll(nn.Module):
    def __init__(self, args, config1, config2, config3, model1, model2, model3, num_labels):
        super(EnsembleAll, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels

        dim = config1.hidden_size + config2.hidden_size + config3.hidden_size
        self.linear1 = nn.Linear(dim, dim // 2)
        self.classifier = nn.Linear(dim // 2, num_labels)

    def forward(self, input1, input2, input3, labels=None):
        output1 = self.model1(**input1)
        output2 = self.model2(**input2)
        output3 = self.model3(**input3)
        weights = torch.cat((output1, output2, output3), 1)

        weights = self.linear1(weights)
        weights = self.dropout(weights)
        logits = self.classifier(weights)

        output = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            output = (loss,) + output

        return output
