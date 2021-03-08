import torch
from transformers import BertModel

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'


class SarcasmDetector(torch.nn.Module):

    def __init__(self, n_classes, pretrained_bert=None, name='sarcasm_datasets', text_col_name='comment'):
        super(SarcasmDetector, self).__init__()
        if pretrained_bert is not None:
            self.bert = pretrained_bert
        else:
            self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.name = name
        self.text_col_name = text_col_name

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(bert_out['pooler_output'])
        return self.out(output)


class DuelBertSarcasmDetector(torch.nn.Module):

    def __init__(self, n_classes, comment_pretrained_bert=None, parent_pretrained_bert=None, name='sarcasm_datasets', text_col_name='comment'):
        super(DuelBertSarcasmDetector, self).__init__()
        if comment_pretrained_bert is not None:
            self.comment_bert = comment_pretrained_bert
        else:
            self.comment_bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        if parent_pretrained_bert is not None:
            self.parent_bert = parent_pretrained_bert
        else:
            self.parent_bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = torch.nn.Dropout(p=0.3)
        self.ll1 = torch.nn.Linear(self.comment_bert.config.hidden_size + self.parent_bert.config.hidden_size, 256)
        self.ll2 = torch.nn.Linear(256, 64)
        self.out = torch.nn.Linear(64, n_classes)
        self.name = name
        self.text_col_name = text_col_name

    def forward(self, comment_input_ids, comment_attention_mask, parent_input_ids, parent_attention_mask):
        comment_bert_out = self.comment_bert(
            input_ids=comment_input_ids,
            attention_mask=comment_attention_mask
        )
        parent_bert_out = self.parent_bert(
            input_ids=parent_input_ids,
            attention_mask=parent_attention_mask
        )
        berts_pooler_output = torch.cat([comment_bert_out['pooler_output'], parent_bert_out['pooler_output']], dim=1)
        output = self.drop(berts_pooler_output)
        output = self.ll1(output)
        output = self.ll2(output)
        return self.out(output)


class SentimentAnalyzer(torch.nn.Module):

    def __init__(self, n_classes):
        super(SentimentAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.name = "sentiment_tweets"
        self.text_col_name = 'tweet'

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(bert_out['pooler_output'])
        return self.out(output)