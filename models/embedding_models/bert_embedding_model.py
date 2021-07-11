import torch.nn as nn

from modules.token_embedders.bert_encoder import BertEncoder
from utils.nn_utils import batched_index_select, gelu


class BertEmbedModel(nn.Module):
    """This class acts as an embeddding layer with bert model
    """
    def __init__(self, cfg, vocab):
        """This function constructs `BertEmbedModel` components and
        sets `BertEmbedModel` parameters

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
            vocab {Vocabulary} -- vocabulary
        """

        super().__init__()
        self.activation = gelu
        self.bert_encoder = BertEncoder(bert_model_name=cfg.bert_model_name,
                                        trainable=cfg.fine_tune,
                                        output_size=cfg.bert_output_size,
                                        activation=self.activation,
                                        dropout=cfg.bert_dropout)
        self.encoder_output_size = self.bert_encoder.get_output_dims()

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch input data
        """

        if 'wordpiece_segment_ids' in batch_inputs:
            batch_seq_bert_encoder_repr, batch_cls_repr = self.bert_encoder(
                batch_inputs['wordpiece_tokens'], batch_inputs['wordpiece_segment_ids'])
        else:
            batch_seq_bert_encoder_repr, batch_cls_repr = self.bert_encoder(
                batch_inputs['wordpiece_tokens'])

        batch_seq_tokens_encoder_repr = batched_index_select(batch_seq_bert_encoder_repr,
                                                             batch_inputs['wordpiece_tokens_index'])

        batch_inputs['seq_encoder_reprs'] = batch_seq_tokens_encoder_repr
        batch_inputs['seq_cls_repr'] = batch_cls_repr

    def get_hidden_size(self):
        """This function returns embedding dimensions
        
        Returns:
            int -- embedding dimensitons
        """

        return self.encoder_output_size
