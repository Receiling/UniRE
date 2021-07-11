import torch.nn as nn

from modules.token_embedders.pretrained_encoder import PretrainedEncoder
from utils.nn_utils import batched_index_select, gelu


class PretrainedEmbedModel(nn.Module):
    """This class acts as an embeddding layer with pre-trained model
    """
    def __init__(self, cfg, vocab):
        """This function constructs `PretrainedEmbedModel` components and
        sets `PretrainedEmbedModel` parameters

        Arguments:
            cfg {dict} -- config parameters for constructing multiple models
            vocab {Vocabulary} -- vocabulary
        """

        super().__init__()
        self.activation = gelu
        self.pretrained_encoder = PretrainedEncoder(pretrained_model_name=cfg.pretrained_model_name,
                                                    trainable=cfg.fine_tune,
                                                    output_size=cfg.bert_output_size,
                                                    activation=self.activation,
                                                    dropout=cfg.bert_dropout)
        self.encoder_output_size = self.pretrained_encoder.get_output_dims()

    def forward(self, batch_inputs):
        """This function propagetes forwardly

        Arguments:
            batch_inputs {dict} -- batch input data
        """

        if 'wordpiece_segment_ids' in batch_inputs:
            batch_seq_pretrained_encoder_repr, batch_cls_repr = self.pretrained_encoder(
                batch_inputs['wordpiece_tokens'], batch_inputs['wordpiece_segment_ids'])
        else:
            batch_seq_pretrained_encoder_repr, batch_cls_repr = self.pretrained_encoder(
                batch_inputs['wordpiece_tokens'])

        batch_seq_tokens_encoder_repr = batched_index_select(batch_seq_pretrained_encoder_repr,
                                                             batch_inputs['wordpiece_tokens_index'])

        batch_inputs['seq_encoder_reprs'] = batch_seq_tokens_encoder_repr
        batch_inputs['seq_cls_repr'] = batch_cls_repr

    def get_hidden_size(self):
        """This function returns embedding dimensions
        
        Returns:
            int -- embedding dimensitons
        """

        return self.encoder_output_size
