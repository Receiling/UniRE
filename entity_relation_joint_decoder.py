from collections import defaultdict
import json
import os
import random
import logging
import time

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, AutoTokenizer, AdamW, get_linear_schedule_with_warmup

from utils.argparse import ConfigurationParer
from utils.prediction_outputs import print_predictions_for_joint_decoding
from utils.eval import eval_file
from inputs.vocabulary import Vocabulary
from inputs.fields.token_field import TokenField
from inputs.fields.raw_token_field import RawTokenField
from inputs.fields.map_token_field import MapTokenField
from inputs.instance import Instance
from inputs.datasets.dataset import Dataset
from inputs.dataset_readers.ace_reader_for_joint_decoding import ACEReaderForJointDecoding
from models.joint_decoding.joint_decoder import EntRelJointDecoder
from utils.nn_utils import get_n_trainable_parameters

logger = logging.getLogger(__name__)


def step(cfg, model, batch_inputs, device):
    batch_inputs["tokens"] = torch.LongTensor(batch_inputs["tokens"])
    batch_inputs["joint_label_matrix"] = torch.LongTensor(batch_inputs["joint_label_matrix"])
    batch_inputs["joint_label_matrix_mask"] = torch.BoolTensor(batch_inputs["joint_label_matrix_mask"])
    batch_inputs["wordpiece_tokens"] = torch.LongTensor(batch_inputs["wordpiece_tokens"])
    batch_inputs["wordpiece_tokens_index"] = torch.LongTensor(batch_inputs["wordpiece_tokens_index"])
    batch_inputs["wordpiece_segment_ids"] = torch.LongTensor(batch_inputs["wordpiece_segment_ids"])

    if device > -1:
        batch_inputs["tokens"] = batch_inputs["tokens"].cuda(device=device, non_blocking=True)
        batch_inputs["joint_label_matrix"] = batch_inputs["joint_label_matrix"].cuda(device=device, non_blocking=True)
        batch_inputs["joint_label_matrix_mask"] = batch_inputs["joint_label_matrix_mask"].cuda(device=device,
                                                                                               non_blocking=True)
        batch_inputs["wordpiece_tokens"] = batch_inputs["wordpiece_tokens"].cuda(device=device, non_blocking=True)
        batch_inputs["wordpiece_tokens_index"] = batch_inputs["wordpiece_tokens_index"].cuda(device=device,
                                                                                             non_blocking=True)
        batch_inputs["wordpiece_segment_ids"] = batch_inputs["wordpiece_segment_ids"].cuda(device=device,
                                                                                           non_blocking=True)

    outputs = model(batch_inputs)
    batch_outputs = []
    if not model.training:
        for sent_idx in range(len(batch_inputs['tokens_lens'])):
            sent_output = dict()
            sent_output['tokens'] = batch_inputs['tokens'][sent_idx].cpu().numpy()
            sent_output['span2ent'] = batch_inputs['span2ent'][sent_idx]
            sent_output['span2rel'] = batch_inputs['span2rel'][sent_idx]
            sent_output['seq_len'] = batch_inputs['tokens_lens'][sent_idx]
            sent_output['joint_label_matrix'] = batch_inputs['joint_label_matrix'][sent_idx].cpu().numpy()
            sent_output['joint_label_preds'] = outputs['joint_label_preds'][sent_idx].cpu().numpy()
            sent_output['separate_positions'] = batch_inputs['separate_positions'][sent_idx]
            sent_output['all_separate_position_preds'] = outputs['all_separate_position_preds'][sent_idx]
            sent_output['all_ent_preds'] = outputs['all_ent_preds'][sent_idx]
            sent_output['all_rel_preds'] = outputs['all_rel_preds'][sent_idx]
            batch_outputs.append(sent_output)
        return batch_outputs

    return outputs['element_loss'], outputs['symmetric_loss'], outputs['implication_loss']


def train(cfg, dataset, model):
    logger.info("Training starting...")

    for name, param in model.named_parameters():
        logger.info("{!r}: size: {} requires_grad: {}.".format(name, param.size(), param.requires_grad))

    logger.info("Trainable parameters size: {}.".format(get_n_trainable_parameters(model)))

    parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_layer_lr = {}
    base_lr = cfg.bert_learning_rate
    for i in range(11, -1, -1):
        bert_layer_lr['.' + str(i) + '.'] = base_lr
        base_lr *= cfg.lr_decay_rate

    optimizer_grouped_parameters = []
    for name, param in parameters:
        params = {'params': [param], 'lr': cfg.learning_rate}
        if any(item in name for item in no_decay):
            params['weight_decay_rate'] = 0.0
        else:
            if 'bert' in name:
                params['weight_decay_rate'] = cfg.adam_bert_weight_decay_rate
            else:
                params['weight_decay_rate'] = cfg.adam_weight_decay_rate

        for bert_layer_name, lr in bert_layer_lr.items():
            if bert_layer_name in name:
                params['lr'] = lr
                break

        optimizer_grouped_parameters.append(params)

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(cfg.adam_beta1, cfg.adam_beta2),
                      lr=cfg.learning_rate,
                      eps=cfg.adam_epsilon,
                      weight_decay=cfg.adam_weight_decay_rate,
                      correct_bias=False)

    total_train_steps = (dataset.get_dataset_size("train") + cfg.train_batch_size * cfg.gradient_accumulation_steps -
                         1) / (cfg.train_batch_size * cfg.gradient_accumulation_steps) * cfg.epochs
    num_warmup_steps = int(cfg.warmup_rate * total_train_steps) + 1
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_train_steps)

    last_epoch = 1
    batch_id = 0
    best_f1 = 0.0
    early_stop_cnt = 0
    accumulation_steps = 0
    model.zero_grad()

    for epoch, batch in dataset.get_batch('train', cfg.train_batch_size, None):

        if last_epoch != epoch or (batch_id != 0 and batch_id % cfg.validate_every == 0):
            if accumulation_steps != 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if epoch > cfg.pretrain_epochs:
                dev_f1 = dev(cfg, dataset, model)
                if dev_f1 > best_f1:
                    early_stop_cnt = 0
                    best_f1 = dev_f1
                    logger.info("Save model...")
                    torch.save(model.state_dict(), open(cfg.best_model_path, "wb"))
                elif last_epoch != epoch:
                    early_stop_cnt += 1
                    if early_stop_cnt > cfg.early_stop:
                        logger.info("Early Stop: best F1 score: {:6.2f}%".format(100 * best_f1))
                        break
        if epoch > cfg.epochs:
            torch.save(model.state_dict(), open(cfg.last_model_path, "wb"))
            logger.info("Training Stop: best F1 score: {:6.2f}%".format(100 * best_f1))
            break

        if last_epoch != epoch:
            batch_id = 0
            last_epoch = epoch

        model.train()
        batch_id += len(batch['tokens_lens'])
        batch['epoch'] = (epoch - 1)
        element_loss, symmetric_loss, implication_loss = step(cfg, model, batch, cfg.device)
        loss = 1.0 * element_loss + 1.0 * symmetric_loss + 1.0 * implication_loss
        if batch_id % cfg.logging_steps == 0:
            logger.info(
                "Epoch: {} Batch: {} Loss: {} (Element_loss: {} Symmetric_loss: {} Implication_loss: {})".format(
                    epoch, batch_id, loss.item(), element_loss.item(), symmetric_loss.item(), implication_loss.item()))

        if cfg.gradient_accumulation_steps > 1:
            loss /= cfg.gradient_accumulation_steps

        loss.backward()

        accumulation_steps = (accumulation_steps + 1) % cfg.gradient_accumulation_steps
        if accumulation_steps == 0:
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=cfg.gradient_clipping)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

    state_dict = torch.load(open(cfg.best_model_path, "rb"), map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    test(cfg, dataset, model)


def dev(cfg, dataset, model):
    logger.info("Validate starting...")
    model.zero_grad()

    all_outputs = []
    cost_time = 0
    for _, batch in dataset.get_batch('dev', cfg.test_batch_size, None):
        model.eval()
        with torch.no_grad():
            cost_time -= time.time()
            batch_outpus = step(cfg, model, batch, cfg.device)
            cost_time += time.time()
        all_outputs.extend(batch_outpus)
    logger.info(f"Cost time: {cost_time}s")
    dev_output_file = os.path.join(cfg.save_dir, "dev.output")
    print_predictions_for_joint_decoding(all_outputs, dev_output_file, dataset.vocab)
    eval_metrics = ['joint-label', 'separate-position', 'ent', 'exact-rel']
    joint_label_score, separate_position_score, ent_score, exact_rel_score = eval_file(dev_output_file, eval_metrics)
    return ent_score + exact_rel_score


def test(cfg, dataset, model):
    logger.info("Testing starting...")
    model.zero_grad()

    all_outputs = []

    cost_time = 0
    for _, batch in dataset.get_batch('test', cfg.test_batch_size, None):
        model.eval()
        with torch.no_grad():
            cost_time -= time.time()
            batch_outpus = step(cfg, model, batch, cfg.device)
            cost_time += time.time()
        all_outputs.extend(batch_outpus)
    logger.info(f"Cost time: {cost_time}s")

    test_output_file = os.path.join(cfg.save_dir, "test.output")
    print_predictions_for_joint_decoding(all_outputs, test_output_file, dataset.vocab)
    eval_metrics = ['joint-label', 'separate-position', 'ent', 'exact-rel']
    eval_file(test_output_file, eval_metrics)


def main():
    # config settings
    parser = ConfigurationParer()
    parser.add_save_cfgs()
    parser.add_data_cfgs()
    parser.add_model_cfgs()
    parser.add_optimizer_cfgs()
    parser.add_run_cfgs()

    cfg = parser.parse_args()
    logger.info(parser.format_values())

    # set random seed
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.device > -1 and not torch.cuda.is_available():
        logger.error('config conflicts: no gpu available, use cpu for training.')
        cfg.device = -1
    if cfg.device > -1:
        torch.cuda.manual_seed(cfg.seed)

    # define fields
    tokens = TokenField("tokens", "tokens", "tokens", True)
    separate_positions = RawTokenField("separate_positions", "separate_positions")
    span2ent = MapTokenField("span2ent", "ent_rel_id", "span2ent", False)
    span2rel = MapTokenField("span2rel", "ent_rel_id", "span2rel", False)
    joint_label_matrix = RawTokenField("joint_label_matrix", "joint_label_matrix")
    wordpiece_tokens = TokenField("wordpiece_tokens", "wordpiece", "wordpiece_tokens", False)
    wordpiece_tokens_index = RawTokenField("wordpiece_tokens_index", "wordpiece_tokens_index")
    wordpiece_segment_ids = RawTokenField("wordpiece_segment_ids", "wordpiece_segment_ids")
    fields = [tokens, separate_positions, span2ent, span2rel, joint_label_matrix]

    if cfg.embedding_model in ['bert', 'pretrained']:
        fields.extend([wordpiece_tokens, wordpiece_tokens_index, wordpiece_segment_ids])

    # define counter and vocabulary
    counter = defaultdict(lambda: defaultdict(int))
    vocab = Vocabulary()

    # define instance
    train_instance = Instance(fields)
    dev_instance = Instance(fields)
    test_instance = Instance(fields)

    # define dataset reader
    max_len = {'tokens': cfg.max_sent_len, 'wordpiece_tokens': cfg.max_wordpiece_len}
    ent_rel_file = json.load(open(cfg.ent_rel_file, 'r', encoding='utf-8'))
    pretrained_vocab = {'ent_rel_id': ent_rel_file["id"]}
    if cfg.embedding_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained(cfg.bert_model_name)
        logger.info("Load bert tokenizer successfully.")
        pretrained_vocab['wordpiece'] = tokenizer.get_vocab()
    elif cfg.embedding_model == 'pretrained':
        tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model_name)
        logger.info("Load {} tokenizer successfully.".format(cfg.pretrained_model_name))
        pretrained_vocab['wordpiece'] = tokenizer.get_vocab()
    ace_train_reader = ACEReaderForJointDecoding(cfg.train_file, False, max_len)
    ace_dev_reader = ACEReaderForJointDecoding(cfg.dev_file, False, max_len)
    ace_test_reader = ACEReaderForJointDecoding(cfg.test_file, False, max_len)

    # define dataset
    ace_dataset = Dataset("ACE2005")
    ace_dataset.add_instance("train", train_instance, ace_train_reader, is_count=True, is_train=True)
    ace_dataset.add_instance("dev", dev_instance, ace_dev_reader, is_count=True, is_train=False)
    ace_dataset.add_instance("test", test_instance, ace_test_reader, is_count=True, is_train=False)

    min_count = {"tokens": 1}
    no_pad_namespace = ["ent_rel_id"]
    no_unk_namespace = ["ent_rel_id"]
    contain_pad_namespace = {"wordpiece": tokenizer.pad_token}
    contain_unk_namespace = {"wordpiece": tokenizer.unk_token}
    ace_dataset.build_dataset(vocab=vocab,
                              counter=counter,
                              min_count=min_count,
                              pretrained_vocab=pretrained_vocab,
                              no_pad_namespace=no_pad_namespace,
                              no_unk_namespace=no_unk_namespace,
                              contain_pad_namespace=contain_pad_namespace,
                              contain_unk_namespace=contain_unk_namespace)
    wo_padding_namespace = ["separate_positions", "span2ent", "span2rel"]
    ace_dataset.set_wo_padding_namespace(wo_padding_namespace=wo_padding_namespace)

    if cfg.test:
        vocab = Vocabulary.load(cfg.vocabulary_file)
    else:
        vocab.save(cfg.vocabulary_file)

    # joint model
    model = EntRelJointDecoder(cfg=cfg, vocab=vocab, ent_rel_file=ent_rel_file)

    if cfg.test and os.path.exists(cfg.best_model_path):
        state_dict = torch.load(open(cfg.best_model_path, 'rb'), map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        logger.info("Loading best training model {} successfully for testing.".format(cfg.best_model_path))

    if cfg.device > -1:
        model.cuda(device=cfg.device)

    if cfg.test:
        dev(cfg, ace_dataset, model)
        test(cfg, ace_dataset, model)
    else:
        train(cfg, ace_dataset, model)


if __name__ == '__main__':
    main()
