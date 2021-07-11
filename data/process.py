import json
import fire

from transformers import AutoTokenizer


def add_cross_sentence(sentences, tokenizer, max_length=200):
    """add_cross_sentence add cross sentences with adding equal number of
    left and right context tokens.
    """

    new_sents = []
    all_tokens = []
    sent_lens = []
    last_id = sentences[0]['sentId'] - 1
    article_id = sentences[0]['articleId']

    for s in sentences:
        assert s['articleId'] == article_id
        assert s['sentId'] > last_id
        last_id = s['sentId']
        tokens = s['sentText'].split(' ')
        all_tokens.extend(tokens)
        sent_lens.append(len(tokens))

    cur_pos = 0
    for sent, sent_len in zip(sentences, sent_lens):
        if max_length > sent_len:
            context_len = (max_length - sent_len) // 2
            left_context = all_tokens[max(cur_pos - context_len, 0):cur_pos]
            right_context = all_tokens[cur_pos + sent_len:cur_pos + sent_len + context_len]
        else:
            left_context = []
            right_context = []

        cls = tokenizer.cls_token
        sep = tokenizer.sep_token

        wordpiece_tokens = [cls]
        for token in left_context:
            tokenized_token = list(tokenizer.tokenize(token))
            wordpiece_tokens.extend(tokenized_token)

        for token in right_context:
            tokenized_token = list(tokenizer.tokenize(token))
            wordpiece_tokens.extend(tokenized_token)
        wordpiece_tokens.append(sep)

        context_len = len(wordpiece_tokens)
        wordpiece_segment_ids = [0] * context_len

        wordpiece_tokens_index = []
        cur_index = len(wordpiece_tokens)
        for token in sent['sentText'].split(' '):
            tokenized_token = list(tokenizer.tokenize(token))
            wordpiece_tokens.extend(tokenized_token)
            wordpiece_tokens_index.append([cur_index, cur_index + len(tokenized_token)])
            cur_index += len(tokenized_token)
        wordpiece_tokens.append(sep)

        wordpiece_segment_ids += [1] * (len(wordpiece_tokens) - context_len)

        new_sent = {
            'articleId': sent['articleId'],
            'sentId': sent['sentId'],
            'sentText': sent['sentText'],
            'entityMentions': sent['entityMentions'],
            'relationMentions': sent['relationMentions'],
            'wordpieceSentText': " ".join(wordpiece_tokens),
            'wordpieceTokensIndex': wordpiece_tokens_index,
            'wordpieceSegmentIds': wordpiece_segment_ids
        }
        new_sents.append(new_sent)

        cur_pos += sent_len

    return new_sents


def add_joint_label(sent, ent_rel_id):
    """add_joint_label add joint labels for sentences
    """

    none_id = ent_rel_id['None']
    sentence_length = len(sent['sentText'].split(' '))
    label_matrix = [[none_id for j in range(sentence_length)] for i in range(sentence_length)]
    ent2offset = {}
    for ent in sent['entityMentions']:
        ent2offset[ent['emId']] = ent['offset']
        for i in range(ent['offset'][0], ent['offset'][1]):
            for j in range(ent['offset'][0], ent['offset'][1]):
                label_matrix[i][j] = ent_rel_id[ent['label']]
    for rel in sent['relationMentions']:
        for i in range(ent2offset[rel['em1Id']][0], ent2offset[rel['em1Id']][1]):
            for j in range(ent2offset[rel['em2Id']][0], ent2offset[rel['em2Id']][1]):
                label_matrix[i][j] = ent_rel_id[rel['label']]
    sent['jointLabelMatrix'] = label_matrix


def process(source_file, ent_rel_file, target_file, pretrained_model, max_length=200):
    auto_tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    print("Load {} tokenizer successfully.".format(pretrained_model))

    ent_rel_id = json.load(open(ent_rel_file, 'r', encoding='utf-8'))["id"]

    with open(source_file, 'r', encoding='utf-8') as fin, open(target_file, 'w', encoding='utf-8') as fout:
        sentences = []
        for line in fin:
            sent = json.loads(line.strip())

            if len(sentences) == 0 or sentences[0]['articleId'] == sent['articleId']:
                sentences.append(sent)
            else:
                for new_sent in add_cross_sentence(sentences, auto_tokenizer, max_length):
                    add_joint_label(new_sent, ent_rel_id)
                    print(json.dumps(new_sent), file=fout)
                sentences = [sent]

        for new_sent in add_cross_sentence(sentences, auto_tokenizer, max_length):
            add_joint_label(new_sent, ent_rel_id)
            print(json.dumps(new_sent), file=fout)


if __name__ == '__main__':
    fire.Fire({"process": process})
