import json
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ACEReaderForJointDecoding():
    """Define text data reader and preprocess data for entity relation
    joint decoding on ACE dataset.
    """
    def __init__(self, file_path, is_test=False, max_len=dict()):
        """This function defines file path and some settings
        
        Arguments:
            file_path {str} -- file path

        Keyword Arguments:
            is_test {bool} -- indicate training or testing (default: {False})
            max_len {dict} -- max length for some namespace (default: {dict()})
        """

        self.file_path = file_path
        self.is_test = is_test
        self.max_len = dict(max_len)
        self.seq_lens = defaultdict(list)

    def __iter__(self):
        """Generator function
        """

        with open(self.file_path, 'r') as fin:
            for line in fin:
                line = json.loads(line)
                sentence = {}

                state, results = self.get_tokens(line)
                self.seq_lens['tokens'].append(len(results['tokens']))
                if not state or ('tokens' in self.max_len and len(results['tokens']) > self.max_len['tokens']
                                 and not self.is_test):
                    if not self.is_test:
                        continue
                sentence.update(results)

                state, results = self.get_wordpiece_tokens(line)
                self.seq_lens['wordpiece_tokens'].append(len(results['wordpiece_tokens']))
                if not state or ('wordpiece_tokens' in self.max_len
                                 and len(results['wordpiece_tokens']) > self.max_len['wordpiece_tokens']):
                    if not self.is_test:
                        continue
                sentence.update(results)

                if len(sentence['tokens']) != len(sentence['wordpiece_tokens_index']):
                    logger.error(
                        "article id: {} sentence id: {} wordpiece_tokens_index length is not equal to tokens.".format(
                            line['articleId'], line['sentId']))
                    continue

                if len(sentence['wordpiece_tokens']) != len(sentence['wordpiece_segment_ids']):
                    logger.error(
                        "article id: {} sentence id: {} wordpiece_tokens length is not equal to wordpiece_segment_ids.".
                        format(line['articleId'], line['sentId']))
                    continue

                state, results = self.get_entity_relation_label(line, len(sentence['tokens']))
                for key, result in results.items():
                    self.seq_lens[key].append(len(result))
                    if key in self.max_len and len(result) > self.max_len[key]:
                        state = False
                if not state:
                    continue
                sentence.update(results)

                yield sentence

    def get_tokens(self, line):
        """This function splits text into tokens

        Arguments:
            line {dict} -- text

        Returns:
            bool -- execute state
            dict -- results: tokens
        """

        results = {}

        if 'sentText' not in line:
            logger.error("article id: {} sentence id: {} doesn't contain 'sentText'.".format(
                line['articleId'], line['sentId']))
            return False, results

        results['text'] = line['sentText']

        if 'tokens' in line:
            results['tokens'] = line['tokens']
        else:
            results['tokens'] = line['sentText'].strip().split(' ')

        return True, results

    def get_wordpiece_tokens(self, line):
        """This function splits wordpiece text into wordpiece tokens

        Arguments:
            line {dict} -- text

        Returns:
            bool -- execute state
            dict -- results: tokens
        """

        results = {}

        if 'wordpieceSentText' not in line or 'wordpieceTokensIndex' not in line or 'wordpieceSegmentIds' not in line:
            logger.error(
                "article id: {} sentence id: {} doesn't contain 'wordpieceSentText' or 'wordpieceTokensIndex' or 'wordpieceSegmentIds'."
                .format(line['articleId'], line['sentId']))
            return False, results

        wordpiece_tokens = line['wordpieceSentText'].strip().split(' ')
        results['wordpiece_tokens'] = wordpiece_tokens
        results['wordpiece_tokens_index'] = [span[0] for span in line['wordpieceTokensIndex']]
        results['wordpiece_segment_ids'] = list(line['wordpieceSegmentIds'])

        return True, results

    def get_entity_relation_label(self, line, sentence_length):
        """This function constructs mapping relation from span to entity label
        and span pair to relation label, and joint entity relation label matrix.

        Arguments:
            line {dict} -- text
            sentence_length {int} -- sentence length

        Returns:
            bool -- execute state
            dict -- ent2rel: entity span mapping to entity label,
            span2rel: two entity span mapping to relation label,
            joint_label_matrix: joint entity relation label matrix
        """

        results = {}

        if 'entityMentions' not in line:
            logger.error("article id: {} sentence id: {} doesn't contain 'entityMentions'.".format(
                line['articleId'], line['sentId']))
            return False, results

        entity_pos = [0] * sentence_length
        idx2ent = {}
        span2ent = {}

        separate_positions = []
        for entity in line['entityMentions']:
            st, ed = entity['offset']
            if st != 0:
                separate_positions.append(st - 1)
            if ed != sentence_length:
                separate_positions.append(ed - 1)
            idx2ent[entity['emId']] = ((st, ed), entity['text'])
            if st >= ed or st < 0 or st > sentence_length or ed < 0 or ed > sentence_length:
                logger.error("article id: {} sentence id: {} offset error'.".format(line['articleId'], line['sentId']))
                return False, results

            span2ent[(st, ed)] = entity['label']

            j = 0
            for i in range(st, ed):
                if entity_pos[i] != 0:
                    logger.error("article id: {} sentence id: {} entity span overlap.".format(
                        line['articleId'], line['sentId']))
                    return False, results
                entity_pos[i] = 1
                j += 1

        results['separate_positions'] = sorted(separate_positions)
        results['span2ent'] = span2ent

        if 'relationMentions' not in line:
            logger.error("article id: {} sentence id: {} doesn't contain 'relationMentions'.".format(
                line['articleId'], line['sentId']))
            return False, results

        span2rel = {}
        for relation in line['relationMentions']:
            if relation['em1Id'] not in idx2ent or relation['em2Id'] not in idx2ent:
                logger.error("article id: {} sentence id: {} entity not exists .".format(
                    line['articleId'], line['sentId']))
                continue

            entity1_span, entity1_text = idx2ent[relation['em1Id']]
            entity2_span, entity2_text = idx2ent[relation['em2Id']]

            if entity1_text != relation['em1Text'] or entity2_text != relation['em2Text']:
                logger.error("article id: {} sentence id: {} entity text doesn't match realtiaon text.".format(
                    line['articleId'], line['sentId']))
                return False, None

            span2rel[(entity1_span, entity2_span)] = relation['label']

        results['span2rel'] = span2rel

        if 'jointLabelMatrix' not in line:
            logger.error("article id: {} sentence id: {} doesn't contain 'jointLabelMatrix'.".format(
                line['articleId'], line['sentId']))
            return False, results

        results['joint_label_matrix'] = line['jointLabelMatrix']

        return True, results

    def get_seq_lens(self):
        return self.seq_lens
