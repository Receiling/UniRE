import json
import fire


def transfer(source_file, target_file, symmetric_rels):
    with open(source_file, 'r') as fin, open(target_file, 'w') as fout:
        sent_cnt = 0
        for line in fin:
            doc = json.loads(line.strip())
            offset = 0
            for sent_id, (sent, ents, rels) in enumerate(zip(doc['sentences'], doc['ner'], doc['relations'])):
                sent_cnt += 1
                entity_mentions = []
                ent2id = {}
                for ent in ents:
                    ent2id[(ent[0], ent[1])] = "E{}".format(len(ent2id) + 1)
                    entity_mentions.append({
                        "emId": ent2id[(ent[0], ent[1])],
                        "text": ' '.join(sent[ent[0] - offset:ent[1] - offset + 1]),
                        "offset": [ent[0] - offset, ent[1] - offset + 1],
                        "label": ent[2]
                    })
                relation_mentions = []
                for rel in rels:
                    relation_mentions.append({
                        "em1Id": ent2id[(rel[0], rel[1])],
                        "em1Text": ' '.join(sent[rel[0] - offset:rel[1] - offset + 1]),
                        "em2Id": ent2id[(rel[2], rel[3])],
                        "em2Text": ' '.join(sent[rel[2] - offset:rel[3] - offset + 1]),
                        "label": rel[4]
                    })
                    if rel[4] in symmetric_rels:
                        relation_mentions.append({
                            "em1Id": ent2id[(rel[2], rel[3])],
                            "em1Text": ' '.join(sent[rel[2] - offset:rel[3] - offset + 1]),
                            "em2Id": ent2id[(rel[0], rel[1])],
                            "em2Text": ' '.join(sent[rel[0] - offset:rel[1] - offset + 1]),
                            "label": rel[4]
                        })
                new_sent = {
                    "sentId": sent_id,
                    "articleId": doc["doc_key"],
                    "sentText": ' '.join(sent),
                    "entityMentions": entity_mentions,
                    "relationMentions": relation_mentions
                }
                print(json.dumps(new_sent), file=fout)
                offset += len(sent)

        print("the number of sentences: {}".format(sent_cnt))


if __name__ == '__main__':
    fire.Fire({'transfer': transfer})
