#!/bin/bash

mkdir -p ACE2004/tmp/fold1

python transfer.py transfer $1/fold1/train.json ACE2004/tmp/fold1/train.json [PER-SOC]
python transfer.py transfer $1/fold1/dev.json ACE2004/tmp/fold1/dev.json [PER-SOC]
python transfer.py transfer $1/fold1/test.json ACE2004/tmp/fold1/test.json [PER-SOC]

python process.py process ACE2004/tmp/fold1/train.json ACE2004/fold1/ent_rel_file.json ACE2004/fold1/train.json bert-base-uncased 200
python process.py process ACE2004/tmp/fold1/dev.json ACE2004/fold1/ent_rel_file.json ACE2004/fold1/dev.json bert-base-uncased 200
python process.py process ACE2004/tmp/fold1/test.json ACE2004/fold1/ent_rel_file.json ACE2004/fold1/test.json bert-base-uncased 200

rm -rf ACE2004/tmp/fold1