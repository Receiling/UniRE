#!/bin/bash

mkdir -p ACE2005/tmp

python transfer.py transfer $1/train.json ACE2005/tmp/train.json [PER-SOC]
python transfer.py transfer $1/dev.json ACE2005/tmp/dev.json [PER-SOC]
python transfer.py transfer $1/test.json ACE2005/tmp/test.json [PER-SOC]

python process.py process ACE2005/tmp/train.json ACE2005/ent_rel_file.json ACE2005/train.json bert-base-uncased 200
python process.py process ACE2005/tmp/dev.json ACE2005/ent_rel_file.json ACE2005/dev.json bert-base-uncased 200
python process.py process ACE2005/tmp/test.json ACE2005/ent_rel_file.json ACE2005/test.json bert-base-uncased 200

rm -rf ACE2005/tmp