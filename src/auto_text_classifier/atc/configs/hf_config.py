import os
base_path = os.path.dirname(os.path.realpath(__file__))
from os.path import join

## en_deberta_large
en_deberta_large_config = {"model_dir": 'microsoft/deberta-large',
                            "save_dir": 'model/en_deberta_large'}

## en_deberta_v3_large
en_deberta_v3_large_config = {"model_dir":'microsoft/deberta-v3-large',
                             "save_dir": 'model/en_deberta_v3_large'}