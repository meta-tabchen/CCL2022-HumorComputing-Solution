from atc.models import *
from atc.configs import *

model_dict = {}
en_model_dict = {
                 "en_deberta_large":{"model_class":DeBERTa,"config":en_deberta_large_config},
                 "en_deberta_v3_large":{"model_class":DeBERTa,"config":en_deberta_v3_large_config}
                 }

model_dict.update(en_model_dict)

default_model_list = list(model_dict.keys())
