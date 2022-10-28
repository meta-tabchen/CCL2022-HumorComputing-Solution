import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d","--df_path",default="data/after_preprocess/task1_charlie_train.csv")
parser.add_argument("-m","--model_name",help="模型名",default="en_deberta_base")
parser.add_argument("-n","--num_labels",help="标签数",default=2)
parser.add_argument("--patience",help="patience",default=2)
parser.add_argument("--epochs",help="epochs",default=200)
parser.add_argument("--up_num",help="up_num",type=int,default=5)
parser.add_argument("--down_num",help="down_num",type=int,default=5)
parser.add_argument("--max_len",help="max len",default=200)
parser.add_argument("--lr",default=1e-5,type=float)
parser.add_argument("--batch_size",help="batch",default=32)
parser.add_argument("--kf_i",help="kf_i",type=int,default=0)
parser.add_argument("--seed",help="seed",type=int,default=0)
parser.add_argument("--refit",type=str,default='auc')
parser.add_argument("--data_seed",help="data_seed",type=int,default=0)
parser.add_argument("--output_dir",help="保存文件夹",default='model/tmp')
args = parser.parse_args()
args = vars(args)

import warnings
warnings.filterwarnings("ignore")
import os
import wandb
import sys
import pandas as pd
sys.path.append('auto_text_classifier')
from atc.models.aml import AML
from utils.data_split import DataGet
import copy
import uuid
from utils.metrics import *
from utils.data_split import add_diff_content
wandb.init()

user_config = copy.deepcopy(args)
user_config['save_dir'] = os.path.join(args['output_dir'],str(uuid.uuid4()),
                                        args['model_name'])


# 数据获取器
data_get = DataGet(user_config['df_path'])
# 初始化AML
ai = AML(save_dir=user_config['save_dir'])
model_class, config = ai.get_model_config(user_config['model_name']) # 获取模型的配置
config.update(user_config) # 更新配置
print("config is :{}".format(config))

model = model_class(config) # 初始化模型
try:
    sep = model.tokenizer.sep_token
except:
    sep = "[SEP]"
df_train, df_dev, df_test = data_get.get_data(kf_i=user_config['kf_i'], 
                    random_state=user_config['data_seed'],
                    up_num=user_config['up_num'], 
                    down_num=user_config['down_num'], sep_token=sep)

df_train, df_dev, df_test = [x for x in [df_train, df_dev, df_test]]

# 第一阶段测试集合（df_final_dev）
df_final_dev = pd.read_csv('data/after_preprocess/task1_charlie_dev.csv')
df_final_dev['label'] = df_final_dev['Label']
df_final_dev = add_diff_content(df_final_dev,up_num=user_config['up_num'],down_num=user_config['down_num'])

# 训练模型
df_report = model.train(df_train, df_dev, df_test)
print(df_report)

# eval

def save_df(df,name,save_dir=user_config["save_dir"]):
    save_path = os.path.join(save_dir,name+'.csv')
    df.to_csv(save_path,index=False)

def reset_dict_k(data,add_str):
    new_data = {}
    for k,v in data.items():
        new_data[f'{k}_{add_str}'] = v
    return new_data

final_report = {"model_save_dir":user_config["save_dir"]}

#dev
df_dev['pred'] = model.predict_list(df_dev['text'])
dev_report = reset_dict_k(get_model_metrics(df_dev['label'], df_dev['pred']),'dev')
final_report.update(dev_report)
save_df(df_dev,"df_dev_with_pred")

#test
df_test['pred'] = model.predict_list(df_test['text'])
test_report = reset_dict_k(get_model_metrics(df_test['label'], df_test['pred']),'test')
final_report.update(test_report)
save_df(df_test,"df_test_with_pred")

#final dev
df_final_dev['pred'] = model.predict_list(df_final_dev['text'])
final_dev_report = reset_dict_k(get_model_metrics(df_final_dev['label'], df_final_dev['pred']),'final_dev')
final_report.update(final_dev_report)
save_df(df_final_dev,"df_final_dev_pred")

# 比赛指标优化
from utils.metrics import search_model

df_test_score = search_model(df_test,['pred'])
best_test_score = df_test_score.sort_values("score",ascending=False).iloc[0][['acc','score','tradeoff']].to_dict()
final_report.update(reset_dict_k(best_test_score,'test'))
save_df(df_test_score,"df_test_score_pred")

#final search
df_final_dev_score = search_model(df_final_dev,['pred'])
best_final_dev_score = df_final_dev_score[df_final_dev_score['tradeoff']==best_test_score['tradeoff']].iloc[0].to_dict()
final_report.update(reset_dict_k(best_final_dev_score,'final_dev'))
save_df(df_final_dev_score,"df_final_dev_score_pred")

wandb.log(final_report)




"""
python aml_shell.py --model_name en_deberta_large --batch_size 40
"""
