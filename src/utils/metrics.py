import os
import sys
import pandas as pd
sys.path.append('/share/tabchen/auto_text_classifier')
sys.path.append('/root/autodl-nas/tabchen/auto_text_classifier/')
from atc.models.aml import AML
from atc.utils.metrics_utils import get_model_metrics
from tqdm import tqdm_notebook
import numpy as np

def get_dialogue_acc(df, pred_col, tradeoff):
    df['true'] = (df[pred_col] > tradeoff) == df['label']
    acc = df.groupby("Dialogue_id").mean()['true'].mean()
    return acc


def get_final_score(df, pred_col, tradeoff):
    result = {}
    acc = get_dialogue_acc(df, pred_col, tradeoff)
    result = get_model_metrics(df['label'], df[pred_col],
                               tradeoff=tradeoff)
    score = result['F_measure']+acc
    result['acc'] = acc
    result['score'] = score
    return result


def search_model(df_report, model_list):
    result_list = []
    for model in tqdm_notebook(model_list):
        for tradeoff in np.arange(0, 1.1, 0.01):
            result = get_final_score(df_report, model, tradeoff=tradeoff)
            result['model'] = model
            result['tradeoff'] = round(tradeoff,3)
            result_list.append(result)

    df_score = pd.DataFrame(result_list)
    df_score = df_score.sort_values("score", ascending=False)
    return df_score


def eval_all_result(data_dir, save_dir, data_set, model_list, config={}):
    if type(data_dir) == list:
        train_path, dev_path, test_path = data_dir
    else:
        train_path = os.path.join(data_dir, 'train.csv')
        dev_path = os.path.join(data_dir, 'dev.csv')
        test_path = os.path.join(data_dir, 'test.csv')

    # 获取模型评价结果
    ai = AML(save_dir=save_dir, config=config)
    df_report_list = ai.get_list_result(
        [train_path, dev_path, test_path], model_list=model_list)
    df_score_list = [search_model(x, model_list) for x in df_report_list]
    return df_report_list, df_score_list


def get_match_reuslt(data_dir, save_dir, model_list, lang='cn', config={}):
    df_report_list, df_score_list = eval_all_result(
        data_dir, save_dir, lang, model_list, config=config)
    return df_report_list, df_score_list


def refit_score(df_score_list):
    df_train_score, df_dev_score, df_score = [df.sort_values(
        ['model', 'tradeoff']) for df in df_score_list]
    df_score['train_score'] = df_train_score['score'] / \
        df_train_score['score'].max()
    df_score['dev_score'] = df_dev_score['score']/df_dev_score['score'].max()
    df_score['test_score'] = df_score['score']/df_score['score'].max()
    df_score['final_score'] = df_score[[
        'train_score', 'dev_score', 'test_score']].mean(axis=1)
    df_score = df_score.sort_values('test_score', ascending=False)  # dev的最接近
    return df_score

