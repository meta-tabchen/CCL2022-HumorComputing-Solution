from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def map_2_original_id(index,ids):
    return [ids[x] for x in index]
    
def get_split_info(ids, n_splits=10, max_random_state=5):
    all_split_info = {}
    for random_state in range(max_random_state):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_info = {}
        for kf_i, (train_ids, test_ids) in enumerate(kf.split(ids)):
            train_ids, dev_ids = train_test_split(train_ids, test_size=0.1,random_state=random_state)
            split_info[kf_i] = {"train_ids": map_2_original_id(train_ids,ids), 
                                "dev_ids": map_2_original_id(dev_ids,ids),
                                "test_ids": map_2_original_id(test_ids,ids)}
        all_split_info[random_state] = split_info
    return all_split_info

def get_split_id(all_split_info,kf_i,random_state=0):
    split_info = all_split_info[random_state][kf_i]
    train_ids, dev_ids,test_ids = split_info['train_ids'],split_info['dev_ids'],split_info['test_ids']
    return train_ids, dev_ids,test_ids

def get_window_text(text_list,up_num,down_num):
    """获取上下文

    Args:
        text_list (_type_): _description_
        up_num (_type_): _description_
        down_num (_type_): _description_

    Returns:
        _type_: _description_
    """
    content_list = []
    for i in range(len(text_list)):
        i_start = max(0,i-up_num)
        i_end = i+down_num+1
        content = "|".join(text_list[i_start:i_end])
        content_list.append(content)
    return content_list

def add_diff_content(df,up_num,down_num,sep_token="[SEP]"):
    """增加上下文

    Args:
        df (_type_): _description_
        up_num (_type_): _description_
        down_num (_type_): _description_
        sep_token (str, optional): _description_. Defaults to "[SEP]".

    Returns:
        _type_: _description_
    """
    df = df.copy()
    df['text_user_info'] = df['Speaker'] + ":"+df['Sentence']
    for dialogue_id, group in df.groupby("Dialogue_id"):
        text_list = group['text_user_info'].tolist()
        content_list = get_window_text(text_list,up_num,down_num)
        df.loc[df['Dialogue_id'] == dialogue_id, 'content'] = content_list
    df['text'] = df['text_user_info']+sep_token+df['content']
    return df

class DataGet():
    """_summary_
    """
    def __init__(self, df_path, n_splits=5, max_random_state=5, pair_max_len=300):
        self.df = pd.read_csv(df_path)
        self.df['content'] = ""
        self.df['label'] = self.df['Label']
        ids = self.df['Dialogue_id'].unique()
        self.all_split_info = get_split_info(ids, n_splits, max_random_state)
        self.pair_max_len = pair_max_len
  
    def get_data_index(self, kf_i, random_state):
        split_info = self.all_split_info[random_state][kf_i]
        train_ids, dev_ids, test_ids = split_info['train_ids'], split_info['dev_ids'], split_info['test_ids']
        return train_ids, dev_ids, test_ids

    def get_index_data(self, ids,up_num=-1, down_num=-1, sep_token="[SEP]"):
        df_seg = self.df[self.df['Dialogue_id'].isin(ids)].copy()
        df_seg.index = range(len(df_seg))
        df_seg = add_diff_content(df_seg, up_num, down_num, sep_token=sep_token)
        return df_seg

    def get_data(self, kf_i, random_state, up_num=-1, down_num=-1, sep_token="[SEP]"):
        train_ids, dev_ids, test_ids = self.get_data_index(kf_i=kf_i, random_state=random_state)
        df_train = self.get_index_data(
            train_ids, up_num=up_num, down_num=down_num, sep_token=sep_token)
        df_dev = self.get_index_data(
            dev_ids, up_num=up_num, down_num=down_num, sep_token=sep_token)
        df_test = self.get_index_data(
            test_ids, up_num=up_num, down_num=down_num, sep_token=sep_token)
        return df_train, df_dev, df_test
