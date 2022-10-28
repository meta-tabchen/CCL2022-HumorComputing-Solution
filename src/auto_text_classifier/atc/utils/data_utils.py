import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from transformers.data.processors.utils import InputFeatures
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

def init_dir(dir_path):
    """Create dir if not exists.

    :param dir_path: path
    :type dir_path: str
    """    
    os.makedirs(dir_path,exist_ok=True)

def train_dev_test_split(df, train_size=0.8):
    """
    Split data to train,dev,test. Train_size can be int or float in (0,1).

    Parameters:
        df: df need to split.
        train_size: can be int or float in (0,1).

    Returns:
        df_train: train data
        df_dev: dev data
        df_test: test data
    
    """
    df = df.sample(frac=1, random_state=0).copy()
    if train_size < 1:
        train_size = int(train_size*df.shape[0])
    num = df.shape[0]
    dev_size = (num-train_size)//2
    df_train = df[:train_size]
    df_dev = df[train_size:dev_size+train_size]
    df_test = df[dev_size+train_size:]
    return df_train, df_dev, df_test

def split_3_save_data(save_dir,df,train_size=0.8):
    """Split data to train,dev,test. Than save data to savedir.Train_size can be int or float in (0,1).

    :param save_dir: where to save data
    :type save_dir: str
    :param df: df need to split.
    :type df: dataframe
    :param train_size: can be int or float in (0,1), defaults to 0.8
    :type train_size: float, optional
    :return: df_train,df_dev,df_test
    :rtype: dataframe
    """    

    df_train,df_dev,df_test = train_dev_test_split(df,train_size)
    init_dir(save_dir)
    df_train.to_csv(os.path.join(save_dir,"train.csv"),index=False)
    df_dev.to_csv(os.path.join(save_dir,"dev.csv"),index=False)
    df_test.to_csv(os.path.join(save_dir,"test.csv"),index=False)
    return df_train, df_dev, df_test

def load_data_from_split(data_dir,file_type=''):
    """读取划分好的数据

    :param data_dir: 数据文件夹
    :type data_dir: [str]
    :param file_type: 文件的后缀，defaults to 'csv'
    :type file_type: str, optional
    :return: df_train,df_dev,df_test
    :rtype: dataframe
    """
    for file_type in ['csv','xlsx']+[file_type]:
        try:
            df_train = load_df(os.path.join(data_dir,"train.{}".format(file_type)))
            df_dev = load_df(os.path.join(data_dir,"dev.{}".format(file_type)))
            df_test = load_df(os.path.join(data_dir,"test.{}".format(file_type)))
            break
        except:
            pass
        
    return df_train,df_dev,df_test


def load_df(path):
    """ load dataframe data, support csv/xlsx/pickle path or df object

    :param path: csv/xlsx/pickle path/df object
    :type path: str
    :return: df
    :rtype: df object
    """    
    
    if isinstance(path,str):
        for pd_read_fun in [pd.read_csv,pd.read_excel,pd.read_pickle]:
            try:
                df = pd_read_fun(path)
                break
            except:
                pass
    else:
        df = path
    
    # df['label'] = df['label'].apply(int)
    # df = df.fillna("")
    return df


def get_one_data_report(path, name=""):
    """
    get report of one data

    Parameters:
        path: train_path
        name: data name

    Returns:
        df_data_report:df_data_report
    
    """
    df = load_df(path)
    report = df['label'].value_counts().to_dict()
    report['总量'] = df.shape[0]
    report['数据集'] = name
    raw_report_norm = df['label'].value_counts(normalize=True).to_dict()
    report_norm = {}
    for key, value in raw_report_norm.items():
        report_norm["{}占比".format(key)] = round(value, 3)
    report.update(report_norm)
    return report

def get_data_report(train_path, dev_path, test_path):
    """get report of all data

    :param train_path: train_path
    :type train_path: dataframe
    :param dev_path: dev_path
    :type dev_path: dataframe
    :param test_path: test_path
    :type test_path: dataframe
    :return: df_data_report
    :rtype: dataframe
    """    

    all_report = [get_one_data_report(train_path, "train"),
                  get_one_data_report(dev_path, "dev"),
                  get_one_data_report(test_path, "test")]
    df_data_report = pd.DataFrame(all_report)
    all_cols = df_data_report.columns.tolist()
    head_cols = ["数据集","总量"]
    other_cols = [x for x in all_cols if x not in head_cols]
    df_data_report = df_data_report[head_cols+other_cols]
    return df_data_report

class DataGet():
    """ 实现K折数据读取，模型会返回 df_train, df_dev, df_test

    :param df: 数据
    :type df: dataframe
    :param n_splits: 折数, defaults to 5
    :type n_splits: int, optional
    :param random_state: 划分数据的随机种子, defaults to 5
    :type random_state: int, optional

    """   
    def __init__(self, df, n_splits=5, random_state=5):         
        self.df = df
        self.n_splits = n_splits
        self.random_state = random_state
        self.df['index_cv'] = range(len(self.df))
        ids = self.df['index_cv'].unique()
        self.index_col = 'index_cv'
        self.all_split_info = self.get_split_info(ids, n_splits) 

    def get_split_id(self, all_split_info, kf_i):
        split_info = all_split_info[kf_i]
        train_ids, dev_ids, test_ids = split_info['train_ids'], split_info['dev_ids'], split_info['test_ids']
        return train_ids, dev_ids, test_ids

    def get_split_info(self, ids, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        split_info = {}
        for kf_i, (train_ids, test_ids) in enumerate(kf.split(ids)):
            train_ids, dev_ids = train_test_split(
                train_ids, test_size=0.1, random_state=self.random_state)
            split_info[kf_i] = {"train_ids": list(train_ids), "dev_ids": list(
                dev_ids), "test_ids": list(test_ids)}
        return split_info

    def get_data_index(self, kf_i):
        split_info = self.all_split_info[kf_i]
        train_ids, dev_ids, test_ids = split_info['train_ids'], split_info['dev_ids'], split_info['test_ids']
        return train_ids, dev_ids, test_ids

    def get_index_data(self, ids, sep_token="[SEP]"):
        df_seg = self.df[self.df[self.index_col].isin(ids)].copy()
        return df_seg

    def get_data(self, kf_i, sep_token="[SEP]"):
        train_ids, dev_ids, test_ids = self.get_data_index(
            kf_i=kf_i)
        df_train = self.get_index_data(train_ids, sep_token=sep_token)
        df_dev = self.get_index_data(dev_ids, sep_token=sep_token)
        df_test = self.get_index_data(test_ids, sep_token=sep_token)
        return df_train, df_dev, df_test


class DFDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):   
        dataframe.index = list(range(len(dataframe)))
        if 'label' not in dataframe.columns:
            dataframe['label'] = 0
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.text[index])
        if title.count("[SEP]") == 1:
            text1, text2 = title.split("[SEP]")
        else:
            text1 = title
            text2 = None
        inputs = self.tokenizer.encode_plus(
            text1,
            text2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
                #
        label = self.data.label[index]
        if type(label) == str:
            # 多标签分类label
            label = eval(label)
            label = [float(x) for x in label]
        else:
            # 单标签分类label
            label = int(label)
        #
        feature = InputFeatures(input_ids=inputs['input_ids'], 
                                attention_mask=inputs['attention_mask'], 
                                token_type_ids=inputs['token_type_ids'],
                                label=label)
        return feature
    
    def __len__(self):
        return len(self.data)
