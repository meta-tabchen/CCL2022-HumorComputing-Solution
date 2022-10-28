# CCL2022 第四届“小牛杯”幽默计算 冠军方案

[官方GitHub](https://github.com/DUTIRSentiment/CCL2022-HumorComputing) / [比赛主页](http://cips-cl.org/static/CCL2022/cclEval/humorcomputation/index.html)

## 环境安装

以下以`humor`举例

1、安装虚拟环境
```sh
conda create --name=humor python=3.7.5
source activate humor
```

>确保当前环境是`humor`

2、安装依赖

第一步：
`conda install tensorflow-gpu==1.13.1  cudatoolkit=10.0.130=0`

第二步:

`pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`

然后

`pip install -r requirements_gpu.txt`

## 任务1

### 思路
任务1和 [第二届“小牛杯”幽默计算——情景喜剧笑点识别](http://cips-cl.org/static/CCL2020/humorcomputation.html) 一模一样。我们队伍是当时的第一名，本次方案和之前一致。方案细节可以参考 [2020 中国计算语言学大会幽默计算评测冠军方案](https://blog.tabchen.com/ccl2020-humor/)。本次最终只融合和两个模型，且只进行了概率平均。

### 预处理
从头到尾执行 `task1数据处理.ipynb`

### 训练
训练框架使用了ATC (auto_text_classifier), 一个自研的自动文本分类框架，可以非常容易实现各种文本分类算法。目前支持30+模型，10+词向量。本次开源的只包含了和比赛相关的内容。完整开源代码后续会开源。

最终使用的模型训练代码如下, 首先进入`src`，然后执行下面进行训练。

```python
#model 1
python aml_shell.py --batch_size=32 --down_num=2 --kf_i=0 --lr=5e-06 --max_len=256 --model_name=en_deberta_v3_large --output_dir=model/task1/cv5 --up_num=6
#model 2
python aml_shell.py --batch_size=32 --down_num=4 --kf_i=0 --lr=5e-06 --max_len=256 --model_name=en_deberta_large --output_dir=model/task1/cv5 --up_num=6
```


### 推理

参考 `task1 模型融合-推理.ipynb`

## 任务2

### 思路
基于 [EleutherAI/gpt-neo-1.3B](https://huggingface.co/EleutherAI/gpt-neo-1.3B)，在全部字幕语料上微调语言模型。对于每条数据，生成多个下一句话，利用 rerank 挑选最终的句子。

### 预处理

从头到尾执行 `task2数据处理.ipynb`, 得到`data/after_preprocess/all_text.txt` 为所有字幕的文本。

### 训练
`task2 gpt.ipynb` 中的训练部分。

### 推理
`task2 gpt.ipynb` 中的推理部分。rerank 在 `task2 rerank.ipynb`