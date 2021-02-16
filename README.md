# A Share Pipeline
- A Share Pipeline with Deep RL
- 强化深度学习A股pipeline


# 运行 DEMO 程序


## 运行环境

- python 3.6+，pip 20+
- pytorch
- pip install -r requirements.txt


## 如何运行

1. 下载代码

    ```
    $ git clone https://github.com/dyh/a_share_pipeline.git
    ```
   
2. 进入目录

    ```
    $ cd a_share_pipeline
    ```

3. 创建 python 虚拟环境

    ```
    $ python3 -m venv venv
    ```

4. 激活虚拟环境

    ```
    $ source venv/bin/activate
    ```
   
5. 升级pip

    ```
    $ python -m pip install --upgrade pip
    ```

6. 安装pytorch

    > 根据你的操作系统，安装工具和CUDA版本，在 https://pytorch.org/get-started/locally/ 找到对应的安装命令。我的环境是 ubuntu 18.04.5、pip 21.0.1、CUDA 11.0，安装命令如下：

    ```
    $ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    ```
   
7. 安装其他软件包

    ```
    $ pip install -r requirements.txt
    ```

    > 如果上述安装出错，可以考虑安装如下：

    ```
    $ sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
    ```
   
8. 运行程序

    ```
    $ python main.py
    ```


# 一、程序开发信息对称

### 开发环境准备

- 操作系统：首选 ubuntu 18.04.5，真机和虚拟机皆可。备选 windows。
- IDE：首选 PyCharm
- python版本：3.6 及以上
- 软件包环境：首选pip，备选conda。


### 基础信息对称

- 尝试跑通 https://github.com/wangshub/RL-Stock 项目，预测单支股票招商银行，以此了解深度强化学习的基本结构和流程。


### A股数据持久化功能

- python操作 baostock.com API 和 mysql数据库
- 输入股票代码，自动增量下载这支股的 日K 数据，以股票代码为表名称，自动创建数据库表，自动增量 insert into 进数据表
- 数据表列，参考如下：date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST
- 具体含义见 baostock.com 的API，注意处理空数据，空转为 int 0 或 float 0.0。


### A股交易手续费功能

- 要求和实际炒股发生的手续费一致。
- 制作买入手续费和卖出手续费函数。
- 输入 买/卖、股票代码、单价和数量，自动计算 抵用开户（买卖）、印花税（卖）、佣金（买卖）、过户费（深市）、代收规费（买卖）等费用，返回此次交易的手续费。
- 参考：https://www.zhihu.com/question/59369402


### 数据集接入功能

- train数据集，从此股上市日期开始，截止到近期3个月。
- test数据集，近期3个月。
- 避免引入明日数据，设 t日=今日，只有开盘价格 open 为t日价格，其余数据，例如「high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM」，都使用 t-1日 的数据。
- 数据集可以接入 FinRL-Library 框架


### 奖惩机制与结束机制功能

- 研究以下3种框架，找出最适合A股的奖惩机制：
    - https://github.com/AI4Finance-LLC/FinRL-Library
    - https://github.com/wangshub/RL-Stock
    - https://www.tensortrade.org/en/latest/index.html


### env接入

- 基于 FinRL-Library 框架制作A股的env


### 算法接入

- 基于 FinRL-Library 框架，将A股数据支持 DQN, Double DQN, DDPG, A2C, SAC, PPO, TD3, GAE, MADDPG, MuZero 等算法


### 训练模型和验证模型

- 基于train数据集，训练A股模型
- 基于test数据集，验证A股模型
- 基于 tensorboard 分析模型训练结果
- 选出最好的 top 10 模型，保存为weights文件，为实际预测做准备

### 真实落地预测功能

> t日

    初始资本10万元，人工买入股票5万元，得到买入价格和买入数量。
    输入1支股票代码，手动输入买入价格和买入数量

> T+1日

    手动输入股票开盘价格，预测今日，买、卖、hold，多少股，多少金额。
    手动交易，按照预测的数量，买、卖、hold 相应数量的股票。
    手动输入 买、卖、hold 价格，买入数量
    
> T+2日

    手动输入股票开盘价格，预测今日，买、卖、hold，多少股，多少金额。
    手动交易，按照预测的数量，买、卖、hold 相应数量的股票。
    手动输入 买、卖、hold 价格，买入数量


# 二、Transformer 分支研究

- 搜集 Transformer 在连续性数值预测和 Deep RL 方面的代码，复现代码
- 将A股数据和env接入复现的项目，对比test结果


# 三、整体思路信息对称

### 做A股的env
1. 账户内资金和持有股票初始值，50%资金，50%股票，不为 0 。
2. 买和卖的手续费，要计算准确。
3. 账户内资金不允许小于0。无法通过账户资金负数，实现买入功能。
4. 惩罚机制，有效。


### 将数据表的数据接入，A股env。
1. 准备训练数据，最近3个月数据，作为test，其余都是train。
2. 实时观察到的数据，是 今日开盘价 和 昨日的相关数据，注意不要引入明日数据。


### 算法和模型的筛选

> 算法层面

1. 训练次数足够多。
2. 不过拟合。
3. tensorboard，注意排除 污染的数据，排除造成 过拟合 的数据。
4. 用Transformer做是最好的选择，因为不会过拟合。但是目前对gpu支持不好？
5. 训练规则：惩罚机制
6. 检测规则：在3个月间内，时间最短，最先达到 30万资金的。股票扣除手续费后的资金，加上账户内原有资金，从10万资金达到30万资金。

> 业务层面

1. 衡量weights好不好，是规避风险，还是投机赚钱。赚的最少的，不见得是最规避风险的。
2. 理论上，每天都有新的weights产生，历史数据+1，weights增量训练+n轮。
3. 对比weights的优劣，卖出全部股票，扣除交易手续费等，合并到账户内的资金，短时间内达到30万资金，前 ？个weights，是胜利者。
4. 检测判定：
成功，在3个月时间内。卖出股票，扣除手续费后的资金，加上账户内原有资金，10万资金达到30万资金。
失败，超过3个月时间。10万资金没有达到30万资金。


### 落地方式

> 从后往前推：

- 输入1支股票代码，自动下载数据，增量，存入数据库。
- 初始资本10万元，人工买入股票5万元。得到买入价格和买入数量。
- 手动输入买入价格和买入数量，程序自动将价格和数量转化为股份，存入数据库，另，其余资本仍然保存在账户里，更新数据库。
- 因为是T+1，今日结束，明日正式开始预测。

> 等待T+1日：

1. 手动输入股票开盘价格，初始化env，预测今日，买、卖、hold，多少股，多少金额。
2. 手动交易，按照预测的数量，买、卖、hold 相应数量的股票。
3. 手动输入 买、卖、hold 价格，买入数量，程序自动计算，将资金金额和股票金额，保存数据库。

> 等待T+2日：

1. 手动输入股票开盘价格，初始化env，预测今日，买、卖、hold，多少股，多少金额。
2. 手动交易，按照预测的数量，买、卖、hold 相应数量的股票。
3. 手动输入 买、卖、hold 价格，买入数量，程序自动计算，将资金金额和股票金额，保存数据库。

### 参考框架：

- https://github.com/AI4Finance-LLC/FinRL-Library
- https://github.com/wangshub/RL-Stock
- https://www.tensortrade.org/en/latest/index.html

### 数据来源：
- http://baostock.com/

### 其他信息
- 模拟炒股。
- 有可能，所有weights的第一天，都是买入。因为如果不投币，连参加游戏的机会都没有，更不用谈胜利了。
- 获得今日开盘价格。准备今日数据。用胜利者weights进行推理1天。会得到若干不同的结果。
- 根据推理结果，人工选择一个动作。记录入数据库。永久更改env初始值，即账户内资金和持有多少股票。