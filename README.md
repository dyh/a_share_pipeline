# A Share Pipeline

- A股预测 Pipeline

### 使用框架：
- https://github.com/AI4Finance-LLC/ElegantRL

### 数据来源：
- baostock.com
- sinajs.cn

## 运行环境

- ubuntu 18.04.5
- python 3.6+，pip 20+
- pytorch 1.7+
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

3. 更新submodule ElegantRL
   
    ```
    $ cd ElegantRL_master/ 
    $ git submodule update --init --recursive
    $ git pull
    $ cd ..
    ```
   
    > 如果git pull 提示没在任何分支上，可以检出ElegantRL的master分支：

    ```
    $ git checkout master
    ```

4. 创建 python 虚拟环境

    ```
    $ python3 -m venv venv
    ```
   
    > 如果提示找不到 venv，则安装venv：

    ```
    $ sudo apt install python3-pip
    $ sudo apt-get install python3-venv
    ```                                        

5. 激活虚拟环境

    ```
    $ source venv/bin/activate
    ```
   
6. 升级pip

    ```
    $ python -m pip install --upgrade pip
    ```

7. 安装pytorch

    > 根据你的操作系统，运行环境工具和CUDA版本，在 https://pytorch.org/get-started/locally/ 找到对应的安装命令，复制粘贴运行。为了方便演示，以下使用CPU运行，安装命令如下：
    
    ```
    $ pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
    ```
   
8. 安装其他软件包

    ```
    $ pip install -r requirements.txt
    ```

    > 如果上述安装出错，可以考虑安装如下：

    ```
    $ sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx
    ```
   
9. 训练模型 sh.600036

    ```
    $ python train_single.py
    ```

10. 预测数据 sh.600036

    ```
    $ python predict_single_sqlite.py
    ```
