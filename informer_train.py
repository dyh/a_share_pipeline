import sys

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('./FinRL_Library_master')

if 'Informer2020_main' not in sys.path:
    sys.path.append('./Informer2020_main')

import os
import time

import torch

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stock_data import StockData
from torch.utils.data import DataLoader

from Informer2020_main.utils.tools import dotdict
from Informer2020_main.models.model import Informer
from pipeline_informer.exp_informer import Exp_Informer
from pipeline_informer.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
import pipeline_utils.datetime


# freq = [t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

def train(freq='h', features='S', patience=5, use_technical_indicator=False,
          download_data=False):

    stock_code = 'sh.600036'
    date_start = '2002-05-01'
    # date_end = '2021-03-19'
    date_end = pipeline_utils.datetime.get_today_date()

    if download_data is True:
        sd = StockData(output_dir='./pipeline_informer/temp_dataset', date_start=date_start, date_end=date_end)
        sd.get_informer_data(stock_code=stock_code, fields=sd.fields_minutes,
                             frequency='15', adjustflag='3', use_technical_indicator=use_technical_indicator)
    pass

    # 保存文件的时间点
    time_point = pipeline_utils.datetime.time_point()

    args = dotdict()

    args.model = 'informer'  # model of experiment, options: [informer, informerstack, informerlight(TBD)]

    args.data = 'ETTh1'  # data
    args.root_path = './pipeline_informer/temp_dataset/'  # root path of data file
    args.data_path = 'ETTh1.csv'  # data file
    args.features = features  # forecasting task, options:[M, S, MS(TBD)]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    # args.features = 'S'  # forecasting task, options:[M, S, MS(TBD)]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
    args.target = 'OT'  # target feature in S or MS task
    args.freq = freq  # freq for time features encoding

    args.seq_len = 96  # input sequence length of Informer encoder
    args.label_len = 48  # start token length of Informer decoder
    args.pred_len = 240  # prediction sequence length
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    args.enc_in = 7  # encoder input size
    args.dec_in = 7  # decoder input size
    args.c_out = 7  # output size
    args.factor = 5  # probsparse attn factor
    args.d_model = 512  # dimension of model
    args.n_heads = 8  # num of heads
    args.e_layers = 3  # num of encoder layers
    args.d_layers = 2  # num of decoder layers
    args.d_ff = 512  # dimension of fcn in model
    args.dropout = 0.05  # dropout
    args.attn = 'prob'  # attention used in encoder, options:[prob, full]
    args.embed = 'timeF'  # time features encoding, options:[timeF, fixed, learned]
    args.activation = 'gelu'  # activation
    args.distil = True  # whether to use distilling in encoder
    args.output_attention = False  # whether to output attention in ecoder

    args.batch_size = 32
    args.learning_rate = 0.0001
    args.loss = 'mse'
    args.lradj = 'type1'

    args.num_workers = 0
    args.itr = 1
    args.train_epochs = 10
    args.patience = patience
    args.des = 'exp'

    args.use_gpu = True if torch.cuda.is_available() else False
    args.gpu = 0

    args.checkpoints = './pipeline_informer/checkpoints'

    # Set augments by using data name
    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    }

    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Informer

    setting = ''

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_eb{}_dt{}_{}_{}_fq{}_pe{}_{}'.format(
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.attn,
            args.embed,
            args.distil,
            args.des,
            ii,
            args.freq,
            args.patience,
            time_point)

        # set experiments
        exp = Exp(args)

        # train
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # test
        # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting)

        torch.cuda.empty_cache()
    pass

    # --------------------------------------------------

    # --------------------------------------------------

    # set model path
    # setting = 'informer_ETTh1_ftM_sl96_ll48_pl24_dm512_nh8_el3_dl2_df512_atprob_ebtimeF_dtTrue_exp_0'
    weights_path = os.path.join('./pipeline_informer/checkpoints/', setting, 'checkpoint.pth')

    # set prediction dataloader (using test dataloader here)
    # 设置数据集类型，Dataset_ETT_hour 或者 Dataset_ETT_minute
    # freq = [t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

    if freq == 'h':
        Data = Dataset_ETT_hour
    else:
        Data = Dataset_ETT_minute
    pass

    timeenc = 0 if args.embed != 'timeF' else 1
    flag = 'val'
    shuffle_flag = False
    drop_last = True
    batch_size = 1

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    # set device
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    args.output_attention = True

    # build model
    model = Informer(
        args.enc_in,
        args.dec_in,
        args.c_out,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.factor,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.dropout,
        args.attn,
        args.embed,
        args.freq,
        args.activation,
        args.output_attention,
        args.distil,
        device
    )

    # load parameters
    model.load_state_dict(torch.load(weights_path))

    model = model.double().to(device)
    model.eval()

    preds = []
    trues = []

    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        batch_x = batch_x.double().to(device)
        batch_y = batch_y.double()
        batch_x_mark = batch_x_mark.double().to(device)
        batch_y_mark = batch_y_mark.double().to(device)

        # decoder input = concat[start token series(label_len), zero padding series(pred_len)]
        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).double()
        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).double().to(device)

        # encoder - decoder
        if args.output_attention:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        batch_y = batch_y[:, -args.pred_len:, :]

        pred = outputs.detach().cpu().numpy()  # .squeeze()
        true = batch_y.detach().cpu().numpy()  # .squeeze()

        preds.append(pred)
        trues.append(true)

    preds = np.array(preds)
    trues = np.array(trues)

    print('prediction shape:', preds.shape, trues.shape)  # [num_samples//batch_size, batch_size, pred_len, c_out]
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('prediction shape:', preds.shape, trues.shape)  # [num_samples, pred_len, c_out]

    trues_inverse_transform = data_set.inverse_transform(trues)
    preds_inverse_transform = data_set.inverse_transform(preds)

    # draw OT prediction
    plt.figure()
    plt.plot(trues_inverse_transform[0, :, -1], label='GroundTruth')
    plt.plot(preds_inverse_transform[0, :, -1], label='Prediction')
    plt.legend()

    plt.savefig(f'./pipeline_informer/results/final_{setting}.png')

    df_trues = pd.DataFrame(trues_inverse_transform[0, :, :])
    df_trues.to_csv(f'./pipeline_informer/results/final_trues_{setting}_{time_point}.csv')

    df_preds = pd.DataFrame(preds_inverse_transform[0, :, :])
    df_preds.to_csv(f'./pipeline_informer/results/final_preds_{setting}_{time_point}.csv')

    # plt.show()

    # draw HUFL prediction
    # plt.figure()
    # plt.ylim(bottom=2, top=5)
    # plt.plot(trues[0, :, 0], label='GroundTruth')
    # plt.plot(preds[0, :, 0], label='Prediction')
    # plt.legend()
    # plt.savefig(f'./pipeline_informer/results/plt_open_price_{time_point}.png')
    #
    # plt.show()
    pass


if __name__ == "__main__":

    # freq = [t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly]

    # 小时，单独列
    train(freq='h', features='S', patience=5, use_technical_indicator=False, download_data=True)

    # 小时，多列
    train(freq='h', features='M', patience=5, use_technical_indicator=False, download_data=False)

    # 分钟，单独列
    train(freq='t', features='S', patience=5, use_technical_indicator=False, download_data=False)

    # 分钟，多列
    train(freq='t', features='M', patience=5, use_technical_indicator=False, download_data=False)

    pass
