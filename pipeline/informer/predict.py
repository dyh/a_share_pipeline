import sys


if 'pipeline' not in sys.path:
    sys.path.append('../../')

if 'Informer2020_main' not in sys.path:
    sys.path.append('../../Informer2020_main')

if 'FinRL_Library_master' not in sys.path:
    sys.path.append('../../FinRL_Library_master')

import torch
import argparse

import pandas as pd

from pipeline import stock_data

from pipeline.informer.exp_informer import Exp_Informer
from pipeline.utils.date_time import get_datetime_from_date_str, get_next_work_day, get_today_date


def create_blank_csv_file(csv_file_path, predict_days=96, date_begin='2021-03-30', default_val=50):
    """
    创建空白csv文件，用于预测
    :param csv_file_path: csv文件路径
    :param predict_days: 要预测的天数 96,288,672
    :param date_begin: 开始的日期 2021-03-30
    :param default_val: 收盘价默认值 50
    :return:
    """
    # 创建空白 csv
    # df = pd.DataFrame(columns=['date', 'open', 'high', 'low', 'volume', 'amount', 'OT'])
    df = pd.read_csv(csv_file_path)

    # 为了计算将要预测的日期
    date_predict_temp = None

    # 将时间点换算成工作日
    time_point_count = len(stock_data.list_time_point_5minutes)
    predict_days = int(predict_days / time_point_count)

    for i in range(predict_days):

        if date_predict_temp is None:
            date_predict_temp = get_datetime_from_date_str(date_begin)
            next_date_predict_temp = date_predict_temp
        else:
            next_date_predict_temp = get_next_work_day(date_predict_temp, next_flag=+1)
        pass

        # 获得 0.mmdd 假数据，用于辨识
        mm = str(next_date_predict_temp).split('-')[1]
        dd = str(next_date_predict_temp).split('-')[2]
        float_0_mmdd = default_val + float('0.' + mm + dd)

        # 每5分钟的K线，一天有48条数据
        for index in range(time_point_count):
            new_row = pd.DataFrame(
                {'date': str(next_date_predict_temp) + ' ' + stock_data.list_time_point_5minutes[index],
                 'open': float_0_mmdd, 'high': float_0_mmdd, 'low': float_0_mmdd, 'volume': float_0_mmdd,
                 'amount': float_0_mmdd, 'OT': float_0_mmdd}, index=[0])

            df = df.append(new_row, ignore_index=True)
            pass
        pass

        date_predict_temp = next_date_predict_temp
    pass

    df.to_csv(csv_file_path, index=False)
    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

    # ----
    # 股票代码
    parser.add_argument('--stock_code', type=str, required=False, default='sh.600036', help='stock code')
    # ----

    # parser.add_argument('--model', type=str, required=True, default='informer',
    #                     help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
    # parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
    # parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')

    parser.add_argument('--model', type=str, required=False, default='informer',
                        help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
    parser.add_argument('--root_path', type=str, default='./temp_dataset/', help='root path of the data file')

    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # data_parser = {
    #     'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    #     'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    #     'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    #     'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    # }

    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [6, 6, 6], 'S': [1, 1, 1], 'MS': [6, 6, 1]},
    }

    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    print('Args in experiment:')
    print(args)

    # ----
    # 预测长度
    # 根据预测长度，从当前日期，向后做空白数据，用来预测
    # pred_len
    csv_file_path1 = args.root_path + args.data_path
    # 开始日期
    date_today = get_today_date()

    # create_blank_csv_file(csv_file_path=csv_file_path1, predict_days=args.pred_len,
    #                       date_begin=date_today, default_val=50)
    # ----

    Exp = Exp_Informer

    # for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model,
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
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, 0)

    exp = Exp(args)  # set experiments
    # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    # exp.train(setting)
    # print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
