# ------------
# train和vali，不带 test，得到 weights 后，用于 pred 的测试，看是否能 pred 整个 csv 文件
# informer_ETTm1_ftS_sl96_ll48_pl48_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_Exp_0
# python -u main_informer.py --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1

# do_predict
#python -u test.py --do_predict --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1

### S
#python -u test.py --do_predict --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1

python -u test.py --do_predict --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1

python -u test.py --do_predict --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1


### M
#python -u test.py --do_predict --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1

python -u test.py --do_predict --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 672 --label_len 288 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1

python -u test.py --do_predict --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1

