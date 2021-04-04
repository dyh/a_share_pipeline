#### S
#
##python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#
##python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#
## python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#
#python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#
#python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#
#
#### M
#
##python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 672 --label_len 96 --pred_len 24 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#
##python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 48 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#
## python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 96 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#
#python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 672 --label_len 288 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#
#python -u train.py --freq t --train_epochs 40 --patience 10 --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
#

### S

python -u train.py --freq t --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1

python -u train.py --freq t --root_path ./temp_dataset/ --model informer --data ETTm1 --features S --seq_len 384 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1


### M

python -u train.py --freq t --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 672 --label_len 288 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1

python -u train.py --freq t --root_path ./temp_dataset/ --model informer --data ETTm1 --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des Exp --itr 1
