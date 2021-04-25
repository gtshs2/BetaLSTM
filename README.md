Bivariate Beta-LSTM, AAAI-20
Kyungwoo Song, JoonHo Jang, Seung jae Shin, Il-Chul Moon

We don't upload the data file becuase of copyright issue.
Please download the data (JSB, Muse, Nottingham, Piano) from other sources

##########
# BLSTM : ibetalstm
# bBLSTM(5G) : gbetalstm with kl_lambda = 0.0
# bBLSTM(5G+p) : gbetalstm with nonzero kl_lambda

# Piano
for itr1 in lstm flstm g2lstm ibetalstm
do
python3 train_speech.py --dataset=Piano --max_epoch=100 --clip=1.0 --random_seed=1000 --model=${itr1} --rnn_d=200 --dropout=0.1 --depth=2 --lr=2e-3
done

for itr1 in 0.0 0.001
do
python3 train_speech.py --dataset=Piano --max_epoch=100 --clip=1.0 --kl_gamma_prior=0.5 --random_seed=1000 --kl_lambda=${itr1} --model=gbetalstm --rnn_d=200 --dropout=0.2 --depth=2 --lr=2e-3
done

# Nott
for itr1 in lstm flstm g2lstm ibetalstm
do
python3 train_speech.py --dataset=Nott --max_epoch=100 --clip=1.0 --random_seed=1000 --model=${itr1} --rnn_d=200 --dropout=0.1 --depth=2 --lr=2e-3
done

for itr1 in 0.0 0.001
do
python3 train_speech.py --dataset=Nott --max_epoch=100 --clip=1.0 --kl_gamma_prior=0.5 --random_seed=1000 --kl_lambda=${itr1} --model=gbetalstm --rnn_d=200 --dropout=0.2 --depth=2 --lr=2e-3
done

# JSB
for itr1 in lstm flstm g2lstm ibetalstm
do
python3 train_speech.py --dataset=JSB --max_epoch=100 --clip=1.0 --random_seed=1000 --model=${itr1} --rnn_d=200 --dropout=0.1 --depth=2 --lr=2e-3
done

for itr1 in 0.0 1.0
do
python3 train_speech.py --dataset=JSB --max_epoch=100 --clip=1.0 --kl_gamma_prior=0.5 --random_seed=1000 --kl_lambda=${itr1} --model=gbetalstm --rnn_d=200 --dropout=0.2 --depth=2 --lr=2e-3
done

# Muse
for itr1 in lstm flstm g2lstm ibetalstm
do
python3 train_speech.py --dataset=Muse --max_epoch=100 --clip=1.0 --random_seed=1000 --model=${itr1} --rnn_d=200 --dropout=0.1 --depth=2 --lr=2e-3
done

for itr1 in 0.0 0.001
do
python3 train_speech.py --dataset=Muse --max_epoch=100 --clip=1.0 --kl_gamma_prior=0.5 --random_seed=1000 --kl_lambda=${itr1} --model=gbetalstm --rnn_d=200 --dropout=0.2 --depth=2 --lr=2e-3
done