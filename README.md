# Sentiment Analysis of Hotel Reviews using RNN
Different variants of Recurrent Neural Networks (RNN), namely Vanilla RNN, Gated Recurrent Unit (GRU), Long Short-Term Memory (LSTM), and a combination of these with multiplicative and additive attention are implemented in PyTorch for Sentiment Analysis of over 515,000 reviews of around 1,400 hotels in Europe (source: https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). 100-dimensional GloVe representations of the words are also used for the embedding of the vocabulary. Effect of different types of padding like pack padded sequence, padded sequence and no padding is observed. Using the train.py file one can choose to use this representations or even choose to keep these representations as trainable parameters. The models with attention work best with an accuracy of 94% for additive attention and 96% for multiplicative attention.

This project consists of the following files:
1. train.py
2. preprocessing.py
3. Models: vRNN.py, GRU.py, LSTM.py, attn_RNN.py
4. Shell scripts: run_vRNN.sh, run_GRU.sh, run_LSTM.sh, run_attnRNN.sh

Following are the parameters that can be set for the models:
lr: learning rate
weight-decay: Weight decay hyperparameter
batch-size: Input batch size for training
epochs: Number of epochs to train
model ('vRNN', 'GRU', 'LSTM','attention'): which model to train/evaluate
hidden_dim: number of hidden features
dropout: Dropout probability to be applied
num_layers: Number of RNN layers
useGlove: Flag for using GloVe Representation
trainable: Set requires_grad=False for GloVe representation matrix
bidirectional: Flag to determine uni-directional or bidirectional RNN
typeOfPadding ('pack_padded', 'padded', 'no_padding'): which type of padding to use
typeOfRNN ('simple', 'GRU', 'LSTM'): which type of RNN to use with attention
typeOfAttention ('multiplicative', 'additive'): which type of attention to use

Instructions for running a model:
1. Set desired parameters in the model's shell script
2. Open bash shell
3. Give permissions for execution using "chmod +x *.sh"
4. Run the desired model's shell script using "bash <filename>"
5. A log file and saved model with .pt extension are created automatically for the current execution

NOTE:
Please take note that before executing additive attention model you need to run one of the other non-attention models to avoid an unexplained runtime error. I tried to resolve this runtime error of binary cross entropy loss but I couldn't figure out why it would throw that error every time only on the first run. Because the raw dataset file is over 200 MB in size, I wasn't able to upload it. The file can be downloaded and preprocessed using preprocessing.py file.
