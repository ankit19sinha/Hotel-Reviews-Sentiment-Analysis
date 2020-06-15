python -u train.py \
    --model LSTM \
    --epochs 5 \
    --lr 0.01 \
    --weight-decay 0.0 \
    --batch-size 500 \
    --hidden_dim 100 \
    --dropout 0.5 \
    --num_layers 3 \
    --useGlove False \
    --trainable False \
    --bidirectional False \
    --typeOfPadding no_padding | tee LSTM.log
