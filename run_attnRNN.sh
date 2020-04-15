python -u train.py \
    --model attention \
    --epochs 5 \
    --lr 0.01 \
    --weight-decay 0.0 \
    --batch-size 128 \
    --hidden_dim 100 \
    --useGlove True \
    --trainable True \
    --bidirectional True \
    --num_layers 3 \
    --dropout 0.3 \
    --typeOfRNN simple \
    --typeOfPadding no_padding | tee attnRNN.log
