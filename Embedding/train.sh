python train.py \
    --data_dir /path/to/horse2zebra \
    --output_dir checkpoints \
    --embedding_dim 128 \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --num_epochs 100 \
    --embedding_weight 1.0 \
    --reconstruction_weight 1.0 \
    --symmetric_weight 0.5