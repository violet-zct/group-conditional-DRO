python -u celebba.py --arch resnet18 \
    --batch_size 256 --epochs 50 \
    --loss 'dro_greedy' --lr 0.0001 \
    --alpha 0.2 --beta 0.1 --recompute 1 \
    --data_path <DATA> --model_path <MODEL> \
    --group_split ['confounder'|'domain'] \
    --run 1 --seed 1

