python -u celebba.py --arch resnet18 \
    --batch_size 256 --epochs 50 \
    --loss 'erm' --lr 0.0001 --reweight \
    --data_path <DATA> --model_path <MODEL> \
    --group_split ['confounder'|'domain'] \
    --run 1 --seed 1

