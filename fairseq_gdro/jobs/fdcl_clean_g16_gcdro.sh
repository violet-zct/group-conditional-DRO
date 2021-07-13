#!/bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=fdcl,dro
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
#SBATCH --cpus-per-task=10
#SBATCH --time=0
#SBATCH --array=0-4

source activate py37

SAVE_ROOT=/private/home/chuntinz/work/fairseq-gdro/hs_models
DATA=/private/home/chuntinz/work/data/hatespeech/founta/bin
splits=(15213 17747 17 222 13)
# random seed
split=${splits[$SLURM_ARRAY_TASK_ID]}

exp_name=fdcl_g16_clean_gcdro_seed${split}
SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh

TOTAL_NUM_UPDATES=50000  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=0        # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=4
MAX_SENTENCES=256        # Batch size.
ROBERTA_PATH=/checkpoint/chuntinz/fairseq-hallucination/pretrain_scripts/container/roberta.base/model.pt

python -u train.py ${DATA} \
    --reweight 0 --seed ${split} --num-train-groups 16 --num-test-groups 16 \
    --skip-invalid-size-inputs-valid-test \
    --label-path ${DATA}/train.fg.labels \
    --criterion cross_entropy_group_dro \
    --dro-alpha 0.2 --beta-cover-instances 0.5 --ema 0.5 --beta-ema 0.5 \
    --resplit-patience 0 --resplit-epoch -1 --conservative 1 \
    --dynamic-group-dro 1 --resplit-greedy 1  \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0  \
    --arch roberta_base \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 40 \
    --find-unused-parameters \
    --log-format simple --log-interval 100 --save-dir ${SAVE} \
    --best-checkpoint-metric "worst_acc" --maximize-best-checkpoint-metric | tee ${SAVE}/log.txt

date
wait
python -u test.py ${DATA} \
    --reweight 0 --seed ${split} --num-train-groups 16 --num-test-groups 14 \
    --skip-invalid-size-inputs-valid-test \
    --label-path ${DATA}/train.fg.labels \
    --criterion cross_entropy_group_dro \
    --dro-alpha 0.2 --beta-cover-instances 1.0 --ema 0.5 \
    --resplit-greedy 1 --resplit-epoch 2 \
    --dynamic-group-dro 1 --resplit-thres -1 --resplit-patience 0 \
    --max-positions 512 \
    --max-sentences 512 \
    --max-tokens 2500 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0  \
    --arch roberta_base \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 30 \
    --find-unused-parameters \
    --log-format simple --log-interval 100 \
    --save-dir ${SAVE}

wait
rm $SAVE/*pt
