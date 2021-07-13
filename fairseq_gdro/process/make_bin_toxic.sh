#!/bin/bash

source activate py37

dname=$1
input_dir="/private/home/chuntinz/work/data/hatespeech/${dname}/split_raw"
opt_dir="/private/home/chuntinz/work/data/hatespeech/${dname}/split_bpe"

for split in train valid test; do
  bash process/gpt2_encode_raw_to_ids.sh ${input_dir}/${split}.sent0 ${opt_dir}/${split}.sent0
  cp ${input_dir}/${split}.label ${opt_dir}/
done

dict_path=/private/home/chuntinz/work/fairseq-hallucination/pretrain_scripts/container/gpt2_bpe
inputdir=${opt_dir}
final_optdir="/private/home/chuntinz/work/data/hatespeech/${dname}/bin"

splits="sent0"
for split in ${splits}; do
  opt_split=input0
  fairseq-preprocess --only-source --trainpref ${inputdir}/train.${split} --validpref ${inputdir}/valid.${split} \
--testpref ${inputdir}/test.${split}  --destdir ${final_optdir}/${opt_split} --workers 40 --srcdict ${dict_path}/dict.txt
done

split=label
rm -rf ${final_optdir}/${split}
fairseq-preprocess --only-source --trainpref ${inputdir}/train.${split} --destdir ${final_optdir}/${split} \
--validpref ${inputdir}/valid.${split} --testpref ${inputdir}/test.${split}  --workers 40

cp ${input_dir}/*labels ${final_optdir}