# group-conditional-DRO

This repository contains code for experiments in the ICML2021 paper [Examining and Combating Spurious Features under Distribution Shift](https://arxiv.org/abs/2106.07171).
## Data

Data processing scripts can be found under fairseq_gdro/process.
- MNLI

   (1) Generate meta data of the MNLI dataset

   ``python process/gen_mnli_meta.py``

   (2) Generate imperfect partitions, under the output folder "*.fg.labels" are labels for clean partitions, and "train.resplit.labels" are labels for imperfect partions

   ``python process/gen_mnli.py``
 
   (3) make binarized data as inputs to fairseq

   ``bash process/make_bin_mnli.sh``

- Toxicity Detection: FDCL18 ((Fortuna & Nunes, 2018)), besides the clean partition,
we also explore imperfect partitions created by a supervised classifier (`fairseq_gdro/process/gen_resplite_labels_toxic.py`) 
and with unsupervised clustering using BERT-sentence embeddings (`fairseq_gdro/process/clustering_with_pretrained_models.py`).

  

- CelebA (Liu et al., 2015):
    Data loader for clean and imperfect partitions can be found in `image_classification/data/celeba.py`
    
    
## Training
For text experiments, please install fairseq under fairseq_gdro:

``cd fairseq_gdro; pip install --editable ./``

Important arguments can be found in Line 488 of `fairseq_gdro/fairseq/options.py` and `fairseq_gdro/fairseq/criterion/group_dro_loss.py`.

- Baseline Models: the scripts to run baseline approaches can be found under `fairseq_gdro/baseline_jobs`.

- GC-DRO: the scripts to run GC-DRO can found under `fariseq_gdro/jobs/`.

For image classification, the scripts to run baseline approaches and GC-DRO can be found under `image_classification/scripts/`.


For details of the methods and results, please refer to our paper. 

```bibtex
@inproceedings{zhou21icml,
    title = {Examining and Combating Spurious Features under Distribution Shift},
    author = {Chunting Zhou and Xuezhe Ma and Paul Michel and Graham Neubig},
    booktitle = {International Conference on Machine Learning (ICML)},
    address = {Virtual},
    month = {July},
    url = {http://arxiv.org/abs/2106.07171},
    year = {2021}
}
}