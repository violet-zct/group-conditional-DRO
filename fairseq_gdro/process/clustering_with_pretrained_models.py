import os
import numpy as np
import faiss
from collections import defaultdict
from collections import Counter

npy = "/private/home/chuntinz/work/data/hatespeech/founta/split_raw/founta_bert.npy"

ncentroids = 8
niter = 20
verbose = True
x = np.memmap(npy, dtype='float32', mode='r', shape=(72246, 1024))
d = x.shape[1]
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
kmeans.train(x)

D, I = kmeans.index.search(x, 1)

with open("/private/home/chuntinz/work/data/hatespeech/founta/bin/train.bert.labels", "w") as fout:
    for ii in I:
        fout.write(str(ii[0]) + "\n")