# Research project author: Gilmour Jia
from pathlib import Path
import sys
ROOT=Path(r'workspace/review_workspace/quant_factor_ashare_review/nn_v2')
sys.path.insert(0,str(ROOT))
import runtime
import torch
import numpy as np
from research_ext.train_v5 import configure, load_v5
configure()
from data import split_samples
import model
pkg=ROOT/'derived/research_training_v5'
s=load_v5(pkg/'samples.npz')
print('samples',len(s),'width',s[0].x.shape[1])
tr,va,fi,ou=split_samples(s,2021)
print('split',len(tr),len(va),len(fi),len(ou))
m,scale,epoch,val,hist=model.fit(tr[:15],va[:4],1e-3,1e-4,0,3,'cuda')
print('fit',epoch,val,'scale',scale)
x=tr[0].x[:5]
y=model.predict(m,x,scale,'cuda')
print('pred',y.shape,float(np.isfinite(y).all()))
