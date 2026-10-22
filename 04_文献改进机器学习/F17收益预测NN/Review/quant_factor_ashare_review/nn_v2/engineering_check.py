# Research project author: Gilmour Jia
"""Synthetic GPU smoke only; never a historical-selection experiment."""
import runtime
import time
import numpy as np
import torch
from test_nn_v2 import sample
from model import fit, predict, save_bundle, load_bundle
from spec import SPEC, code_hash, digest, write_json


def main():
    out=runtime.ROOT/"engineering"
    out.mkdir(exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"
    start=time.monotonic()
    train=[sample(seed=i) for i in range(4)]
    valid=[sample("2019-06-01","2019-07-01",seed=44)]
    model,scale,epoch,val,history=fit(train,valid,.001,.0001,0,device=device,smoke=True)
    path=out/"SYNTHETIC_ONLY_smoke_bundle.pt"
    save_bundle(path,model,scale,0,epoch,{"data_version":"SYNTHETIC_ENGINEERING_ONLY",
                "code_version":code_hash(),"spec_hash":digest(SPEC),
                "train_range":["synthetic_date_1","synthetic_date_4"],
                "hyperparameters":{"lr":.001,"weight_decay":.0001}})
    reload,bundle=load_bundle(path,device)
    a=predict(model,valid[0].x,scale,device)
    b=predict(reload,valid[0].x,bundle["scalers"]["label_scale"],device)
    assert np.array_equal(a,b)
    result={"status":"PASS","device":device,"torch":torch.__version__,
            "gpu":torch.cuda.get_device_name(0) if device=="cuda" else None,
            "data":"synthetic only; not financial performance",
            "epochs_run":len(history),"best_epoch":epoch,"validation_huber":val,
            "reload_bitwise_equal":True,"seconds":time.monotonic()-start,
            "formal_nn_fits":0,"history":history}
    write_json(out/"smoke_result.json",result)
    print(result)


if __name__=="__main__":main()
