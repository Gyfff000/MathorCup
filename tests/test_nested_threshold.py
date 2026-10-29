import numpy as np
from scripts.nested_threshold_experiment import select_thresholds

def test_thresholds_use_training_predictions_and_sparse_fallback():
    p=np.zeros((20,2,7)); q=np.zeros_like(p); q[:,:,0]=1
    q[:10,0,0]=0; q[:10,0,1]=1
    p[:,:,0]=.9; p[:10,0,0]=.2
    t=select_thresholds(p,q,np.array([.2,.5,.8]))
    assert t[0]==.5
    assert t[1]==.5
