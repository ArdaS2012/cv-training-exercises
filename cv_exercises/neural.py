"""Supplied two-layer CPU training plumbing; activation and loss belong to students."""
import numpy as np
from cv_exercises.data import classification_split


def train(activation, loss_function, steps=80, rate=0.5):
    x,y,ids,tr,va,te=classification_split()
    rng=np.random.default_rng(42)
    w1=rng.normal(0,0.15,(64,24)); b1=np.zeros(24)
    w2=rng.normal(0,0.15,(24,10)); b2=np.zeros(10)
    target=np.eye(10)[y[tr]]
    history=[]; validation=[]
    for _ in range(steps+1):
        z=x[tr]@w1+b1; h=activation(z)
        if h is None: return None
        h=np.asarray(h)
        if h.shape != z.shape or not np.isfinite(h).all(): raise ValueError('Activation must return finite values with the same shape.')
        logits=h@w2+b2; exp=np.exp(logits-logits.max(axis=1,keepdims=True)); p=exp/exp.sum(axis=1,keepdims=True)
        loss=loss_function(p,y[tr])
        if loss is None: return None
        if not np.isscalar(loss) or not np.isfinite(loss) or loss<0: raise ValueError('Loss must be one finite nonnegative scalar.')
        history.append(float(loss))
        vh=activation(x[va]@w1+b1); vl=vh@w2+b2
        validation.append(float(np.mean(vl.argmax(axis=1)==y[va])))
        if _ == steps: break
        # Prepared derivatives for softmax+cross-entropy and ReLU.
        # Do not edit this plumbing during the bounded practice.
        dlogits=(p-target)/len(tr)
        dw2=h.T@dlogits; db2=dlogits.sum(axis=0)
        dz=(dlogits@w2.T)*(z>0)
        dw1=x[tr].T@dz; db1=dz.sum(axis=0)
        w1-=rate*dw1; b1-=rate*db1; w2-=rate*dw2; b2-=rate*db2
    predictions=(activation(x[va]@w1+b1)@w2+b2).argmax(axis=1)
    return {'loss':history,'validation_accuracy':validation,'predictions':predictions.tolist(),'targets':y[va].tolist(),
            'validation_ids':ids[va].tolist(),'train_count':len(tr),'validation_count':len(va),'test_count':len(te),
            'test_used':False,'steps':steps,'rate':rate}
