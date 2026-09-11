import re, zipfile, json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

INPUT='step7_input.csv'; ZIP='40.zip'
_PHONE=re.compile(r"\(?\d{3}\)?\s*[- ]\s*\d{3}\s*[- ]\s*\d{4}")
_YEAR=re.compile(r"\b(?:19|20)\d{2}\b")
_TOKEN=re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+")
def toks(text):
    text=str(text).replace('’',"'"); text=_PHONE.sub(' ',text); text=_YEAR.sub(' ',text)
    return [t for t in _TOKEN.findall(text) if t.lower()!='at' and not (len(t)==1 and t.isupper())]

df=pd.read_csv(INPUT)
raw=[t for s in df.job_title.astype(str) for t in toks(s)]
needed=set()
for t in raw:
    needed|={t,t.lower()}
    if t.lower().endswith("'s"): needed|={t[:-2],t[:-2].lower()}

vectors={}
with zipfile.ZipFile(ZIP) as z:
    member=next(n for n in z.namelist() if n.endswith('model.txt'))
    with z.open(member) as fh:
        h=fh.readline().decode('utf-8','replace').strip().split(); vocab,dim=map(int,h[:2])
        if (vocab,dim)!=(4027169,100): raise RuntimeError((vocab,dim))
        for line in fh:
            pos=line.find(b' ')
            if pos<=0: continue
            try: w=line[:pos].decode('utf-8')
            except UnicodeDecodeError: continue
            if w in needed:
                a=np.fromstring(line[pos+1:].decode('ascii','ignore'),sep=' ',dtype=np.float32)
                if a.size==100: vectors[w]=a

def resolve(t):
    for x in (t,t.lower()):
        if x in vectors:return x
    if t.lower().endswith("'s"):
        for x in (t[:-2],t[:-2].lower()):
            if x in vectors:return x
    return None

def meanvec(s):
    rr=[resolve(t) for t in toks(s)]; rr=[r for r in rr if r]
    if not rr: raise RuntimeError(f'no vectors for {s}')
    return np.mean(np.vstack([vectors[r] for r in rr]),axis=0)

X=np.vstack([meanvec(s) for s in df.job_title.astype(str)])
coverage=sum(resolve(t) is not None for t in raw)
if coverage!=len(raw): raise RuntimeError(f'coverage {coverage}/{len(raw)}')

# Save exact 100D candidate embeddings.
emb=pd.DataFrame(X,columns=[f'w2v_{i+1:03d}' for i in range(100)])
emb.insert(0,'job_title',df.job_title.values); emb.insert(0,'representative_id',df.representative_id.values)
emb.to_csv('step8_embeddings_100d.csv',index=False)

# PCA: sklearn PCA mean-centres by default. No standardization / z-scoring.
pca=PCA(svd_solver='full')
Z=pca.fit_transform(X)
ev=pca.explained_variance_ratio_; cum=np.cumsum(ev)
ks={str(t): int(np.argmax(cum>=t)+1) for t in (0.80,0.90,0.95,0.99)}

# Data-driven knee of cumulative EV curve: maximum distance to chord joining endpoints.
x=np.arange(1,len(cum)+1,dtype=float); y=cum.astype(float)
xn=(x-x[0])/(x[-1]-x[0]); yn=(y-y[0])/(y[-1]-y[0])
dist=yn-xn
knee=int(np.argmax(dist)+1)

# Use 90% variance as locked retention rule for downstream Ridge.
k=ks['0.9']
Zk=Z[:,:k]
Xrec=pca.inverse_transform(np.pad(Zk,((0,0),(0,Z.shape[1]-k)),constant_values=0))
recon_rmse=float(np.sqrt(np.mean((X-Xrec)**2)))
rel_frob=float(np.linalg.norm(X-Xrec,'fro')/np.linalg.norm(X-pca.mean_,'fro'))

# Pairwise geometry preservation.
D0=pairwise_distances(X,metric='euclidean'); Dk=pairwise_distances(Zk,metric='euclidean')
tri=np.triu_indices_from(D0,k=1)
dist_pear=float(pearsonr(D0[tri],Dk[tri]).statistic)
dist_spear=float(spearmanr(D0[tri],Dk[tri]).statistic)

pca_scores=pd.DataFrame(Z,columns=[f'PC{i+1:02d}' for i in range(Z.shape[1])])
pca_scores.insert(0,'job_title',df.job_title.values); pca_scores.insert(0,'representative_id',df.representative_id.values)
pca_scores.to_csv('step8_pca_scores_all.csv',index=False)

ret=pd.DataFrame(Zk,columns=[f'PC{i+1:02d}' for i in range(k)])
ret.insert(0,'job_title',df.job_title.values); ret.insert(0,'representative_id',df.representative_id.values)
ret.to_csv('step8_pca_scores_retained.csv',index=False)

var=pd.DataFrame({'component':np.arange(1,len(ev)+1),'explained_variance_ratio':ev,'cumulative_explained_variance':cum})
var.to_csv('step8_explained_variance.csv',index=False)

summary={
 'n_candidates':int(X.shape[0]),'original_dimensions':int(X.shape[1]),'max_nonzero_pcs':int(min(X.shape[0]-1,X.shape[1])),
 'coverage':{'covered':int(coverage),'total':int(len(raw))},
 'threshold_components':ks,'knee_component':knee,'retention_rule':'smallest k with cumulative explained variance >= 90%',
 'retained_components':k,'retained_variance':float(cum[k-1]),
 'first_10_explained_variance':[float(v) for v in ev[:10]],'first_10_cumulative':[float(v) for v in cum[:10]],
 'reconstruction_rmse':recon_rmse,'relative_frobenius_reconstruction_error':rel_frob,
 'pairwise_distance_pearson':dist_pear,'pairwise_distance_spearman':dist_spear,
 'centering':'mean-centred by PCA','standardization':'none'
}
with open('step8_summary.json','w') as f: json.dump(summary,f,indent=2)
print('STEP8_JSON_START'); print(json.dumps(summary,separators=(',',':'))); print('STEP8_JSON_END')
