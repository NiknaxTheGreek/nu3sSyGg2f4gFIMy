import os, re, json, zipfile, csv, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import ndcg_score
from sklearn.metrics.pairwise import cosine_similarity

INPUT = "step7_input.csv"
ZIP = "40.zip"
QUERIES = ("aspiring human resources", "seeking human resources")

_PHONE = re.compile(r"\(?\d{3}\)?\s*[- ]\s*\d{3}\s*[- ]\s*\d{4}")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_TOKEN = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|\d+")

def audit_tokens(text):
    text = str(text).replace("’", "'")
    text = _PHONE.sub(" ", text)
    text = _YEAR.sub(" ", text)
    toks = _TOKEN.findall(text)
    return [t for t in toks if t.lower() != "at" and not (len(t) == 1 and t.isupper())]

df = pd.read_csv(INPUT)
texts = list(df["job_title"].astype(str)) + list(QUERIES)
raw = [t for s in texts for t in audit_tokens(s)]
needed = set()
for t in raw:
    needed |= {t, t.lower()}
    if t.lower().endswith("'s"):
        needed |= {t[:-2], t[:-2].lower()}

vectors = {}
with zipfile.ZipFile(ZIP) as z:
    members = z.namelist()
    member = next(n for n in members if n.endswith("model.txt"))
    with z.open(member) as fh:
        header = fh.readline().decode("utf-8", "replace").strip().split()
        vocab, dim = int(header[0]), int(header[1])
        if vocab != 4027169 or dim != 100:
            raise RuntimeError(f"Unexpected Model 40 header: {vocab} {dim}")
        for line in fh:
            pos = line.find(b" ")
            if pos <= 0:
                continue
            try:
                word = line[:pos].decode("utf-8")
            except UnicodeDecodeError:
                continue
            if word in needed:
                vals = np.fromstring(line[pos+1:].decode("ascii", "ignore"), sep=" ", dtype=np.float32)
                if vals.size == dim:
                    vectors[word] = vals

def resolve(t):
    for x in (t, t.lower()):
        if x in vectors:
            return x
    if t.lower().endswith("'s"):
        for x in (t[:-2], t[:-2].lower()):
            if x in vectors:
                return x
    return None

def meanvec(s):
    rs = [resolve(t) for t in audit_tokens(s)]
    rs = [r for r in rs if r]
    if not rs:
        raise RuntimeError(f"No recognized tokens for: {s}")
    return np.mean(np.vstack([vectors[r] for r in rs]), axis=0)

X = np.vstack([meanvec(t) for t in df["job_title"]])
qvecs = {q: meanvec(q) for q in QUERIES}

coverage = sum(resolve(t) is not None for t in raw)
oov = sorted({t for t in raw if resolve(t) is None})

def corr(a, b):
    return {
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
        "kendall_tau_b": float(kendalltau(a, b, variant="b").statistic),
        "mae": float(np.mean(np.abs(np.asarray(a)-np.asarray(b)))),
        "rmse": float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2))),
    }

scores = {}
for q in QUERIES:
    cos = cosine_similarity(X, qvecs[q].reshape(1,-1)).ravel()
    norm = (cos + 1.0) / 2.0
    scores[q] = {"cos": cos, "norm": norm}

W_asp = scores[QUERIES[0]]["norm"]
W_seek = scores[QUERIES[1]]["norm"]
W_avg = (W_asp + W_seek) / 2.0

M = df["manual_relevance_score"].to_numpy(float)
R_asp = df["rule_score_aspiring"].to_numpy(float)
R_seek = df["rule_score_seeking"].to_numpy(float)
R_avg = df["rule_score_average"].to_numpy(float)

def top_ids(vals, k=10):
    ids = df["representative_id"].to_numpy(int)
    order = np.lexsort((ids, -np.asarray(vals)))
    return [int(x) for x in ids[order[:k]]]

def top_overlap(a, b, k):
    A, B = set(top_ids(a,k)), set(top_ids(b,k))
    return {"count": len(A&B), "fraction": len(A&B)/k, "shared_ids": sorted(A&B)}

metrics = {
    "coverage": {"covered": int(coverage), "total": int(len(raw)), "unique_oov": oov},
    "combined_word2vec_vs_manual": corr(W_avg, M),
    "combined_word2vec_vs_rule": corr(W_avg, R_avg),
    "aspiring_word2vec_vs_manual": corr(W_asp, M),
    "aspiring_word2vec_vs_rule": corr(W_asp, R_asp),
    "seeking_word2vec_vs_manual": corr(W_seek, M),
    "seeking_word2vec_vs_rule": corr(W_seek, R_seek),
    "manual_vs_rule": corr(M, R_avg),
    "ndcg_at_10": {
        "aspiring_vs_manual": float(ndcg_score(M.reshape(1,-1), W_asp.reshape(1,-1), k=10)),
        "aspiring_vs_rule": float(ndcg_score(R_asp.reshape(1,-1), W_asp.reshape(1,-1), k=10)),
        "seeking_vs_manual": float(ndcg_score(M.reshape(1,-1), W_seek.reshape(1,-1), k=10)),
        "seeking_vs_rule": float(ndcg_score(R_seek.reshape(1,-1), W_seek.reshape(1,-1), k=10)),
        "combined_vs_manual": float(ndcg_score(M.reshape(1,-1), W_avg.reshape(1,-1), k=10)),
        "combined_vs_rule": float(ndcg_score(R_avg.reshape(1,-1), W_avg.reshape(1,-1), k=10)),
    },
    "top_overlap": {
        "combined_W_vs_M_top5": top_overlap(W_avg, M, 5),
        "combined_W_vs_R_top5": top_overlap(W_avg, R_avg, 5),
        "combined_W_vs_M_top10": top_overlap(W_avg, M, 10),
        "combined_W_vs_R_top10": top_overlap(W_avg, R_avg, 10),
    }
}

out = df.copy()
out["word2vec_cosine_aspiring"] = scores[QUERIES[0]]["cos"]
out["word2vec_score_aspiring"] = W_asp
out["word2vec_cosine_seeking"] = scores[QUERIES[1]]["cos"]
out["word2vec_score_seeking"] = W_seek
out["word2vec_score_average"] = W_avg
out["word2vec_rank_average"] = pd.Series(-W_avg).rank(method="min").astype(int)
out = out.sort_values(["word2vec_rank_average","representative_id"])
out.to_csv("step7_results.csv", index=False)

payload = {
    "model": {
        "source": "NLPL Model 40",
        "corpus": "English CoNLL17",
        "algorithm": "Word2Vec Continuous Skipgram",
        "dimensions": 100,
        "window": 10,
        "vocabulary": 4027169,
    },
    "queries": list(QUERIES),
    "metrics": metrics,
    "rows": out.to_dict(orient="records"),
}
print("STEP7_JSON_START")
print(json.dumps(payload, separators=(",",":")))
print("STEP7_JSON_END")
