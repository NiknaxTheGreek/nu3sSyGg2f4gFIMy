import pandas as pd
import numpy as np
import step7 as s

# Candidate embeddings: one row per frozen candidate, 100 Word2Vec dimensions.
emb = pd.DataFrame(s.X, columns=[f"w2v_{i+1:03d}" for i in range(s.X.shape[1])])
emb.insert(0, "job_title", s.df["job_title"].astype(str).values)
emb.insert(0, "representative_id", s.df["representative_id"].astype(int).values)
emb.to_csv("model40_candidate_embeddings_50x100.csv", index=False)
np.save("model40_candidate_embeddings_50x100.npy", s.X.astype(np.float32))

# Query embeddings.
qrows = []
for name, q in [("aspiring", s.QUERIES[0]), ("seeking", s.QUERIES[1])]:
    row = {"query_name": name, "query_text": q}
    row.update({f"w2v_{i+1:03d}": float(v) for i, v in enumerate(s.qvecs[q])})
    qrows.append(row)
pd.DataFrame(qrows).to_csv("model40_query_embeddings_2x100.csv", index=False)

# Compact token-vector cache: enough to rebuild every title/query embedding without 40.zip.
token_rows = []
for token in sorted(s.vectors):
    row = {"token": token}
    row.update({f"w2v_{i+1:03d}": float(v) for i, v in enumerate(s.vectors[token])})
    token_rows.append(row)
pd.DataFrame(token_rows).to_csv("model40_required_token_vectors.csv", index=False)

print(f"SAVED candidate embeddings: {s.X.shape}")
print(f"SAVED query embeddings: {len(qrows)} x {s.X.shape[1]}")
print(f"SAVED token-vector cache: {len(token_rows)} vectors")
