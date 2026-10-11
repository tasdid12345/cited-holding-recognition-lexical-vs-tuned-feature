"""CaseHOLD: TF-IDF similarity baseline vs tuned LightGBM on richer lexical-similarity features."""
import io, json, os, sys, time
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

SEED, N_TRAIN, N_TEST, SEEDS = 42, 6000, 2000, [42, 43, 44]
np.random.seed(SEED)
t0 = time.time()

# ---------- data loading (HTTP, retry with backoff) ----------
def get(url):
    for k in range(5):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  retry {k} for {url}: {e}")
            time.sleep(2 ** k)
    raise RuntimeError("download failed: " + url)

def read_split(url):
    df = pd.read_parquet(io.BytesIO(get(url).content))
    if "endings" not in df:  # casehold/casehold layout
        df = df.rename(columns={"citing_prompt": "context"})
        df["endings"] = df[[f"holding_{i}" for i in range(5)]].values.tolist()
    df["endings"] = df["endings"].map(list)
    df["label"] = df["label"].astype(int)
    df["context"] = df["context"].str.replace("<HOLDING>", " ", regex=False)
    return df[["context", "endings", "label"]]

def load():
    base = "https://huggingface.co/datasets/"
    for repo, cfg, direct in [("coastalcph/lex_glue", "case_hold", True), ("casehold/casehold", None, False)]:
        try:
            api = get(f"{base.replace('datasets/', 'api/datasets/')}{repo}/parquet").json()
            d = api[cfg] if cfg else list(api.values())[0]
            return repo, read_split(d["train"][0]), read_split(d["test"][0])
        except Exception as e:
            print("API route failed:", e)
            if direct:
                try:
                    u = f"{base}{repo}/resolve/refs%2Fconvert%2Fparquet/{cfg}/%s/0000.parquet"
                    return repo, read_split(u % "train"), read_split(u % "test")
                except Exception as e2:
                    print("direct route failed:", e2)
    sys.exit("could not load CaseHOLD")

repo, tr, te = load()
tr = tr.sample(n=min(N_TRAIN, len(tr)), random_state=SEED).reset_index(drop=True)
te = te.sample(n=min(N_TEST, len(te)), random_state=SEED).reset_index(drop=True)
print(f"loaded {repo}: train={len(tr)} test={len(te)} ({time.time()-t0:.0f}s)")

# ---------- shared preprocessing / features ----------
def texts(df):
    return df.context.tolist(), [e for es in df.endings for e in es]

def fit_vecs(df):
    ctx, cands = texts(df)
    allt = ctx + cands
    V = {"w": TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=500000, dtype=np.float32).fit(allt),
         "c": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3,
                              max_features=300000, dtype=np.float32).fit(allt),
         "t": CountVectorizer(min_df=2).fit(allt)}
    cp = V["t"].transform(ctx)
    dfreq = np.asarray((cp > 0).sum(0)).ravel()
    V["idf"] = np.log(1 + (len(ctx) - dfreq + 0.5) / (dfreq + 0.5))
    V["rare"] = (dfreq <= 10).astype(float)  # terms seen in <=10 training passages
    V["avg"] = np.asarray(cp.sum(1)).mean()
    return V

rsum = lambda M: np.asarray(M.sum(1)).ravel()

def build(df, V, k1=1.5, b=0.75):
    """Return (n*5 x F feature matrix, per-passage max candidate word-overlap)."""
    n = len(df); gi = np.repeat(np.arange(n), 5)
    ctx, cands = texts(df)
    cos = {k: rsum(V[k].transform(ctx)[gi].multiply(V[k].transform(cands))) for k in "wc"}
    cp, cc = V["t"].transform(ctx)[gi].tocsr(), V["t"].transform(cands)
    P, Q = (cp > 0).astype(float), (cc > 0).astype(float)
    ov, ql = rsum(Q.multiply(P)), rsum(Q)
    ovf, jac = ov / np.maximum(ql, 1), ov / np.maximum(ql + rsum(P) - ov, 1)
    rare_ov = Q.multiply(P) @ V["rare"]
    rare_frac = rare_ov / np.maximum(Q @ V["rare"], 1)
    # BM25: candidate terms as query, citing passage as document
    dl = rsum(cp); rows = np.repeat(np.arange(cp.shape[0]), np.diff(cp.indptr)); tf = cp.data.astype(float)
    cp.data = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / V["avg"]))
    bm25 = Q.multiply(cp) @ V["idf"]
    clen = np.array([len(c.split()) for c in cands], float)
    G = np.column_stack([cos["w"], cos["c"], bm25, rare_ov, rare_frac, ovf, jac, clen]).reshape(n, 5, -1)
    rank = G.argsort(1).argsort(1)  # rank among the 5 candidates
    top1, top2 = np.sort(G, 1)[:, -1:], np.sort(G, 1)[:, -2:-1]
    gap = G - np.where(G == top1, top2, top1)  # margin vs best other candidate
    X = np.concatenate([G, G - G.mean(1, keepdims=True), rank, gap], 2).reshape(n * 5, -1)
    return X, ovf.reshape(n, 5).max(1)

V = fit_vecs(tr)
Xtr, _ = build(tr, V); Xte, ov_te = build(te, V)
ytr = (np.arange(5)[None] == tr.label.values[:, None]).ravel().astype(int)
print(f"features {Xtr.shape[1]} cols built ({time.time()-t0:.0f}s)")

# ---------- shared evaluation + CV tuning ----------
rows_of = lambda idx: (np.asarray(idx)[:, None] * 5 + np.arange(5)).ravel()
pick = lambda p: p.reshape(-1, 5).argmax(1)

def cv_select(make, grid, X, y, lab):
    """Pick params by 3-fold CV (folds over passages) on 5-way accuracy; training data only."""
    kf, best = KFold(3, shuffle=True, random_state=SEED), (-1, None)
    for p in grid:
        s = np.mean([(pick(make(p, SEED).fit(X[rows_of(a)], y[rows_of(a)]).predict_proba(X[rows_of(v)])[:, 1]) == lab[v]).mean()
                     for a, v in kf.split(lab)])
        if s > best[0]: best = (s, p)
    return best

def evaluate(model, X):
    pr = pick(model.predict_proba(X)[:, 1]); lab = te.label.values
    yb = (np.arange(5)[None] == lab[:, None]).ravel(); pb = (np.arange(5)[None] == pr[:, None]).ravel()
    low = ov_te <= np.quantile(ov_te, 1 / 3)
    return {"acc": (pr == lab).mean(), "f1": f1_score(yb, pb, average="macro"), "low": (pr == lab)[low].mean()}

def run(name, make, grid, X, Xt):
    cvs, p = cv_select(make, grid, X, ytr, tr.label.values)
    print(f"{name}: CV acc {cvs:.4f} params {p} ({time.time()-t0:.0f}s)")
    res = [evaluate(make(p, s).fit(X, ytr), Xt) for s in SEEDS]
    return p, {k: [r[k] for r in res] for k in res[0]}

mk_base = lambda p, s: make_pipeline(StandardScaler(), LogisticRegression(C=p["C"], random_state=s, max_iter=1000))
mk_gbm = lambda p, s: LGBMClassifier(**p, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                     random_state=s, verbose=-1, n_jobs=4)
pb, B = run("baseline", mk_base, [{"C": c} for c in (0.1, 1, 10, 100)], Xtr[:, :1], Xte[:, :1])  # col 0 = word 1-2gram TF-IDF cosine
space = {"num_leaves": [7, 15, 31], "learning_rate": [0.03, 0.1], "n_estimators": [150, 300], "min_child_samples": [20, 50]}
pi, I = run("improved", mk_gbm, list(ParameterSampler(space, 6, random_state=SEED)), Xtr, Xte)

# ---------- results ----------
m = {}
for tag, R in (("baseline", B), ("improved", I)):
    m[f"{tag}_score"], m[f"{tag}_score_std"] = float(np.mean(R["acc"])), float(np.std(R["acc"]))
    m[f"{tag}_f1"], m[f"{tag}_f1_std"] = float(np.mean(R["f1"])), float(np.std(R["f1"]))
    m[f"{tag}_low_overlap_acc"] = float(np.mean(R["low"]))
m["improvement"] = m["improved_score"] - m["baseline_score"]
m["improvement_low_overlap"] = m["improved_low_overlap_acc"] - m["baseline_low_overlap_acc"]
m["chance_accuracy"] = 0.2
if not all(np.isfinite(v) for v in m.values()): sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4)); xs = np.arange(2)
ax.bar(xs - 0.2, [m["baseline_score"], m["baseline_low_overlap_acc"]], 0.4, label="baseline TF-IDF")
ax.bar(xs + 0.2, [m["improved_score"], m["improved_low_overlap_acc"]], 0.4, label="LightGBM")
ax.set_xticks(xs); ax.set_xticklabels(["all test", "lowest-overlap third"]); ax.set_ylabel("5-way accuracy")
ax.axhline(0.2, color="gray", ls="--", lw=1); ax.legend(); plt.tight_layout(); plt.savefig("figures/accuracy.png", dpi=120)

out = {
    "question": "Can a tuned gradient-boosting model on richer lexical-similarity features match the correct cited legal holding to a citing passage more accurately than a standard n-gram overlap baseline, including when wording overlap is low?",
    "method": "CaseHOLD (LexGLUE) official train/test splits subsampled; both models pick the top-scoring of 5 candidate holdings; hyper-parameters tuned by 3-fold CV on training passages only; test used once.",
    "dataset": {"name": "CaseHOLD (LexGLUE case_hold)", "source_url": "https://huggingface.co/datasets/coastalcph/lex_glue",
                "n_rows": int(len(tr) + len(te)), "n_train": int(len(tr)), "n_test": int(len(te)),
                "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "5-way accuracy of picking the correct holding", "higher_is_better": True,
    "baseline": "Word 1-2-gram TF-IDF cosine between passage and candidate, logistic regression on that single score, top candidate chosen.",
    "improved": "LightGBM binary classifier on word/char n-gram cosines, BM25, rare-term overlap, overlap/length features and each candidate's score relative to the other four; params tuned by CV.",
    "metrics": m,
    "notes": (f"Data source repo: {repo}. Train subsampled from the first parquet shard of the train split. Seeds {SEEDS}; baseline is deterministic so its std is ~0. "
              f"Low-overlap subset = test third with lowest max candidate-token overlap with the passage (label-free). Macro-F1 is over binary correct/incorrect candidate rows, "
              f"predicting 1 for the chosen candidate. Best baseline params {pb}; best LightGBM params {pi}. Single subsample, so differences of ~1-2 points are within sampling noise (n_test={len(te)}). "
              f"Runtime {time.time()-t0:.0f}s."),
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(m, indent=2))
