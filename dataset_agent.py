"""MVP: prompt -> find dataset (Hugging Face) -> rank -> download -> profile -> ML clean -> report.
Usage: python dataset_agent.py "customer churn data"
Install: pip install requests pandas scikit-learn pyarrow
"""
import sys, io, json, hashlib, datetime, requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

HF = "https://huggingface.co"
STOP = {"data", "dataset", "datasets", "me", "the", "a", "of", "for", "i", "need", "want", "get", "find", "with", "and"}
MAX_MB = 50


def search(prompt):
    words = [w for w in prompt.lower().split() if w not in STOP]
    seen = {}
    for q in [" ".join(words)] + words:
        r = requests.get(f"{HF}/api/datasets", params={"search": q, "limit": 15}, timeout=20)
        for d in r.json():
            seen[d["id"]] = d
        if len(seen) >= 6:
            break
    return list(seen.values())


def rank(prompt, cands):
    texts = [c["id"].replace("/", " ").replace("-", " ").replace("_", " ") + " " + " ".join(c.get("tags", [])) for c in cands]
    m = TfidfVectorizer().fit_transform([prompt] + texts)
    scores = cosine_similarity(m[0], m[1:])[0]
    return sorted(zip(scores, cands), key=lambda x: -x[0])[:3]


def get_file(ds_id):
    info = requests.get(f"{HF}/api/datasets/{ds_id}", timeout=20).json()
    files = [s["rfilename"] for s in info.get("siblings", [])]
    f = next((x for x in files if x.endswith((".csv", ".parquet"))), None)
    if not f:
        sys.exit("No CSV/Parquet file found in this dataset.")
    url = f"{HF}/datasets/{ds_id}/resolve/main/{f}"
    head = requests.head(url, allow_redirects=True, timeout=20)
    if int(head.headers.get("content-length", 0)) > MAX_MB * 1e6:
        sys.exit(f"File larger than {MAX_MB} MB, blocked.")
    raw = requests.get(url, timeout=120).content
    df = pd.read_parquet(io.BytesIO(raw)) if f.endswith(".parquet") else pd.read_csv(io.BytesIO(raw))
    return df, url, hashlib.sha256(raw).hexdigest(), info.get("cardData", {}).get("license", "unknown")


def profile(df):
    return {"rows": len(df), "cols": df.shape[1], "missing": int(df.isna().sum().sum()), "duplicates": int(df.duplicated().sum())}


def score(p):
    return round(100 - 100 * (p["missing"] / max(p["rows"] * p["cols"], 1)) - 100 * (p["duplicates"] / max(p["rows"], 1)), 1)


def clean(df):
    log = []
    n = len(df)
    df = df.drop_duplicates()
    log.append(f"Removed {n - len(df)} duplicate rows")
    num = df.select_dtypes("number").columns
    if len(num) and df[num].isna().any().any():
        df[num] = KNNImputer(n_neighbors=3).fit_transform(df[num])
        log.append("KNN-imputed missing numeric values")
    for c in df.select_dtypes("object").columns:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "unknown")
    if len(num) and len(df) > 20:
        mask = IsolationForest(contamination=0.02, random_state=0).fit_predict(df[num]) == 1
        log.append(f"Removed {int((~mask).sum())} outliers (Isolation Forest)")
        df = df[mask]
    return df, log


def main():
    prompt = " ".join(sys.argv[1:]) or input("What data do you need? ")
    top = rank(prompt, search(prompt))
    if not top:
        sys.exit("No datasets found.")
    for i, (s, c) in enumerate(top, 1):
        print(f"[{i}] {c['id']}  (match {s:.2f})")
    pick = int(input("Pick 1-3: ") or 1) - 1
    ds_id = top[pick][1]["id"]
    df, url, sha, lic = get_file(ds_id)
    before = profile(df)
    print("License:", lic, "| Before:", before, "| Quality:", score(before))
    if input("Clean it? (y/n): ").lower() != "y":
        return
    df, log = clean(df)
    after = profile(df)
    out = ds_id.replace("/", "_") + "_clean.csv"
    df.to_csv(out, index=False)
    prov = {"source": url, "license": lic, "sha256": sha, "time": datetime.datetime.now().isoformat(),
            "before": before, "after": after, "quality_before": score(before), "quality_after": score(after), "steps": log}
    json.dump(prov, open(out.replace(".csv", "_provenance.json"), "w"), indent=2)
    print("\n".join(log), f"\nQuality: {score(before)} -> {score(after)}\nSaved: {out}")


if __name__ == "__main__":
    main()
