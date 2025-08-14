import os, csv, json, subprocess, tempfile, requests, argparse, hashlib

GAS_URL   = os.environ.get("GAS_URL")
GAS_TOKEN = os.environ.get("GAS_TOKEN")

def fetch_pending():
    r = requests.get(GAS_URL, params={"mode":"pending", "token":GAS_TOKEN}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError("GAS pending error: " + json.dumps(data))
    return data.get("items", [])  # [{row,url}, ...]

def run_analyzer(items):
    # items -> temp input.csv
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "input.csv")
        out = os.path.join(td, "result.csv")
        with open(inp, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["photo_id","url"])
            for it in items:
                w.writerow([f"R{it['row']}", it["url"]])

        # 무료 분석기 실행 (언어옵션 kor+eng 로 변경)
        cmd = [
            "python", "photo_guard_free.py",
            "--input", inp,
            "--output", out,
            "--workers", "8",
            "--tesslang", "kor+eng"   # ★ 여기만 eng -> kor+eng
        ]
        subprocess.check_call(cmd)

        # 결과 읽기
        out_rows = []
        with open(out, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out_rows.append(row)
        return out_rows

def apply_results(items, out_rows):
    by_id = { r["photo_id"]: r for r in out_rows }
    results = []
    for it in items:
        rid = f"R{it['row']}"
        r = by_id.get(rid)
        if r:
            results.append({"row": it["row"], "label": r.get("label",""), "reason": r.get("reason","")})
    if not results:
        print("no results to apply")
        return
    payload = {"results": results, "batch": True}
    r = requests.post(GAS_URL, params={"mode":"apply", "token":GAS_TOKEN}, json=payload, timeout=120)
    r.raise_for_status()
    print("apply:", r.text)

# ---- 병렬 샤딩 옵션 ----
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)   # 내 샤드 번호 (0..N-1)
    ap.add_argument("--shards", type=int, default=1)  # 총 샤드 수
    return ap.parse_args()

def pick_shard(items, shard, shards):
    sel = []
    for it in items:
        h = int(hashlib.md5(it["url"].encode("utf-8")).hexdigest(), 16)
        if (h % shards) == shard:
            sel.append(it)
    return sel

def main():
    if not GAS_URL or not GAS_TOKEN:
        raise RuntimeError("GAS_URL / GAS_TOKEN not set")
    args = parse_args()
    items = fetch_pending()
    if not items:
        print("no pending."); return
    items = pick_shard(items, args.shard, args.shards)
    if not items:
        print(f"no items for shard {args.shard}/{args.shards}"); return
    # items = items[:1500]  # 필요 시 1회 처리 상한
    out_rows = run_analyzer(items)
    apply_results(items, out_rows)

if __name__ == "__main__":
    main()
