
from pathlib import Path
import csv, ast, hashlib, json, re
from collections import defaultdict

SUB_DIR = Path('/root/code/artifacts/submissions')
SUB_DIR.mkdir(parents=True, exist_ok=True)

PATHS = {
    'm4': SUB_DIR / 'submission_m4.csv',
    'm4_chat': SUB_DIR / 'submission_m4_chatfilter.csv',
    'm6_chat': SUB_DIR / 'submission_m6_dense_chatfilter.csv',
}

for name, path in PATHS.items():
    print(f'{name:8s} exists={path.exists()} path={path}')

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def normalize_key(k):
    return (k or '').replace('\ufeff', '').strip()

def looks_like_jsonl(path: Path) -> bool:
    with path.open('r', encoding='utf-8-sig', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            return s.startswith('{') and s.endswith('}')
    return False

def read_submission(path: Path):
    if looks_like_jsonl(path):
        rows = []
        with path.open('r', encoding='utf-8-sig') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                rows.append(json.loads(s))
        norm = []
        for r in rows:
            norm.append({normalize_key(k): v for k, v in r.items()})
        return norm, 'jsonl'
    else:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        norm = []
        for r in rows:
            norm.append({normalize_key(k): v for k, v in r.items()})
        return norm, 'csv'

def detect_id_col(rows):
    if not rows:
        raise ValueError('empty rows')
    candidates = ['eval_id', 'id', 'query_id', 'question_id', 'qid']
    keys = list(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    lowered = {k.lower(): k for k in keys}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    raise KeyError(f'id column not found. keys={keys}')

def detect_ref_col(rows):
    if not rows:
        raise ValueError('empty rows')
    candidates = ['references', 'topk', 'docids', 'pred_docids', 'contexts', 'retrieved_doc_ids']
    keys = list(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    lowered = {k.lower(): k for k in keys}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    raise KeyError(f'reference column not found. keys={keys}')

def detect_query_col(rows):
    if not rows:
        return None
    candidates = ['standalone_query', 'query', 'question', 'user_query']
    keys = list(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    lowered = {k.lower(): k for k in keys}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None

def parse_refs(value):
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                if 'docid' in item:
                    out.append(str(item['docid']))
                elif 'id' in item:
                    out.append(str(item['id']))
                else:
                    out.append(str(item))
            else:
                out.append(str(item))
        return out
    if isinstance(value, dict):
        if 'docid' in value:
            return [str(value['docid'])]
        if 'id' in value:
            return [str(value['id'])]
        return [str(value)]
    s = str(value).strip()
    if not s:
        return []
    try:
        out = ast.literal_eval(s)
        return parse_refs(out)
    except Exception:
        pass
    if '|' in s:
        return [x.strip() for x in s.split('|') if x.strip()]
    if ',' in s and '[' not in s:
        return [x.strip() for x in s.split(',') if x.strip()]
    return [s]

def refs_by_eval(rows, id_col, ref_col):
    out = {}
    for r in rows:
        eid = str(r[id_col]).strip()
        out[eid] = tuple(parse_refs(r.get(ref_col, []))[:3])
    return out

def write_rows(path: Path, out_rows, fmt: str):
    if fmt == 'jsonl':
        with path.open('w', encoding='utf-8') as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    else:
        with path.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            w.writeheader()
            w.writerows(out_rows)

subs = {}
meta = {}
for k, p in PATHS.items():
    if p.exists():
        rows, fmt = read_submission(p)
        subs[k] = rows
        meta[k] = {
            'format': fmt,
            'id_col': detect_id_col(rows),
            'ref_col': detect_ref_col(rows),
            'query_col': detect_query_col(rows),
            'sha256': file_sha256(p),
            'size': p.stat().st_size,
            'n_rows': len(rows),
            'keys': list(rows[0].keys()) if rows else [],
        }
print(json.dumps(meta, ensure_ascii=False, indent=2))

refmaps = {k: refs_by_eval(subs[k], meta[k]['id_col'], meta[k]['ref_col']) for k in subs}
keys = list(refmaps)
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = keys[i], keys[j]
        common = sorted(set(refmaps[a]) & set(refmaps[b]))
        same = sum(refmaps[a][eid] == refmaps[b][eid] for eid in common)
        ratio = (same / len(common)) if common else 0.0
        print(f'{a} vs {b}: same_top3={same}/{len(common)} ({ratio:.2%})')

# dual fuse
rows_a = subs['m4']; rows_b = subs['m4_chat']
id_a, ref_a = meta['m4']['id_col'], meta['m4']['ref_col']
id_b, ref_b = meta['m4_chat']['id_col'], meta['m4_chat']['ref_col']
map_b = {str(r[id_b]).strip(): r for r in rows_b}
out_rows = []
for r in rows_a:
    eid = str(r[id_a]).strip()
    ra = parse_refs(r.get(ref_a, []))
    rb = parse_refs(map_b.get(eid, {}).get(ref_b, []))
    score = defaultdict(float)
    for rank, docid in enumerate(ra[:10], start=1):
        score[docid] += 1.0 / (20 + rank)
    for rank, docid in enumerate(rb[:10], start=1):
        score[docid] += 1.0 / (20 + rank)
    top3 = [doc for doc, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))[:3]]
    nr = dict(r)
    nr[ref_a] = top3 if isinstance(r.get(ref_a), list) else str(top3)
    out_rows.append(nr)
p_dual = SUB_DIR / 'submission_m4_dualfuse.csv'
write_rows(p_dual, out_rows, meta['m4']['format'])
print('saved:', p_dual)
print('sha256:', file_sha256(p_dual))