#!/usr/bin/env python
"""Build UNSEEN held-out test sets for the remaining classes from FREE sources.

  US <- CMU ARCTIC (bdl, clb, rms, slt)  + SLR45 Free ST American English
  CA <- CMU ARCTIC (jmk)                 [1 speaker bootstrap - documented]
  UK <- OpenSLR-83 UK dialects (Southern/Northern/Midlands/Scottish/Welsh; NOT Irish)
  NG <- OpenSLR-70 Nigerian English (Lagos + London)

All disjoint (corpus/speaker/channel) from training sources. Output ->
scratchpad/free_out/<CC>/{audio/,manifest.csv}. Upload is done separately after review.
"""
import io, os, csv, sys, zipfile, tarfile, random, urllib.request, shutil, traceback
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

random.seed(42)
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(ROOT, "free_out")
DL   = os.path.join(ROOT, "dl")
os.makedirs(DL, exist_ok=True)
CAP_CMU = 150     # clips/speaker for studio single-speaker corpora
CAP_CROWD = 12    # clips/speaker for crowdsourced many-speaker corpora

def log(*a): print(*a, flush=True)

def sniff_ext(b):
    if b[:4]==b"RIFF": return "wav"
    if b[:4]==b"fLaC": return "flac"
    if b[:4]==b"OggS": return "ogg"
    if b[:3]==b"ID3" or b[:2] in (b"\xff\xfb",b"\xff\xf3",b"\xff\xf2"): return "mp3"
    return "wav"

def add(manifests, cc, fname, source, speaker, gender, accent, blob, adir):
    with open(os.path.join(adir, fname), "wb") as f: f.write(blob)
    manifests.setdefault(cc, []).append(dict(fname=fname, source=source,
        speaker=speaker, gender=gender, age="", accent=accent))

def dl(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        log("  cached", os.path.basename(dest)); return dest
    log("  downloading", url)
    urllib.request.urlretrieve(url, dest)
    log("  ->", round(os.path.getsize(dest)/1e6,1), "MB")
    return dest

# ---------- CMU ARCTIC (US + CA) ----------
CMU = {"bdl":("US","M"),"clb":("US","F"),"rms":("US","M"),"slt":("US","F"),"jmk":("CA","M")}
def build_cmu(manifests):
    log("[CMU ARCTIC] US(bdl,clb,rms,slt) + CA(jmk)")
    for spk,(cc,gender) in CMU.items():
        try:
            p = hf_hub_download("NathanRoll/cmu-arctic", f"data/{spk}-00000-of-00001.parquet", repo_type="dataset")
            t = pq.read_table(p); d = t.to_pydict()
            acol = next((c for c in d if d[c] and isinstance(d[c][0],dict) and "bytes" in d[c][0]), "audio")
            idcol = "file_id" if "file_id" in d else ("path" if "path" in d else acol)
            n=len(d[acol]); idx=list(range(n)); random.shuffle(idx); idx=idx[:CAP_CMU]
            adir=os.path.join(OUT,cc,"audio"); os.makedirs(adir,exist_ok=True)
            for i in idx:
                a=d[acol][i]; b=a["bytes"] if isinstance(a,dict) else a
                if not b: continue
                fid=str(d[idcol][i]).split("/")[-1].split(".")[0] if idcol in d else f"{i:05d}"
                add(manifests,cc,f"cmuarctic_{spk}_{fid}.{sniff_ext(b)}","CMU-ARCTIC",
                    f"cmuarctic_{spk}",gender,f"CMU-ARCTIC-{cc}",b,adir)
            log(f"  {spk}->{cc}: {min(n,CAP_CMU)} clips")
        except Exception as e:
            log(f"  [ERR] {spk}:", repr(e))

# ---------- OpenSLR zip/tgz crowdsourced (UK, NG) ----------
def speaker_of(fn):
    base=os.path.splitext(os.path.basename(fn))[0]
    parts=base.split("_")
    return "_".join(parts[:-1]) if len(parts)>=2 else base

def build_openslr_archive(manifests, cc, source, accent_prefix, url, is_tar=False):
    log(f"[{source}] -> {cc}  ({os.path.basename(url)})")
    dest=os.path.join(DL, os.path.basename(url))
    try: dl(url,dest)
    except Exception as e: log("  [ERR] download:",repr(e)); return
    # gather (name, bytes) for wavs, grouped by speaker
    by_spk={}
    if is_tar:
        tf=tarfile.open(dest,"r:*")
        members=[m for m in tf.getmembers() if m.name.lower().endswith((".wav",".flac"))]
        for m in members: by_spk.setdefault(speaker_of(m.name),[]).append(("tar",m))
        reader=lambda x: tf.extractfile(x[1]).read()
    else:
        zf=zipfile.ZipFile(dest)
        names=[n for n in zf.namelist() if n.lower().endswith((".wav",".flac"))]
        for n in names: by_spk.setdefault(speaker_of(n),[]).append(("zip",n))
        reader=lambda x: zf.read(x[1])
    log(f"  speakers={len(by_spk)} total_wavs={sum(len(v) for v in by_spk.values())}")
    adir=os.path.join(OUT,cc,"audio"); os.makedirs(adir,exist_ok=True)
    for spk,items in sorted(by_spk.items()):
        random.shuffle(items); items=items[:CAP_CROWD]
        for kind,ref in items:
            b=reader((kind,ref))
            if not b: continue
            fid=os.path.splitext(os.path.basename(ref if isinstance(ref,str) else ref.name))[0]
            gender = "F" if "female" in url else ("M" if "male" in url else "U")
            add(manifests,cc,f"{accent_prefix}_{fid}.{sniff_ext(b)}",source,
                f"{accent_prefix}_{spk}",gender,f"{source}-{cc}",b,adir)
    log(f"  {cc} += {sum(min(len(v),CAP_CROWD) for v in by_spk.values())} clips (cap {CAP_CROWD}/spk)")

def main():
    manifests={}
    try: build_cmu(manifests)
    except Exception: traceback.print_exc()

    # US supplement: SLR45 Free ST American English (mobile channel)
    try:
        build_openslr_archive(manifests,"US","SLR45-ST-AmEng","slr45",
            "https://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS.tgz", is_tar=True)
    except Exception: traceback.print_exc()

    # UK: OpenSLR-83 dialects (exclude Irish English = Republic of Ireland)
    UK83=["southern_english_female","northern_english_female","midlands_english_female",
          "midlands_english_male","scottish_english_female","welsh_english_female"]
    for name in UK83:
        try:
            build_openslr_archive(manifests,"UK","SLR83-UKdialect",f"slr83_{name}",
                f"https://www.openslr.org/resources/83/{name}.zip")
        except Exception: traceback.print_exc()

    # NG: OpenSLR-70 Nigerian English
    for name in ["en_ng_female","en_ng_male"]:
        try:
            build_openslr_archive(manifests,"NG","SLR70-NgEng",f"slr70_{name}",
                f"https://www.openslr.org/resources/70/{name}.zip")
        except Exception: traceback.print_exc()

    log("\n=== WRITING MANIFESTS ===")
    for cc,rows in manifests.items():
        rows.sort(key=lambda x:x["fname"])
        with open(os.path.join(OUT,cc,"manifest.csv"),"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=["fname","source","speaker","gender","age","accent"])
            w.writeheader(); w.writerows(rows)
        nspk=len(set(r["speaker"] for r in rows))
        log(f"  {cc}: {len(rows)} clips, {nspk} speakers, sources={sorted(set(r['source'] for r in rows))}")
    log("DONE. OUT=",OUT)

if __name__=="__main__":
    main()
