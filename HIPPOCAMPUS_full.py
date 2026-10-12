# =====================================================================================================================================
# AKBASCORE MAM — THE SILICON HIPPOCAMPUS · PHYSICAL MEMORY CARTRIDGES FOR A FROZEN LANGUAGE MODEL — QWEN2.5-7B · PART 1 / 3 · ENGINE
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 12 October 2026
# Run PART 1 → PART 2 → PART 3 as three consecutive Google Colab cells in the SAME runtime (GPU: A100 40 GB recommended; ~5 GB free disk).
# PART 1 embeds the locked TEST627 demo engine VERBATIM (SHA-256 verified) and executes it unchanged in its own module namespace.
# The engine bootstraps its own chain: 626.py → 624.py → 610.py → 566.py / 575.py. PART 1 downloads 626.py and 624.py from the
# repository, verifies their SHA-256 and places them at the local paths the engine reads; 624.py verifies 610.py by SHA-256 and 610.py
# verifies 566.py / 575.py by git blob. At start-up the engine itself WRITES the 1,453 physical CUT3 cartridge files (frozen Qwen2.5-7B),
# builds the label-free PCA128 + FAISS IVF128 address index, runs its locked PEAK50 READ acceptance (48 original + 48 new-wording
# questions answered from the disk files) and audits all 1,453 files. PART 2 adds the fail-closed RUN pipeline and the four figures.
# PART 3 runs a GPU-free self-test and starts the Gradio demonstration (one RUN button, no inputs).
# AKBASCORE RESEARCH SOFTWARE LICENSE · Copyright © 2026 Mustafa Akbaş · https://github.com/ceceli33/titan-cognitive-core-v2
#
# ENGINE         : TEST627 demo engine (locked; CUT3 / PCA128 / FAISS IVF128 / PEAK50 / SHA-256 unchanged), embedded byte-for-byte as
#                  ENGINE_SRC and executed completely once at start-up. Nothing in the memory, index, routing, reranking or answering is
#                  modified, recalibrated or re-implemented.
# DEMO CALLS     : the RUN pipeline only calls engine functions — init(·,3) (CUT3 WRITE), question_rep / apply (question frame),
#                  candidate626 / features626 / rerank626 with the locked PEAK50 setting (READ address), sha627 + torch.load (SSD file),
#                  answer(memory, question) (READ answer).
# INSTRUMENTATION: (1) a recording-only forward pre-hook on the causal-LM module, attached only while an engine call runs (kind and length
#                  of every model input, installed cache length); (2) an all-parameter weight guard plus the engine's full-weight SHA-256
#                  and TEST566 sentinel; (3) re-derivation of every reported quantity from raw records.
# =====================================================================================================================================
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,gc,types,contextlib,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
DEMO_TITLE="AKBASCORE MAM — The Silicon Hippocampus"
DEMO_SUB="Physical memory cartridges for a frozen language model · WRITE · SSD · READ"
DEMO_SHORT="AKBASCORE MAM · Silicon Hippocampus · Qwen2.5-7B · 1453 CUT3 cartridges"
MODEL_FAMILY="Qwen2.5-7B-Instruct"
AUTHOR="Mustafa Akbaş";ORG="AkbasCore AI Teknoloji";DATE_TXT="12 October 2026";DATE_ISO="2026-10-12";AUTHOR_PLACE="Mersin, Türkiye"
COPYRIGHT="Copyright © 2026 Mustafa Akbaş";LICENSE_NAME="AKBASCORE RESEARCH SOFTWARE LICENSE"
LICENSE_URL="https://github.com/ceceli33/titan-cognitive-core-v2"
REPO_BLOB="https://github.com/ceceli33/titan-cognitive-core-v2/blob/main/";REPO_RAW="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"
URL_627=REPO_BLOB+"627.py";URL_627LOG=REPO_BLOB+"627.log";URL_626=REPO_BLOB+"626.py";URL_626LOG=REPO_BLOB+"626.log";URL_624=REPO_BLOB+"624.py";URL_610=REPO_BLOB+"610.py"
NAME_NOTE=("“The Silicon Hippocampus” names the role of the system: a memory is formed once (WRITE), kept outside the model (SSD) and recalled later (READ). "
           "It is a name, not a claim of biological equivalence.")
PARADIGM="Removable physical memory for a frozen language model"
DISCOVERY="Discovered and developed by Mustafa Akbaş"
SCI_QUESTION=("Can a frozen language model keep each fact as its own internal key/value state in an independent file on disk, find the right file later "
              "from the question alone, and answer from it without being given the source text again?")
PRIOR_PUB=dict(title="AKBASCORE MAM v1.0 — Source-Free Persistent Memory for Frozen LLMs via DC6 Consolidation (Mistral-7B)",doi="10.5281/zenodo.23245358",url="https://doi.org/10.5281/zenodo.23245358",date="8 October 2026")
QWEN_PUB=dict(label="AKBASCORE MAM v1.1 — Qwen2.5-7B-Instruct (DC3) record",doi="10.5281/zenodo.23257347",url="https://doi.org/10.5281/zenodo.23257347",date="9 October 2026")
V12_PUB=dict(label="AKBASCORE MAM v1.2.0 — Qwen2.5-7B CUT3 + FRAM demonstration record",doi="10.5281/zenodo.23282902",url="https://doi.org/10.5281/zenodo.23282902",date="10 October 2026")
PRIORITY=("Public technical demonstration of AKBASCORE MAM: physical CUT3 memory cartridges written by frozen Qwen2.5-7B-Instruct, stored as independent files, "
          "addressed from the question by a label-free geometric index and answered from without the source text. Mustafa Akbaş (AkbasCore AI Teknoloji) "
          f"discovered and developed this memory mechanism (earlier public records: DOI {PRIOR_PUB['doi']}, {PRIOR_PUB['date']}; DOI {QWEN_PUB['doi']}, {QWEN_PUB['date']}; "
          f"DOI {V12_PUB['doi']}, {V12_PUB['date']}); this record is dated 12 October 2026.")
CORE_MESSAGE=["WRITE · The frozen model reads each fact once; its own attention key/value state (CUT3) is saved as an independent file.",
              "SSD · 1,453 cartridges are kept as 1,453 separate files with a SHA-256 manifest.",
              "READ · From the question alone, a geometric address in RAM (PCA128 + FAISS IVF128) and the locked PEAK50 rule select one file.",
              "The file is loaded and the frozen model answers from it; the source text is not given again.",
              "RESULTS · Every number comes from the engine in this runtime or from the sealed TEST627 log; the limits are shown with it.",
              "Model weights are not changed."]
CROSS_ALLOWED=(PRIOR_PUB["title"],NAME_NOTE,DEMO_TITLE,"The Silicon Hippocampus","Silicon Hippocampus","SILICON_HIPPOCAMPUS","silicon_hippocampus")
# ---- locked sources ----
ENGINE_FILE="TEST627 demo engine"
ENGINE_SHA256_EXPECTED="ab53e2d58fb3dc5f11409a5112e92feea8762544e4b30b8a77a39b60332de4d1"
ENGINE_BLOB_EXPECTED="66c5abfd024091eebd86a5f3676010dceddd6959"
CHAIN={"626.py":dict(sha256="b31481abebb73b98567253ff7d40b2d510e9e89beb512ff7f0bfc47a69fbb9b8",local="/content/TEST626_NATIVE_RANKING_CALIBRATION.py"),
       "624.py":dict(sha256="39438a48217f8ed52db7df94ab5c9631dd9b3a09acf15f404eaa75f30d922387",local="/content/TEST624_1453_PHYSICAL_CUT3.py")}
SHA610_EXPECTED="a62611a4ffd91c2fcf03796c31da2af11ccd8141345655b562f3a614568af3e7"
SOURCE_BLOBS_EXPECTED={"566.py":"b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71","575.py":"569c92c61224ea408cdb6e81dbe45dde06dc05fd"}
WEIGHT_SHA_EXPECTED="b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";ARCH=(28,3584,28,4,128);CUT_EXPECTED=3;PREFIX_EXPECTED=29
N_CART=1453;N_ORIG=96;N_CASES=48;RANK_EXPECTED=128;NLIST_EXPECTED=128;LOCKED_CAND=(2,16,16);LOCKED_RERANK=("PEAK50",0.5,0.0)
ENGINE_ROOT="/content/AKBASCORE_TEST624";ENGINE_OUT_NAME="AKBASCORE_MAM_DEMO_ENGINE_RESULT.json"
CANON_GPU="NVIDIA A100-SXM4-40GB"
def arch_txt(a=ARCH):return f"{a[0]} layers · hidden {a[1]} · {a[2]} Q heads · {a[3]} KV heads · head {a[4]}"
# ---- sealed record: repository 627.log (TEST627, 2026-10-12); reproduced verbatim, never recomputed ----
TEST627_LOG=dict(name="TEST627",log="627.log",log_sha256="c909ce23d98bfc883641c82144adceacdaf7de13fc56e00a6153fbbb46c00687",decision="READY_FOR_DEMO_WITH_DISCLOSED_NONBLIND_LIMITATION",
    e2e={"ORIGINAL":{"cases":48,"selected_correct":48,"answer_correct":48,"errors":0,"touched_mean":102.0,"address_p50_ms":23.8674,"load_p50_ms":17.8692,"answer_p50_ms":236.6135},
         "ALTERNATE":{"cases":48,"selected_correct":48,"answer_correct":48,"errors":0,"touched_mean":117.104,"address_p50_ms":25.045,"load_p50_ms":17.5035,"answer_p50_ms":235.2036}},
    storage={"cartridges":1453,"disk_pool_bytes":3973450961,"avg_file_bytes":2734653.1,"min_file_bytes":2252853,"max_file_bytes":2883637,"total_tensor_payload_bytes":3949568000,"avg_tensors_per_file":56.0,
             "gzip_sample_files":32,"gzip_sample_ratio":0.796091,"gzip_sample_all_bitwise_restored":True},
    index={"native_positions":26738,"ivf_lists":128,"index_build_ms":26.21,"pca_ms":138.85},
    integrity={"source_sha_lock":True,"manifest_1453_sha_verified":True,"weight_unchanged":True,"sentinel_unchanged":True,"zero_trainable":True,"no_hooks":True,"tensor_load_errors_zero":True,"lossless_sample_bitwise":True})
# ---- sealed record: repository 626.log (TEST626): how the reranker came to be locked (disclosed, not hidden) ----
TEST626_LOG=dict(name="TEST626",log="626.log",log_sha256="653ff1a90a9ad0acd611d0040f1f6d83e710e2c22a4a6c14a945ad05656bc4b8",locked_by_own_rule="PEAK25",
    fram_only_new_wording="40/48",peak25_new_wording="47/48",peak50_new_wording="48/48",e2e="47/48 + 47/48",decision="REPORT_ACTUAL_ACCURACY_AND_CONTINUE_DIAGNOSIS")
SCOPE={
 "demonstrated":[
  "WRITE (re-executed live for the fact of the case): the engine's init(fact, 3) runs one token-id forward of the instruction prefix + the fact and one all-zero-embedding forward; the resulting 28 × (K, V) cartridge is compared bitwise with the file written at start-up.",
  "SSD: all 1,453 cartridge files are present with the manifest sizes; the file of the case and the selected file are re-hashed against the SHA-256 manifest; the files contain no fact text.",
  "READ: from the question alone (one question-frame forward, no fact text), the label-free PCA128 + FAISS IVF128 address and the locked PEAK50 rule select one cartridge; its file is loaded from disk and the frozen model answers over it.",
  "The model input while answering is the question template only, over the loaded cartridge; the source text is not given again.",
  "Control in every run: the same question over the runner-up candidate's file (diagnostic; whether the expected answer appears is reported).",
  "Model weights are unchanged (all-parameter guard before and after every run; full-weight SHA-256 at start-up equal to the reference)."],
 "startup":[
  "At start-up in this runtime the locked engine wrote the 1,453 cartridge files, built the PCA128 + FAISS IVF128 index without labels, ran its PEAK50 acceptance on 48 original + 48 new-wording questions answered from the disk files, and audited all 1,453 files (sizes, SHA-256, tensor structure)."],
 "archived":[
  "TEST627 (sealed, 627.log): READY_FOR_DEMO_WITH_DISCLOSED_NONBLIND_LIMITATION · original wording 48/48 · new wording 48/48 · 0 errors · 1,453 files · 3,973,450,961 bytes · integrity 8/8.",
  "TEST626 (sealed, 626.log): its own pre-declared rule locked PEAK25 (47/48 + 47/48 answered); FRAM-only ranking chose the right file for 40/48 new-wording questions; PEAK50 was fixed afterwards."],
 "not_established":[
  "Blind generalisation: the 96 questions reuse the same 48 facts, and PEAK50 was fixed after the TEST626 results were known.",
  "Behaviour at millions of cartridges: 1,453 files do not demonstrate it.",
  "Cold-SSD latency: disk reads may be served from the operating-system page cache.",
  "Facts outside the synthetic capital-of-country template and other language models.",
  "Hardware other than the NVIDIA A100 40 GB (Colab) used for TEST627 and this demonstration; timings on other machines are not measured."],
 "notes":[
  "A cartridge is much larger than its text (about 2.7 MB per fact): it stores the model's own working state, not a compact text copy.",
  "The RAM side holds the PCA basis, the FAISS IVF index and the native layer-3 vectors used for reranking; these are reported with their sizes.",
  "The lossless gzip figure is measured on 32 sample files only; the full-pool value is an estimate.",
  "Expected answers come from the panel construction and are audit metadata; they are not an input to WRITE, addressing, selection or answering.",
  "Hashes are artifact-integrity seals; they are not scientific proof or third-party verification."]}
class AuditFail(RuntimeError):pass
def jsafe(o):
    if isinstance(o,dict):return{str(k):jsafe(v)for k,v in o.items()}
    if isinstance(o,(list,tuple)):return[jsafe(v)for v in o]
    if hasattr(o,"item")and not isinstance(o,(str,bytes)):
        try:o=o.item()
        except Exception:pass
    if isinstance(o,float):return o if math.isfinite(o)else f"non-finite:{o}"
    if isinstance(o,(str,int,bool))or o is None:return o
    return str(o)
def canon(o):return json.dumps(jsafe(o),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
def norm_ans(s):return " ".join(re.sub(r"[^\w\s-]"," ",str(s).casefold().strip(),flags=re.UNICODE).split())
def hit_ans(s,g):return re.search(r"(?<!\w)"+re.escape(norm_ans(g))+r"(?!\w)",norm_ans(s))is not None
def git_blob_sha1(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
# ---- claim-discipline scanner: applied to every generated text (figures, logs, JSON). Panel names are removed first. ----
ALLOWED_TESTS={"566","575","610","624","626","627"}
CLAIM_RULES=[("capacity-overclaim",r"(?i)\bunlimited\b|\binfinite\b"),("universal-claim",r"(?i)\buniversal\b"),("human-like-claim",r"(?i)human-like"),
    ("model-independence-claim",r"(?i)model-independent"),("all-transformers-claim",r"(?i)all transformers|every transformer|all (?:large )?language models"),("solved-claim",r"(?i)\bsolved\b"),
    ("perfect-claim",r"(?i)\bperfect\b"),("world-first-claim",r"(?i)world'?s first|first in the world|independently proven|third-party verified"),
    ("legal-status-claim",r"(?i)\bpatent(ed|-pending)\b|non-obvious|\bpatentab"),("metaphor",r"(?i)\bbrain\b|\bquantum\b|\bneuron|hippocamp"),
    ("complexity-claim",r"O\(1\)"),("compression-claim",r"(?i)compressed (?:memory|superposition)|memory compression"),
    ("hallucination-claim",r"(?i)(?:prevents?|eliminates?|no) hallucination")]
LEAK_RULES=[("mistral-name",r"(?i)mistral"),("mistral-cut",r"\bH6\b|\bDC6\b|\b(?:layers?|L)\s?0[–-]6\b|\bL7[–-]31\b|\blayers 7[–-]31\b"),
    ("mistral-tests",r"TEST\s?56[03]\b"),("mistral-arch-words",r"\b32 (?:layers|L\b|Q)|\b8 KV\b|\b27-token")]
LEAK_RULES_NUM=[("mistral-hidden",r"(?<![\d.,])4096(?![\d.,])")]
def _scan_norm(text):
    for a in sorted(CROSS_ALLOWED,key=len,reverse=True):text=text.replace(a,"<cited>")
    text=re.sub(r"[0-9a-fA-F]{12,}","<hex>",text);return text
def claim_scan(name,text,allow=(),numeric=False):
    for a in allow:
        if a:text=text.replace(a,"<allowed>")
    text=_scan_norm(text)
    hits=[(name,lab,m.group(0))for lab,p in CLAIM_RULES for m in re.finditer(p,text)]
    hits+=[(name,"unexpected-test-number",m.group(0))for m in re.finditer(r"TEST\s?(\d+)",text)if m.group(1)not in ALLOWED_TESTS]
    hits+=[(name,"foreign-model-"+lab,m.group(0))for lab,p in LEAK_RULES+(LEAK_RULES_NUM if numeric else[])for m in re.finditer(p,text)]
    return hits
def strip_words(text,words):
    ws=sorted({w for w in words if w and len(w)>=2},key=len,reverse=True)
    if not ws:return text
    out=text
    for a in range(0,len(ws),400):out=re.sub(r"(?<![A-Za-z0-9])(?:"+"|".join(map(re.escape,ws[a:a+400]))+r")(?![A-Za-z0-9])","",out)
    return out
def cpu_selftest():
    n=0
    assert hit_ans("Melket","Melket") and hit_ans("The capital is Melket.","melket") and not hit_ans("Melketon","Melket");n+=1
    ok_txt=(DEMO_TITLE+" "+DEMO_SUB+" "+NAME_NOTE+" Qwen2.5-7B-Instruct TEST566 TEST575 TEST610 TEST624 TEST626 TEST627 CUT3 PCA128 FAISS IVF128 PEAK50 1,453 cartridges 28 layers "
            +PARADIGM+" "+DISCOVERY+" "+PRIORITY+" "+SCI_QUESTION+" "+PRIOR_PUB["title"]+" "+" ".join(sum(SCOPE.values(),[])+CORE_MESSAGE)+" lossless gzip sample ratio 0.796")
    assert claim_scan("t",ok_txt,numeric=True)==[],claim_scan("t",ok_txt,numeric=True);n+=1
    for b in("unlimited memory","universal memory","human-like memory","model-independent","works on all transformers","memory solved","perfect recall","world's first","patent-pending",
             "third-party verified","brain","quantum","neurons","hippocampus","O(1)","compressed memory","prevents hallucination","TEST528","TEST560","TEST619","Mistral","H6 residual","DC6","layers 0–6","32 layers","8 KV heads"):
        assert claim_scan("t",b),b;n+=1
    assert claim_scan("t","hidden 4096",numeric=True) and not claim_scan("t","hidden 4096") and not claim_scan("t","bytes 2,734,096 · sha 4096ab12cd34ef56",numeric=True);n+=1
    assert strip_words("Zorvan BRAIN carried",["BRAIN"])=="Zorvan  carried";n+=1
    e=TEST627_LOG["e2e"];assert all(e[p]["answer_correct"]==e[p]["cases"]==48 and e[p]["errors"]==0 for p in e) and all(TEST627_LOG["integrity"].values());n+=1
    assert TEST627_LOG["storage"]["cartridges"]==N_CART and TEST627_LOG["index"]["ivf_lists"]==NLIST_EXPECTED;n+=1
    return n
#<<CONST_END>>
#<<ENGINE_BEGIN>>
for _m,_p in[("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("gradio","gradio"),("matplotlib","matplotlib"),("PIL","pillow"),("faiss","faiss-cpu")]:
    if importlib.util.find_spec(_m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",_p])
import numpy as np,torch,transformers,gradio as gr
QUIET=False
def say(*a):
    if not QUIET:print(*a,flush=True)
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
say("="*140);say(DEMO_TITLE+" — "+MODEL_FAMILY+" · PART 1 / 3 · ENGINE");say(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");say(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");say("="*140)
say(f"[1/7] CPU self-test PASS ({cpu_selftest()} checks)")
# ---------------- TEST627 demo engine — embedded verbatim (do not edit; its SHA-256 and git blob SHA-1 are verified below) ----------------
ENGINE_SRC=r'''# AKBASCORE MAM — LOCKED DEMO ENGINE (NO UI / NO GRADIO)
# Executable Colab single-cell backend, based on the successful TEST627 code.
# Claude owns ALL Gradio interface and wiring. Do not change CUT3/FAISS/PEAK50/model/panel.
# Run this script to produce /content/AKBASCORE_TEST624/AKBASCORE_MAM_DEMO_ENGINE_RESULT.json
# Input: none. Output: JSON report with 96 physical disk answers, integrity, storage audit.
# This backend does not install, import, create, or launch any UI.

# TEST627 — LOCKED PEAK50 + PHYSICAL CUT3 FINAL ACCEPTANCE + LOSSLESS STORAGE AUDIT
# Single Colab cell. Original TEST626 / TEST624 backbone, model, CUT3, FAISS and original disk files unchanged.
import os,time,json,hashlib,urllib.request,math,io,gzip,statistics,collections
from pathlib import Path
import numpy as np,torch
T627=time.perf_counter();ROOT627=Path('/content/AKBASCORE_TEST624');OUT627=ROOT627/'AKBASCORE_MAM_DEMO_ENGINE_RESULT.json'
LOCAL627=Path('/content/TEST626_NATIVE_RANKING_CALIBRATION.py');URL627='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/626.py'
def log627(*a):print(*a,flush=True)
def med627(a):return round(float(np.median(a)),4) if a else None
def sha627(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
NEED627=('candidate626','features626','rerank626','CFG626','QUEST','PANEL624','MANIFEST','POOL','CAR','model','WB','SB','src','LOCK','TOTAL','answer','hit','DEVICE','weight_sha','sentinel','question_rep','apply','FIT','tok')
if not all(k in globals() for k in NEED627):
 if LOCAL627.exists():b627=LOCAL627.read_bytes()
 else:b627=urllib.request.urlopen(urllib.request.Request(URL627,headers={'User-Agent':'TEST627'}),timeout=120).read()
 marker627="log626('[1/6] PREDECLARED ORIGINAL QUERY PANEL"
 assert b627.decode().count(marker627)==1,'TEST626 BACKBONE BOUNDARY ERROR'
 log627('BOOTSTRAP_TEST626_SHA256',hashlib.sha256(b627).hexdigest())
 exec(compile(b627.decode().split(marker627,1)[0],'TEST626_BACKBONE.py','exec'),globals())
assert TOTAL==1453 and len(MANIFEST)==1453 and len(QUEST)==48 and len(PANEL624)==48
LOCKED627=next(c for c in CFG626 if c[0]=='PEAK50')
assert LOCKED627==('PEAK50',.5,0.),'RERANK_LOCK_MISMATCH'
assert not any(p.requires_grad for p in model.parameters())
log627('='*112);log627('TEST627 — PEAK50 LOCKED / 1453 PHYSICAL CUT3 / LOSSLESS STORAGE AUDIT');log627('='*112)
log627('[1/7] LOCKED PEAK50 ROUTING / ORIGINAL 48 + ALTERNATE 48')
ROUTES627=[]
for panel in ('ORIGINAL','ALTERNATE'):
 for i,case in enumerate(PANEL624):
  if panel=='ORIGINAL':question=case['question'];q=QUEST[i]
  else:
   country=CAR[case['target']]['fact'].split('The current capital of ',1)[1].split(' is ',1)[0]
   question=f'Name the present capital of {country}.' if i%2==0 else f'Which city currently serves as the capital of {country}?'
   rep,_=question_rep(question);q=apply(rep[3],FIT).detach().float().cpu().contiguous()
  t=time.perf_counter();ids,route=candidate626(q);fs=features626(q,ids);ranking,_=rerank626(ids,fs,LOCKED627)
  ROUTES627.append(dict(panel=panel,case=i,question=question,target=case['target'],selected=ranking[0] if ranking else None,
   top5=ranking[:5],gold_candidate=case['target'] in ids,selected_correct=bool(ranking and ranking[0]==case['target']),
   touched=route['touched'],address_ms=(time.perf_counter()-t)*1000))
  if len(ROUTES627)%16==0:log627('LOCKED_ROUTE',len(ROUTES627),'/96','correct',sum(x['selected_correct'] for x in ROUTES627))
log627('[2/7] PHYSICAL DISK + SHA + FROZEN ANSWERS / 96')
E2E627=[]
for r in ROUTES627:
 row=dict(panel=r['panel'],case=r['case'],target=r['target'],selected=r['selected'],selected_correct=r['selected_correct'])
 try:
  ci=r['selected'];assert ci is not None,'NO_CANDIDATE';p=POOL/f'cut3_{ci:05d}.pt'
  t=time.perf_counter();assert sha627(p)==MANIFEST[ci]['sha256'],'SHA_MISMATCH'
  stored=torch.load(p,map_location='cpu',weights_only=True);mem=tuple((a.to(DEVICE),b.to(DEVICE)) for a,b in stored)
  row['load_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
  with torch.no_grad():ans=answer(mem,r['question'])
  row.update(answer=str(ans),correct=bool(hit(ans,PANEL624[r['case']]['gold'])),answer_ms=(time.perf_counter()-t)*1000)
  del mem,stored
 except Exception as e:row.update(correct=False,error=repr(e))
 E2E627.append(row)
 if len(E2E627)%16==0:log627('PHYSICAL_E2E',len(E2E627),'/96','correct',sum(x['correct'] for x in E2E627))
log627('[3/7] ALL 1453 PHYSICAL FILE SIZES + SHA256')
SIZES627=[];BAD627=[]
for i,r in enumerate(MANIFEST):
 p=POOL/r['file'];size=p.stat().st_size if p.exists() else -1;ok=size==r['bytes'] and sha627(p)==r['sha256'] if size>=0 else False
 SIZES627.append(size)
 if not ok:BAD627.append(i)
 if (i+1)%300==0:log627('FILE_AUDIT',i+1,'/1453')
log627('[4/7] TENSOR STRUCTURE / ALL FILES; LOSSLESS GZIP ON 32 PREDECLARED SAMPLES')
# Read-only inspection: no write-back, no conversion of tensor dtype, no deletion of original files.
TENSOR627=collections.Counter();DTYPE627=collections.Counter();SHAPE627=collections.Counter();TENSOR_BYTES627=0;TENSOR_COUNTS627=[]
SAMPLE627=sorted(set(np.linspace(0,TOTAL-1,32,dtype=int).tolist()))
COMP627=[];LOAD_ERR627=[]
def tensors627(o):
 if torch.is_tensor(o):yield o
 elif isinstance(o,(tuple,list)):
  for v in o:yield from tensors627(v)
 elif isinstance(o,dict):
  for v in o.values():yield from tensors627(v)
for i,r in enumerate(MANIFEST):
 p=POOL/r['file']
 try:
  obj=torch.load(p,map_location='cpu',weights_only=True)
  tt=list(tensors627(obj));nb=sum(t.numel()*t.element_size() for t in tt)
  TENSOR_BYTES627+=nb;TENSOR_COUNTS627.append(len(tt))
  for t in tt:
   DTYPE627[str(t.dtype)]+=t.numel()*t.element_size()
   SHAPE627[str(tuple(t.shape))]+=1
  TENSOR627['tensor_count']+=len(tt);TENSOR627['numel']+=sum(t.numel() for t in tt)
  if i in SAMPLE627:
   raw=p.read_bytes();t=time.perf_counter();packed=gzip.compress(raw,compresslevel=6,mtime=0);comp_ms=(time.perf_counter()-t)*1000
   t=time.perf_counter();restored=gzip.decompress(packed);decomp_ms=(time.perf_counter()-t)*1000
   ok=hashlib.sha256(restored).digest()==hashlib.sha256(raw).digest() and restored==raw
   COMP627.append(dict(id=i,raw_bytes=len(raw),gzip_bytes=len(packed),ratio=round(len(packed)/len(raw),6),
    compress_ms=round(comp_ms,3),decompress_ms=round(decomp_ms,3),bitwise_restored=ok,tensor_bytes=nb,
    overhead_bytes=len(raw)-nb,dtypes=sorted({str(t.dtype) for t in tt}),shapes=[list(t.shape) for t in tt[:4]]))
  del obj,tt
 except Exception as e:LOAD_ERR627.append(dict(id=i,error=repr(e)))
 if (i+1)%300==0:log627('TENSOR_AUDIT',i+1,'/1453')
RAW627=sum(SIZES627);RATIO627=sum(x['gzip_bytes'] for x in COMP627)/sum(x['raw_bytes'] for x in COMP627) if COMP627 else None
STORAGE627=dict(cartridges=TOTAL,disk_pool_bytes=RAW627,disk_pool_GB_decimal=round(RAW627/1e9,6),
 avg_file_bytes=round(float(np.mean(SIZES627)),2),median_file_bytes=med627(SIZES627),
 min_file_bytes=min(SIZES627),max_file_bytes=max(SIZES627),total_tensor_payload_bytes=TENSOR_BYTES627,
 file_container_overhead_bytes=RAW627-TENSOR_BYTES627,avg_tensors_per_file=round(float(np.mean(TENSOR_COUNTS627)),3),
 tensor_dtype_bytes=dict(DTYPE627),common_shapes=SHAPE627.most_common(12),
 lossless_gzip_sample_files=len(COMP627),gzip_sample_ratio=round(RATIO627,6) if RATIO627 else None,
 gzip_sample_estimated_pool_bytes=int(RAW627*RATIO627) if RATIO627 else None,
 gzip_sample_estimate_not_measured_full_pool=True,
 gzip_sample_all_bitwise_restored=all(x['bitwise_restored'] for x in COMP627),
 gzip_compress_p50_ms=med627([x['compress_ms'] for x in COMP627]),gzip_decompress_p50_ms=med627([x['decompress_ms'] for x in COMP627]),
 sample=COMP627,load_errors=LOAD_ERR627)
log627('TEST627_STORAGE',json.dumps({k:v for k,v in STORAGE627.items() if k not in ('sample','common_shapes','load_errors')}))
log627('[5/7] INTEGRITY / NO TRAINING / NO HOOKS')
HOOKS627=any(m._forward_hooks or m._forward_pre_hooks for layer in model.model.layers for m in (layer,layer.self_attn,layer.self_attn.q_proj))
INTEGRITY627=dict(source_sha_lock=hashlib.sha256(src).hexdigest()==LOCK,manifest_1453_sha_verified=not BAD627,
 weight_unchanged=weight_sha()==WB,sentinel_unchanged=sentinel()==SB,
 zero_trainable=not any(p.requires_grad for p in model.parameters()),no_hooks=not HOOKS627,
 tensor_load_errors_zero=not LOAD_ERR627,lossless_sample_bitwise=bool(COMP627) and all(x['bitwise_restored'] for x in COMP627))
SUM627={p:dict(cases=48,selected_correct=sum(r['selected_correct'] for r in ROUTES627 if r['panel']==p),
 answer_correct=sum(r['correct'] for r in E2E627 if r['panel']==p),
 errors=sum('error' in r for r in E2E627 if r['panel']==p),
 touched_mean=round(float(np.mean([r['touched'] for r in ROUTES627 if r['panel']==p])),3),
 address_p50_ms=med627([r['address_ms'] for r in ROUTES627 if r['panel']==p]),
 load_p50_ms=med627([r['load_ms'] for r in E2E627 if r['panel']==p and 'load_ms' in r]),
 answer_p50_ms=med627([r['answer_ms'] for r in E2E627 if r['panel']==p and 'answer_ms' in r])) for p in ('ORIGINAL','ALTERNATE')}
RESULT627=dict(test='627',status='COMPLETE' if all(INTEGRITY627.values()) else 'INTEGRITY_FAILURE',
 locked_reranker='PEAK50',locked_candidates=dict(nprobe=2,token_k=16,budget=16),
 summary=SUM627,storage=STORAGE627,integrity=INTEGRITY627,per_query=ROUTES627,physical_answers=E2E627,
 elapsed_seconds=round(time.perf_counter()-T627,2),limitations=[
 'Original 48 questions and alternative 48 wording questions reuse the same previously exposed 48 facts: NOT an independent record holdout.',
 'PEAK50 was chosen after inspecting TEST626 validation results; this test is confirmation on previously exposed facts, not blind generalization.',
 'Lossless gzip ratio extrapolation is from 32 deterministic sampled files only, not actual compressed full pool.',
 'Gzip samples are in-memory, original disk files unchanged; compression adds decompression latency and requires integration testing before deployment.',
 'Disk read may be OS page-cache served, not verified cold SSD.',
 '1453 physical cartridges do not demonstrate million-cartridge scalability.',
 'Storage report excludes model VRAM/RAM and Python allocator overhead.'])
ROOT627.mkdir(parents=True,exist_ok=True)
with open(OUT627,'w',encoding='utf-8') as f:json.dump(RESULT627,f,ensure_ascii=False,indent=2,default=str)
log627('[6/7] FINAL');log627('TEST627_E2E',json.dumps(SUM627));log627('TEST627_INTEGRITY',json.dumps(INTEGRITY627))
log627('TEST627_STORAGE_SUMMARY',json.dumps({k:v for k,v in STORAGE627.items() if k not in ('sample','common_shapes','load_errors')}))
log627('[7/7] DEMO ACCEPTANCE')
ready627=all(INTEGRITY627.values()) and all(SUM627[p]['answer_correct']==48 for p in SUM627)
log627('TEST627_DECISION','READY_FOR_DEMO_WITH_DISCLOSED_NONBLIND_LIMITATION' if ready627 else 'REPORT_ACTUAL_ACCURACY')
log627('TEST627_OUTPUT',str(OUT627));log627('TEST627_STATUS',RESULT627['status']);log627('DEMO_ENGINE_OUTPUT',str(OUT627))
if not all(INTEGRITY627.values()):raise RuntimeError('TEST627_INTEGRITY_FAILURE')
'''
_ENGINE_BYTES=ENGINE_SRC.encode("utf-8");ENGINE_SHA256=hashlib.sha256(_ENGINE_BYTES).hexdigest();ENGINE_BLOB=git_blob_sha1(_ENGINE_BYTES)
if ENGINE_SHA256!=ENGINE_SHA256_EXPECTED or ENGINE_BLOB!=ENGINE_BLOB_EXPECTED:
    raise RuntimeError(f"Embedded TEST627 demo engine changed (SHA-256 {ENGINE_SHA256}). Re-paste PART 1 unchanged.")
say(f"[2/7] Engine source verified · {ENGINE_FILE} · SHA-256 {ENGINE_SHA256} · git blob {ENGINE_BLOB}")
def fetch_chain():
    """Download 626.py and 624.py, verify SHA-256 and place them where the engine reads them (it then never downloads them itself)."""
    got={}
    for name,c in CHAIN.items():
        b=None;lp=Path(c["local"])
        if lp.is_file()and hashlib.sha256(lp.read_bytes()).hexdigest()==c["sha256"]:b=lp.read_bytes()
        else:
            with urllib.request.urlopen(urllib.request.Request(REPO_RAW+name,headers={"User-Agent":"AKBASCORE-MAM-DEMO"}),timeout=120)as f:b=f.read()
            if hashlib.sha256(b).hexdigest()!=c["sha256"]:raise RuntimeError(f"{name} SHA-256 mismatch: {hashlib.sha256(b).hexdigest()}")
            lp.parent.mkdir(parents=True,exist_ok=True);lp.write_bytes(b)
        got[name]=hashlib.sha256(b).hexdigest()
    return got
CHAIN_MEASURED=fetch_chain()
say("[3/7] Engine chain verified · "+" · ".join(f"{k} SHA-256 {v[:16]}…"for k,v in CHAIN_MEASURED.items())+" (placed at the engine's local paths)")
STARTUP_UTC=utc_now()
STARTUP_GUIDE=[("SOURCE_VERIFIED … / TEST566 … / TEST610 … / CASE …","model load + 96-cartridge TEST610 baseline (610.py)"),
               ("[1/7] PHYSICAL CUT3 WRITE · PHYSICAL_WRITE n / 1453","A · WRITE: the frozen model writes the 1,453 cartridge files (624.py)"),
               ("[2/7] LABEL-FREE PCA128 + FAISS IVF128 · INDEX {…}","RAM address index built without labels (624.py)"),
               ("[3/7] FROZEN QUERY PANEL","the 48 predeclared questions are encoded (624.py)"),
               ("TEST627 … [1/7] LOCKED PEAK50 ROUTING · LOCKED_ROUTE n/96","C · READ address on 48 original + 48 new-wording questions"),
               ("[2/7] PHYSICAL DISK + SHA + FROZEN ANSWERS · PHYSICAL_E2E","D · answers from the disk files"),
               ("[3/7] FILE_AUDIT · [4/7] TENSOR_AUDIT","B · SSD: sizes, SHA-256 and tensor structure of all 1,453 files"),
               ("[5/7]–[7/7] · TEST627_STATUS COMPLETE","integrity, result JSON, start-up acceptance decision")]
if"ENG"in globals()and getattr(globals()["ENG"],"__engine_sha256__",None)==ENGINE_SHA256 and globals().get("ENGINE_READY"):
    say("[4/7] Engine already initialised in this runtime — reusing it (no second model load, no second WRITE).");ENGINE_INIT_SECONDS=globals().get("ENGINE_INIT_SECONDS",float("nan"))
else:
    say("[4/7] Executing the TEST627 demo engine unchanged in its own module namespace. Duration on an A100: several minutes (no fixed estimate; the")
    say("      measured duration is printed at the end). The engine prints its own progress lines; they mean:")
    for _a,_b in STARTUP_GUIDE:say(f"        {_a:<52} → {_b}")
    say(f"      start UTC {utc_now()} · the following lines are printed by the engine itself");_t=time.perf_counter()
    ENG=types.ModuleType("akbascore_mam_test627_engine");ENG.__file__=ENGINE_FILE+" (embedded verbatim)";sys.modules[ENG.__name__]=ENG
    exec(compile(ENGINE_SRC,"TEST627_DEMO_ENGINE.py","exec"),ENG.__dict__)
    ENG.__engine_sha256__=ENGINE_SHA256;torch.cuda.synchronize();ENGINE_INIT_SECONDS=time.perf_counter()-_t;ENGINE_READY=True;ENGINE_READY_UTC=utc_now()
    say(f"      engine finished · {ENGINE_INIT_SECONDS:.0f} s measured · end UTC {ENGINE_READY_UTC}")
# ---------------- measurement adapter (read-only with respect to the engine; never alters writing, addressing, selection or answering) ----------------
def hook_module(fn):
    f=getattr(fn,"func",fn);return str(getattr(f,"__module__",None)or type(f).__module__ or"")
def hook_stats(model):
    tot=0;foreign=Counter()
    for m in model.modules():
        for d in(m._forward_hooks,m._forward_pre_hooks,m._backward_hooks):
            for fn in d.values():
                tot+=1;mod=hook_module(fn)
                if not mod.startswith(("transformers","accelerate","torch")):foreign[mod]+=1
    return tot,dict(foreign)
def residual_hooks(model):
    out=[]
    for li,layer in enumerate(model.model.layers):
        for label,module in(("layer",layer),("attention",layer.self_attn),("q_proj",layer.self_attn.q_proj)):
            if module._forward_hooks or module._forward_pre_hooks:out.append(f"L{li}:{label}")
    return out
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values())+list(ENG.__dict__.values()))
def _tensor_bytes(x):
    x=x.detach().contiguous().cpu()
    if x.dtype==torch.bfloat16:x=x.view(torch.uint16)
    return x.numpy().tobytes()
def memory_sha(layers):
    h=hashlib.sha256()
    for k,v in layers:h.update(_tensor_bytes(k));h.update(_tensor_bytes(v))
    return h.hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with open(p,"rb")as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()
class Instrument:
    """Thin measurement adapter around the engine namespace. Every scientific quantity comes from the engine itself."""
    def __init__(self,init_seconds):
        self.kind="ENGINE";m=ENG.model;c=m.config;self.lock=threading.RLock()
        self.info=dict(model_id=ENG.MODEL_ID,arch=[ENG.NL,ENG.H,ENG.QH,ENG.KVH,ENG.HD],dtype=str(next(m.parameters()).dtype).replace("torch.",""),
            attn=getattr(c,"_attn_implementation",None),gpu=torch.cuda.get_device_name(0),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,python=sys.version.split()[0],platform=platform.platform(),
            params=sum(p.numel()for p in m.parameters()),engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256,engine_blob=ENGINE_BLOB,init_seconds=init_seconds,startup_utc=STARTUP_UTC,engine_ready_utc=globals().get("ENGINE_READY_UTC","reused"),
            chain_sha256={"626.py":hashlib.sha256(ENG.b627).hexdigest(),"624.py":hashlib.sha256(ENG.b626).hexdigest(),"610.py":hashlib.sha256(ENG.src).hexdigest()},
            source_blobs={k:git_blob_sha1((ENG.s566 if k=="566.py"else ENG.s575).encode("utf-8"))for k in("566.py","575.py")},
            sentinel_method="TEST566 sentinel: SHA-256 over 16 slices × 256 values of 7 weight tensors (o_proj of layers 0/7/14/21/27, final norm, lm_head)",
            guard_method="all-parameter guard: SHA-256 over (name, shape, dtype, sum, |sum|, sum of squares) of every parameter tensor",
            full_weight_method="weight_sha(): SHA-256 over (name, shape, dtype, raw bytes) of every parameter, computed by the engine at start-up",
            forward_counter="recording-only forward pre-hook on the causal-LM module, attached only during an engine call")
        self.info["weight_sha0"]=ENG.WB;self.info["sentinel0"]=self.sentinel();self.info["sentinel_engine"]=ENG.SB
        self.info["guard0"]=self.guard();self.info["hooks0"]=hook_stats(m)[0]
        p=Path(ENG.OUT627);self.result_path=p;self.result_sha=file_sha(p)if p.is_file()else None;self._startup=self._read_startup()
    # ---- read-only views of engine state ----
    def config(self):
        return dict(MODEL_ID=ENG.MODEL_ID,arch=[ENG.NL,ENG.H,ENG.QH,ENG.KVH,ENG.HD],CUT=CUT_EXPECTED,prefix_tokens=int(ENG.P),n_cartridges=int(ENG.TOTAL),n_original=int(ENG.ORIGINAL),
                    manifest=len(ENG.MANIFEST),rank=int(ENG.RANK),nlist=int(ENG.NLIST),ntok=int(ENG.NTOK),dim=int(ENG.DIM),index_ntotal=int(ENG.INDEX.ntotal),
                    candidates=[int(ENG.NP626),int(ENG.TK626),int(ENG.BUDGET626)],reranker=list(ENG.LOCKED627),reranker_family=[c[0]for c in ENG.CFG626],beta=float(ENG.BETA626),
                    max_new=int(ENG.MAX_NEW),pool=str(ENG.POOL),engine_sha256=hashlib.sha256(ENGINE_SRC.encode("utf-8")).hexdigest(),engine_blob=git_blob_sha1(ENGINE_SRC.encode("utf-8")),
                    chain_sha256=dict(self.info["chain_sha256"]),lock_610=ENG.LOCK,source_blobs=dict(ENG.BLOBS),source_blobs_measured=dict(self.info["source_blobs"]),
                    baseline_610=dict(H1=int(ENG.H1),H5=int(ENG.H5)),n_questions=len(ENG.PANEL624))
    def _read_startup(self):
        R=ENG.RESULT627;st={k:v for k,v in R["storage"].items()if k not in("sample","common_shapes","load_errors")}
        bt=[float(x)for x in ENG.build_times];sizes=[int(r["bytes"])for r in ENG.MANIFEST]
        native=int(ENG.TOK.numel()*ENG.TOK.element_size());basis=int(ENG.BASIS.numel()*ENG.BASIS.element_size());center=int(ENG.CENTER.numel()*ENG.CENTER.element_size())
        ls=[int(x)for x in ENG.LISTSIZE]
        return dict(measured="start-up of this runtime",measured_utc=globals().get("ENGINE_READY_UTC","reused"),gpu=torch.cuda.get_device_name(0),status=R["status"],summary=jsafe(R["summary"]),integrity=jsafe(R["integrity"]),storage=jsafe(st),locked_reranker=R["locked_reranker"],locked_candidates=jsafe(R["locked_candidates"]),
                    limitations=list(R["limitations"]),elapsed_seconds=R.get("elapsed_seconds"),result_file=self.result_path.name,result_sha256=self.result_sha,
                    write=dict(files=len(bt),ms_p50=float(np.median(bt)),ms_p95=float(np.percentile(bt,95)),seconds_total=float(sum(bt))/1000.0),
                    sizes=sizes,index=dict(native_positions=int(ENG.NTOK),ivf_lists=int(ENG.NLIST),pca_dims=int(ENG.RANK),index_bytes=int(ENG.INDEX_BYTES),index_build_ms=float(ENG.INDEX_BUILD_MS),
                    pca_basis_bytes=basis,pca_center_bytes=center,owner_bytes=int(ENG.OWNER.nbytes),native_activation_bytes=native,list_min=min(ls),list_max=max(ls),pca_train_vectors=int(ENG.SAMPLE)),
                    baseline_610=dict(H1=int(ENG.H1),H5=int(ENG.H5)))
    def startup(self):return json.loads(json.dumps(self._startup))
    def engine_result_path(self):return self.result_path
    def cases(self):
        rows=[r for r in ENG.RESULT627["per_query"]if r["panel"]=="ALTERNATE"]
        return[dict(case=int(r["case"]),question=str(r["question"]),target=int(r["target"]))for r in sorted(rows,key=lambda r:r["case"])]
    def sealed_row(self,case):
        rq=next(r for r in ENG.RESULT627["per_query"]if r["panel"]=="ALTERNATE"and int(r["case"])==int(case))
        pa=next(r for r in ENG.RESULT627["physical_answers"]if r["panel"]=="ALTERNATE"and int(r["case"])==int(case))
        return dict(selected=rq["selected"],top5=list(rq["top5"]),selected_correct=bool(rq["selected_correct"]),answer=pa.get("answer"),correct=bool(pa.get("correct")))
    def car(self,ci):
        c=ENG.CAR[ci];f=c["fact"];m=re.match(r"The current capital of (\S+) is",f)
        return dict(id=int(ci),fact=f,gold=c["gold"],body_len=len(c["body"]),country=(m.group(1)if m else None),file=ENG.MANIFEST[ci]["file"],bytes=int(ENG.MANIFEST[ci]["bytes"]),sha256=ENG.MANIFEST[ci]["sha256"])
    def source_ids(self,ci):return list(ENG.PIDS)+list(ENG.CAR[ci]["body"])
    def query_ids(self,q):return list(ENG.tok(ENG.suffix(q),add_special_tokens=False).input_ids)
    def frame_ids(self,q):return list(ENG.tok(ENG.PREFIX+q+"\n\n",add_special_tokens=False).input_ids)
    def decode_each(self,ids):return[ENG.tok.decode([t])for t in ids]
    def panel_strings(self):
        out=set()
        for c in ENG.CAR:
            out.add(str(c["gold"]));m=re.match(r"The current capital of (\S+) is",c["fact"])
            if m:out.add(m.group(1))
        return sorted(out)
    def hit(self,ans,gold):return bool(ENG.hit(ans,gold))
    def pool_path(self,ci):return Path(ENG.POOL)/ENG.MANIFEST[ci]["file"]
    def pool_stat(self):
        files=sorted(Path(ENG.POOL).glob("cut3_*.pt"));bad=[]
        for i,r in enumerate(ENG.MANIFEST):
            p=Path(ENG.POOL)/r["file"]
            if not p.is_file()or p.stat().st_size!=int(r["bytes"]):bad.append(i)
        return dict(files_in_pool=len(files),manifest=len(ENG.MANIFEST),size_mismatch=bad,total_bytes=int(sum(p.stat().st_size for p in files)))
    def pool_sha_audit(self):
        bad=[];t=time.perf_counter()
        for i,r in enumerate(ENG.MANIFEST):
            p=Path(ENG.POOL)/r["file"]
            if not p.is_file()or file_sha(p)!=r["sha256"]:bad.append(i)
        return dict(files=len(ENG.MANIFEST),mismatch=bad,seconds=time.perf_counter()-t)
    # ---- integrity ----
    def sentinel(self):return ENG.sentinel()
    def guard(self):
        h=hashlib.sha256()
        with torch.inference_mode():
            for name,p in ENG.model.named_parameters():
                x=p.detach().float();vals=(float(x.sum().item()),float(x.abs().sum().item()),float((x*x).sum().item()))
                h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode());h.update(np.asarray(vals,dtype=np.float64).tobytes());del x
        torch.cuda.empty_cache();return h.hexdigest()
    def full_weight_sha(self):return ENG.weight_sha()
    def frozen(self):
        m=ENG.model;tot,fo=hook_stats(m)
        return dict(training=bool(m.training),trainable_tensors=sum(int(p.requires_grad)for p in m.parameters()),
                    lora=bool(hasattr(m,"peft_config")or any("lora"in n.lower()for n,_ in m.named_modules())),optimizer=optimizer_present(),
                    hooks=tot,foreign_hooks=sum(fo.values()),foreign_modules=fo,residual_hooks=residual_hooks(m),sentinel=self.sentinel())
    # ---- measured engine calls ----
    def _recorder(self):
        rec=[]
        def _rec(mod,a,kw):
            ids=kw.get("input_ids",a[0]if a else None);emb=kw.get("inputs_embeds");pkv=kw.get("past_key_values");pl=None
            if pkv is not None:
                try:pl=int(pkv.get_seq_length())
                except Exception:pl=-1
            if ids is not None:
                n=int(ids.shape[-1]);rec.append(dict(kind="ids",n=n,ids=[int(t)for t in ids.reshape(-1).tolist()]if n<=8192 else None,embeds_zero=None,has_past=pkv is not None,past_len=pl))
            else:
                n=int(emb.shape[1])if emb is not None else -1
                rec.append(dict(kind="embeds",n=n,ids=None,embeds_zero=(bool(float(emb.detach().abs().max())==0.0)if emb is not None else None),has_past=pkv is not None,past_len=pl))
        return rec,_rec
    def _recorded(self,fn,*a):
        rec,_rec=self._recorder()
        with self.lock,torch.no_grad():
            h=ENG.model.register_forward_pre_hook(_rec,with_kwargs=True)
            try:torch.cuda.synchronize();t=time.perf_counter();res=fn(*a);torch.cuda.synchronize();dt=time.perf_counter()-t
            finally:h.remove()
        return res,rec,dt
    def write(self,ci):
        """WRITE: the engine's init(ci, 3) — the same call that produced the start-up file — with the forward recorder attached."""
        mem,rec,dt=self._recorded(ENG.init,int(ci),CUT_EXPECTED)
        return[(a.detach().cpu().contiguous(),b.detach().cpu().contiguous())for a,b in mem],rec,dt
    def read_file(self,ci,device=False):
        """SSD: SHA-256 against the manifest (engine sha627), then torch.load (weights_only), exactly as the engine's READ."""
        p=self.pool_path(ci);t=time.perf_counter();digest=ENG.sha627(p);ok=digest==ENG.MANIFEST[ci]["sha256"]
        if not ok:return None,dict(file=p.name,sha256=digest,sha_ok=False,seconds=time.perf_counter()-t)
        stored=torch.load(p,map_location="cpu",weights_only=True)
        mem=tuple((a.to(ENG.DEVICE),b.to(ENG.DEVICE))for a,b in stored)if device else[(a,b)for a,b in stored]
        if device:torch.cuda.synchronize()
        return mem,dict(file=p.name,sha256=digest,sha_ok=True,bytes=p.stat().st_size,seconds=time.perf_counter()-t)
    def file_bytes(self,ci):return self.pool_path(ci).read_bytes()
    def build_ms(self,ci):return float(ENG.build_times[ci])
    def address(self,question):
        """READ address: question frame → layer-3 vectors (as the engine) → candidate626 → features626 → rerank626 with the locked PEAK50 setting."""
        def enc(q):
            rep,_=ENG.question_rep(q);return ENG.apply(rep[3],ENG.FIT).detach().float().cpu().contiguous()
        q,rec,enc_s=self._recorded(enc,str(question))
        with self.lock,torch.no_grad():
            t=time.perf_counter();ids,route=ENG.candidate626(q);t1=time.perf_counter();fs=ENG.features626(q,ids);ranking,scores=ENG.rerank626(ids,fs,ENG.LOCKED627);t2=time.perf_counter()
        n,p,c=fs
        cand=[dict(id=int(ci),score=float(scores[ci]),fram=float(n[ci]),peak=float(p[ci]))for ci in ranking]
        return dict(ranking=[int(x)for x in ranking],candidate626=[int(x)for x in ids],candidates=cand,touched=int(route["touched"]),faiss_ms=float(route["faiss_ms"]),top2_ms=float(route["top2_ms"]),
                    rerank_ms=(t2-t1)*1000,address_ms=(t2-t)*1000,encode_ms=enc_s*1000,query_rows=int(q.shape[0]),forwards=rec)
    def ask(self,mem,q):
        res,rec,dt=self._recorded(ENG.answer,mem,str(q));return str(res),rec,dt
    def compare(self,a,b):
        """Bitwise comparison of two cartridges: every K and V tensor of every layer, compared as raw bytes (uint8 view), with diagnostics."""
        rows=[]
        for li,((ka,va),(kb,vb))in enumerate(zip(a,b)):
            for nm,x,y in(("K",ka,kb),("V",va,vb)):
                x=x.detach().cpu().contiguous();y=y.detach().cpu().contiguous();ss=list(x.shape)==list(y.shape);sd=x.dtype==y.dtype
                if ss and sd:
                    bit=bool(torch.equal(x.view(torch.uint8),y.view(torch.uint8)))
                    d=(x.float()-y.float()).abs();mx=float(d.max())if d.numel()else 0.0;nd=int((x.view(torch.uint8).reshape(x.numel(),-1)!=y.view(torch.uint8).reshape(y.numel(),-1)).any(1).sum())
                else:bit=False;mx=None;nd=None
                rows.append(dict(layer=li,tensor=nm,shape=list(x.shape),shape_equal=ss,dtype_equal=sd,bitwise_equal=bit,max_abs_diff=mx,differing_elements=nd,elements=int(x.numel())))
        return dict(layers_a=len(a),layers_b=len(b),tensors_compared=len(rows),all_equal=len(a)==len(b)and bool(rows)and all(r["bitwise_equal"]for r in rows),
                    layers_differing=sorted({r["layer"]for r in rows if not r["bitwise_equal"]}),max_abs_diff=max([r["max_abs_diff"]or 0.0 for r in rows],default=0.0),rows=rows)
    def equal(self,a,b):return self.compare(a,b)["all_equal"]
    def memory_len(self,mem):return int(mem[0][0].shape[-2])
    def memory_sha(self,mem):return memory_sha(mem)
    def memory_bytes(self,mem):return int(sum(k.numel()*k.element_size()+v.numel()*v.element_size()for k,v in mem))
    def memory_shape(self,mem):return dict(layers=len(mem),tensors=2*len(mem),shape=list(mem[0][0].shape),dtype=str(mem[0][0].dtype).replace("torch.",""),T=int(mem[0][0].shape[-2]),
                                           uniform=all(list(k.shape)==list(mem[0][0].shape)==list(v.shape)and k.dtype==v.dtype==mem[0][0].dtype for k,v in mem))
    def release(self,*mems):
        for m in mems:del m
        gc.collect();torch.cuda.empty_cache()
    def sync(self):torch.cuda.synchronize()
    def reset_peak(self):torch.cuda.reset_peak_memory_stats()
    def peak_gib(self):return torch.cuda.max_memory_allocated()/2**30
say("[5/7] Measurement adapter · TEST566 weight sentinel · all-parameter guard · start-up gates …")
INSTR=Instrument(ENGINE_INIT_SECONDS)
_cfg=INSTR.config();_st=INSTR.startup();_fz=INSTR.frozen();_ps=INSTR.pool_stat()
say(f"[6/7] Full-weight SHA-256 (engine, start-up) {INSTR.info['weight_sha0']} · reference {WEIGHT_SHA_EXPECTED}")
ENGINE_LOCK=[("TEST627 demo engine SHA-256 and git blob = embedded reference",_cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED and _cfg["engine_blob"]==ENGINE_BLOB_EXPECTED),
 ("626.py and 624.py executed by the engine = SHA-256 references",all(_cfg["chain_sha256"][k]==CHAIN[k]["sha256"]for k in CHAIN)),
 ("610.py executed by 624.py = SHA-256 lock",_cfg["chain_sha256"]["610.py"]==_cfg["lock_610"]==SHA610_EXPECTED),
 ("566.py and 575.py git blobs verified by 610.py = expected",_cfg["source_blobs"]==SOURCE_BLOBS_EXPECTED==_cfg["source_blobs_measured"]),
 (f"model = {MODEL_ID} · {arch_txt()} · {PREFIX_EXPECTED}-token instruction prefix",_cfg["MODEL_ID"]==MODEL_ID and tuple(_cfg["arch"])==ARCH and _cfg["prefix_tokens"]==PREFIX_EXPECTED),
 ("full-weight SHA-256 at start-up = reference",INSTR.info["weight_sha0"]==WEIGHT_SHA_EXPECTED),
 ("weight sentinel now = value at the engine's start",INSTR.info["sentinel0"]==INSTR.info["sentinel_engine"]),
 ("TEST610 baseline inside the engine: FRAM holdout R@1 12/12 · R@5 12/12",(_cfg["baseline_610"]["H1"],_cfg["baseline_610"]["H5"])==(12,12)),
 (f"{N_CART} cartridges · {N_CART} manifest entries · {N_CART} files in the pool with the manifest sizes",_cfg["n_cartridges"]==_cfg["manifest"]==_ps["files_in_pool"]==N_CART and not _ps["size_mismatch"]),
 (f"label-free index: PCA{RANK_EXPECTED} · FAISS IVF{NLIST_EXPECTED} · every native position indexed",_cfg["rank"]==RANK_EXPECTED and _cfg["nlist"]==NLIST_EXPECTED and _cfg["index_ntotal"]==_cfg["ntok"]),
 ("locked READ setting: nprobe 2 · token_k 16 · budget 16 · reranker PEAK50",tuple(_cfg["candidates"])==LOCKED_CAND and tuple(_cfg["reranker"])==LOCKED_RERANK),
 ("TEST627 engine at start-up: status COMPLETE · integrity 8/8 · result file written",_st["status"]=="COMPLETE" and all(_st["integrity"].values())and len(_st["integrity"])==8 and INSTR.result_sha is not None),
 (f"{N_CASES} original + {N_CASES} new-wording questions recorded by the engine",all(_st["summary"][p]["cases"]==N_CASES for p in("ORIGINAL","ALTERNATE"))and len(INSTR.cases())==N_CASES),
 ("trainable tensors = 0 · eval mode · no LoRA · no optimizer",_fz["trainable_tensors"]==0 and not _fz["training"]and not _fz["lora"]and not _fz["optimizer"]),
 ("no foreign hooks · no hooks on decoder layers, attention or q_proj",_fz["foreign_hooks"]==0 and not _fz["residual_hooks"])]
_bad=[n for n,ok in ENGINE_LOCK if not ok]
if _bad:raise RuntimeError(f"ENGINE LOCK FAILED: {_bad}")
_S=_st["summary"]
say(f"[7/7] Engine lock PASS · {len(ENGINE_LOCK)} checks · init {ENGINE_INIT_SECONDS:.1f}s · start-up acceptance: original {_S['ORIGINAL']['answer_correct']}/48 · new wording {_S['ALTERNATE']['answer_correct']}/48 answered from disk")
say("PART 1 / 3 complete. Now run PART 2 / 3 in the next cell.");say("="*140)
MAM_SH_PART1_OK=True
#<<ENGINE_END>>
# =====================================================================================================================================
# AKBASCORE MAM — THE SILICON HIPPOCAMPUS · QWEN2.5-7B · PART 2 / 3 · RUN PIPELINE AND FIGURES
# Same demo as PART 1. Run this cell after PART 1 / 3, in the same Colab runtime; then run PART 3 / 3.
# Defines the fail-closed RUN (integrity → WRITE re-check → SSD → READ address → READ load + answer → control → verification →
# sealed payload, logs, ZIP), four figures (JPEG 300 dpi + vector PDF; Fig. 1 also in Turkish). Nothing here changes the engine.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 12 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_SH_PART1_OK"):raise RuntimeError("PART 1 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<CORE_BEGIN>>
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch,Circle
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path("/content/AKBASCORE_MAM_SILICON_HIPPOCAMPUS")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_SILICON_HIPPOCAMPUS")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
N_STAGES=10
def ev(stage,title,body):return dict(stage=stage,title=title,body=body)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
RUN_PREFIX="MAMSH-"
def prune_runs(keep=4,prefix=RUN_PREFIX):
    runs=sorted([p for p in ROOT.glob(prefix+"*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def mb(b):return b/1e6
def fw_txt(f):
    if f["kind"]=="ids":return f"token ids n={f['n']}"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
    return f"embeddings n={f['n']} ({'all zero'if f['embeds_zero']else'NON-ZERO'})"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
def check_query(chk,I,fw,q,T,label):
    qids=I.query_ids(q)
    chk(f"{label}: model input = question-template tokens only, over the loaded {T}-row cartridge (no fact text, no source text)",len(fw)>=1 and fw[0]["kind"]=="ids"and fw[0]["ids"]==qids and fw[0]["has_past"]and fw[0]["past_len"]==T)
    chk(f"{label}: generation forwards = one new token each, over the growing cache",all(f["kind"]=="ids"and f["n"]==1 and f["has_past"]and f["past_len"]==T+len(qids)+n_ for n_,f in enumerate(fw[1:])))
    return qids
def post_seal_audit(imgs,pdfs,extra,zp,member_names,pp,sha,tp,mp,verdict):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise AuditFail("POST-SEAL AUDIT FAILED: "+name)
    ok(f"{N_FIGS}/{N_FIGS} figures created (JPEG)",len(imgs)==N_FIGS and all(file_ready(p)for p,_ in imgs));ok(f"{N_FIGS}/{N_FIGS} vector PDF copies",len(pdfs)==N_FIGS and all(file_ready(p)for p in pdfs))
    ok("Turkish Fig. 1 (JPEG + PDF) created",all(file_ready(p)for p in extra))
    bad=[]
    for p in[p for p,_ in imgs]+[x for x in extra if str(x).endswith(".jpg")]:
        with Image.open(p)as im:
            if not(im.format=="JPEG"and im.mode=="RGB"and im.size[0]>=3600):bad.append(Path(p).name)
    ok("all figures are JPEG/RGB at ≥3600 px width",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);nm=z.namelist()
        ok("ZIP is flat (no sub-folders)",all("/"not in n for n in nm));ok("ZIP contents = expected package",sorted(nm)==sorted(member_names))
        jp=sorted(n for n in nm if n.lower().endswith(".jpg")and n.startswith("fig_0")and"_TR_"not in n);ok("figure numbering 01..%02d"%N_FIGS,[n[4:6]for n in jp]==[f"{i:02d}"for i in range(1,N_FIGS+1)])
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha);ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    m=json.loads(mp.read_text(encoding="utf-8"));ok("manifest hash and verdict match payload",m["payload_sha256"]==sha and m["verdict"]==verdict)
    ok("readable log exists and is non-empty",file_ready(tp));ok("ZIP exists and is non-empty",file_ready(zp))
    return checks
def make_txt(P,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];L=P["live"];cfg=P["engine"]["config"];fz=P["frozen"];su=P["startup"];Wr=L["write"];Sd=L["ssd"];Rd=L["read"];Ct=L["control"]
    a(S);a(f"{DEMO_TITLE} — READABLE RUN LOG");a(DEMO_SUB);a(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");a(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");a(S)
    a(PRIORITY);a("Scientific question: "+SCI_QUESTION);a(NAME_NOTE)
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.")
    a("This is a LIVE run of the locked TEST627 demo engine of this runtime. Start-up values were produced by the same engine in this runtime; sealed 627.log / 626.log values are reproduced verbatim, never recomputed.")
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("VERDICT",P["verdict"]),
        ("MODEL",f"{cfg['MODEL_ID']} · frozen · {P['model']['dtype']} · attention {P['model']['attn']}"),("ARCHITECTURE",arch_txt(tuple(P["model"]["arch"]))),
        ("GPU",f"{E_['gpu']} (TEST627 reference GPU: {CANON_GPU}; same: {P['hardware']['same']})"),("TORCH / TRANSFORMERS / GRADIO",f"{E_['torch']} / {E_['transformers']} / {E_['gradio']}"),
        ("ENGINE",f"{P['engine']['file']} · SHA-256 {P['engine']['sha256']} · git blob {P['engine']['blob']}"),
        ("ENGINE CHAIN",f"626.py {cfg['chain_sha256']['626.py']} · 624.py {cfg['chain_sha256']['624.py']} · 610.py {cfg['chain_sha256']['610.py']}"),
        ("SOURCE FILES",f"566.py {cfg['source_blobs']['566.py']} · 575.py {cfg['source_blobs']['575.py']} (git blobs verified by 610.py)"),
        ("CARTRIDGE BANK",f"{cfg['n_cartridges']} physical CUT3 files ({cfg['n_original']} original + {cfg['n_cartridges']-cfg['n_original']} synthetic facts) · pool {cfg['pool']}"),
        ("ADDRESS INDEX",f"PCA{cfg['rank']} + FAISS IVF{cfg['nlist']} over {cfg['ntok']} native layer-3 positions (label-free)"),
        ("LOCKED READ",f"nprobe {cfg['candidates'][0]} · token_k {cfg['candidates'][1]} · budget {cfg['candidates'][2]} · reranker {cfg['reranker'][0]}"),
        ("ENGINE RESULT",f"{su['result_file']} · SHA-256 {su['result_sha256']}"),("FULL-WEIGHT SHA-256",f"{fz['weight_sha_startup']} (reference {WEIGHT_SHA_EXPECTED})")):a(f"{k_:<30}: {v}")
    a("");a("CASE (predeclared rotation over the 48 new-wording questions of the engine)");a(Dd)
    a(f"case {L['case']:02d} · question: {L['question']}");a(f"fact of the case (used only by WRITE): {L['fact']}");a(f"expected answer (audit metadata; not an input to WRITE, addressing, selection or answering): {L['expected']}")
    a("");a("A · WRITE  [LIVE re-execution of the engine's init(fact, 3) for this fact]");a(Dd)
    for n_,f in enumerate(Wr["forwards"],1):a(f"forward {n_}: {fw_txt(f)}")
    a(f"rows {Wr['T']} = {Wr['prefix_tokens']} instruction tokens + {Wr['fact_tokens']} fact tokens · {Wr['layers']} layers × (K, V) = {Wr['tensors']} tensors {Wr['shape']} {Wr['dtype']} · {Wr['memory_bytes']} bytes · {1000*Wr['seconds']:.1f} ms")
    a(f"start-up file {Wr['file']} · {Wr['file_bytes']} bytes · SHA-256 {Wr['file_sha256']} (manifest match: {Wr['sha_ok']}) · written at start-up in {Wr['startup_write_ms']:.1f} ms")
    a(f"live re-write bitwise identical to the start-up file in all {Wr['layers']} layers: {Wr['bit_equal']} · fact text in file: {Wr['fact_in_file']} · answer word in file: {Wr['gold_in_file']} · question in file: {Wr['question_in_file']}")
    a("");a("B · SSD  [LIVE]");a(Dd)
    a(f"files in pool {Sd['files_in_pool']} · manifest {Sd['manifest']} · size mismatches {len(Sd['size_mismatch'])} · total {Sd['total_bytes']} bytes ({Sd['total_bytes']/1e9:.3f} GB)")
    a(f"file sizes: min {Sd['min']} · median {Sd['median']:.0f} · max {Sd['max']} bytes · SHA-256 of all {Sd['manifest']} files verified by the engine at start-up: {Sd['startup_sha_verified']}")
    a("");a("C · READ — address  [LIVE · engine functions candidate626 / features626 / rerank626, locked PEAK50]");a(Dd)
    for f in Rd["frame_forwards"]:a(f"question frame forward: {fw_txt(f)} (instruction prefix + question only)")
    a(f"{Rd['query_rows']} question vectors → PCA{cfg['rank']} → IVF{cfg['nlist']} (nprobe {cfg['candidates'][0]}, token_k {cfg['candidates'][1]}) → touched {Rd['touched']} of {cfg['n_cartridges']} cartridges → {len(Rd['candidates'])} candidates → PEAK50")
    for n_,c in enumerate(Rd["candidates"],1):a(f"   {n_:>2}. #{c['id']:04d} · PEAK50 {c['score']:.6f} · FRAM {c['fram']:.6f} · peak {c['peak']:.6f}"+("   ← SELECTED"if n_==1 else"")+("   (expected cartridge, audit)"if c["id"]==L["target"]else""))
    a(f"timing: question encode {Rd['encode_ms']:.1f} ms · FAISS {Rd['faiss_ms']:.2f} ms · TOP2 {Rd['top2_ms']:.2f} ms · PEAK50 {Rd['rerank_ms']:.2f} ms · address total {Rd['address_ms']:.1f} ms")
    a(f"selected #{Rd['selected']:04d} · expected cartridge among the candidates: {Rd['target_in_candidates']} · selected = expected: {Rd['selected_correct']}")
    a("");a("D · READ — load + answer  [LIVE · sha627 + torch.load + answer(memory, question)]");a(Dd)
    a(f"file {Rd['file']} · SHA-256 = manifest {Rd['sha_ok']} · load {Rd['load_ms']:.1f} ms · {Rd['T']} rows")
    for f in Rd["answer_forwards"][:1]:a(f"answer forward 1: {fw_txt(f)} — question template only")
    a(f"answer  : {L['answer']}");a(f"expected: {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'} (re-derived: {L['correct_rederived']}) · {Rd['answer_ms']:.0f} ms · generation forwards {Rd['gen_forwards']}")
    a(f"start-up engine row for this question: selected #{L['startup_row']['selected']} · answer {L['startup_row']['answer']!r} · live identical: selection {L['matches_startup_selection']} · answer {L['matches_startup_answer']} (reported, not a gate)")
    a("");a("E · CONTROL — runner-up file  [LIVE · diagnostic]");a(Dd)
    if Ct.get("skipped"):a("skipped: "+Ct["skipped"])
    else:a(f"runner-up #{Ct['id']:04d} ({Ct['file']}) · answer {Ct['answer']!r} · expected answer produced: {Ct['gold_produced']} · runner-up is the expected cartridge: {Ct['is_target']}")
    a("");a("START-UP IN THIS RUNTIME (TEST627 demo engine, before any RUN)");a(Dd)
    for p_ in("ORIGINAL","ALTERNATE"):
        s_=su["summary"][p_];a(f"{p_:<9}: selected {s_['selected_correct']}/{s_['cases']} · answered {s_['answer_correct']}/{s_['cases']} · errors {s_['errors']} · address p50 {s_['address_p50_ms']} ms · load p50 {s_['load_p50_ms']} ms · answer p50 {s_['answer_p50_ms']} ms")
    a("integrity: "+" · ".join(f"{k} {v}"for k,v in su["integrity"].items()))
    st=su["storage"];a(f"storage: {st['cartridges']} files · {st['disk_pool_bytes']} bytes · avg {st['avg_file_bytes']} · tensor payload {st['total_tensor_payload_bytes']} · lossless gzip sample ({st['lossless_gzip_sample_files']} files) ratio {st['gzip_sample_ratio']} (estimate for the full pool, not measured)")
    ix=su["index"];a(f"index: {ix['native_positions']} positions · IVF lists {ix['ivf_lists']} ({ix['list_min']}–{ix['list_max']} per list) · index {ix['index_bytes']} bytes · PCA basis {ix['pca_basis_bytes']} bytes · native layer-3 vectors {ix['native_activation_bytes']} bytes · build {ix['index_build_ms']:.1f} ms")
    a(f"WRITE at start-up: {su['write']['files']} files · p50 {su['write']['ms_p50']:.1f} ms per file · total {su['write']['seconds_total']:.1f} s")
    a("engine limitations (verbatim): "+" | ".join(su["limitations"]))
    a("");a("SEALED REFERENCES (archived; not produced by this run)");a(Dd)
    T7=TEST627_LOG;a(f"TEST627 · {T7['decision']} · 627.log SHA-256 {T7['log_sha256']}")
    for p_,s_ in T7["e2e"].items():a(f"   {p_:<9}: selected {s_['selected_correct']}/48 · answered {s_['answer_correct']}/48 · errors {s_['errors']} · address p50 {s_['address_p50_ms']} ms · load p50 {s_['load_p50_ms']} ms · answer p50 {s_['answer_p50_ms']} ms")
    T6=TEST626_LOG;a(f"TEST626 · own pre-declared rule locked {T6['locked_by_own_rule']} ({T6['e2e']} answered) · FRAM-only new wording {T6['fram_only_new_wording']} · PEAK50 new wording {T6['peak50_new_wording']} · 626.log SHA-256 {T6['log_sha256']}")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("weight_sha_startup","sentinel_startup","sentinel_before","sentinel_after","guard_startup","guard_before","guard_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} · lora={fz['lora']} · optimizer={fz['optimizer']} · training_mode={fz['training_mode']} · foreign hooks outside engine calls {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']} · hooks on layers/attention/q_proj {len(fz['residual_hooks_before'])}→{len(fz['residual_hooks_after'])}")
    a(fz["full_weight_method"]);a(fz["guard_method"]);a(fz["sentinel_method"]);a(P["instrumentation"]["forward_counter"])
    for t_,k_ in(("DEMONSTRATED IN THIS RUN","demonstrated"),("PRODUCED AT START-UP IN THIS RUNTIME","startup"),("ARCHIVED","archived"),("NOT ESTABLISHED","not_established"),("TECHNICAL NOTES","notes")):
        a("");a(t_);a(Dd)
        for t in P["scope"][k_]:a("• "+t)
    a("");a("PRE-SEAL CHECKS");a(Dd)
    for c_ in P["checks_pre_seal"]:a("PASS · "+c_)
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<24}: {v:.3f}")
    a(f"peak GPU memory {P['gpu']['peak_allocated_gib']:.2f} GiB");a("");a(f"Raw forward records are in {names['payload']} and {names['jsonl']}.");a(S)
    return"\n".join(o)
def execute_run(I,ctl):
    """Fail-closed wrapper: on a technical audit failure the raw outputs gathered so far are preserved (never sealed, no figures)."""
    state={}
    try:return(yield from _execute_run(I,ctl,state))
    except AuditFail as ex:
        raw=state.get("raw")
        if raw and any(v is not None for v in raw.values()):
            try:
                p=Path(ctl["run_dir"])/f"FAILED_AUDIT_RAW_{ctl['run_id']}.json"
                p.write_bytes(canon(dict(status="FAILED AUDIT — NOT SEALED",run_id=ctl["run_id"],failed_check=str(ex),checks_passed=state.get("checks",[]),raw=raw,note="Raw outputs preserved. No figures, no seal and no verdict were produced.")));ex.partial=str(p)
            except Exception:pass
        raise
    finally:
        try:I.release()
        except Exception:pass
def _execute_run(I,ctl,state):
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);case=int(ctl["case"])
    info=I.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks
    raw=dict(write=None,ssd=None,address=None,answer=None,control=None);state["raw"]=raw
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();say("="*140);say(f"RUN {run_id} · case {case:02d}");say("="*140)
    # ---- 1 integrity ----
    yield ev(1,"Integrity","Engine and chain hashes, source blobs, locked READ setting, start-up acceptance status, weight sentinel, all-parameter guard, hooks.")
    I.reset_peak();cfg=I.config();su=I.startup();fz0=I.frozen();t=time.perf_counter();g_before=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("TEST627 demo engine SHA-256 and git blob = embedded reference",cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED and cfg["engine_blob"]==ENGINE_BLOB_EXPECTED)
    chk("626.py · 624.py · 610.py executed by the engine = SHA-256 references",all(cfg["chain_sha256"][k]==CHAIN[k]["sha256"]for k in CHAIN)and cfg["chain_sha256"]["610.py"]==cfg["lock_610"]==SHA610_EXPECTED)
    chk("566.py and 575.py git blobs verified by 610.py = expected",cfg["source_blobs"]==SOURCE_BLOBS_EXPECTED==cfg["source_blobs_measured"])
    chk(f"model, architecture, CUT{CUT_EXPECTED} and {PREFIX_EXPECTED}-token instruction prefix",cfg["MODEL_ID"]==MODEL_ID and tuple(cfg["arch"])==ARCH and cfg["CUT"]==CUT_EXPECTED and cfg["prefix_tokens"]==PREFIX_EXPECTED)
    chk(f"{N_CART} cartridges · PCA{RANK_EXPECTED} · IVF{NLIST_EXPECTED} · every native position indexed",cfg["n_cartridges"]==cfg["manifest"]==N_CART and cfg["rank"]==RANK_EXPECTED and cfg["nlist"]==NLIST_EXPECTED and cfg["index_ntotal"]==cfg["ntok"])
    chk("locked READ setting nprobe 2 · token_k 16 · budget 16 · PEAK50",tuple(cfg["candidates"])==LOCKED_CAND and tuple(cfg["reranker"])==LOCKED_RERANK)
    chk("TEST627 engine at start-up in this runtime: status COMPLETE · integrity 8/8",su["status"]=="COMPLETE"and len(su["integrity"])==8 and all(su["integrity"].values()))
    chk("full-weight SHA-256 at start-up = reference",info["weight_sha0"]==WEIGHT_SHA_EXPECTED)
    chk("weight sentinel before run = value at initialisation",fz0["sentinel"]==info["sentinel0"]);chk("all-parameter guard before run = value at start-up",g_before==info["guard0"])
    chk("trainable tensors = 0, eval mode, no LoRA, no optimizer",fz0["trainable_tensors"]==0 and not fz0["training"]and not fz0["lora"]and not fz0["optimizer"])
    chk("no foreign hooks outside engine calls · no hooks on decoder layers, attention or q_proj",fz0["foreign_hooks"]==0 and not fz0["residual_hooks"])
    cases=I.cases();chk(f"case in 0..{N_CASES-1} of the engine's {N_CASES} new-wording questions",len(cases)==N_CASES and 0<=case<N_CASES and cases[case]["case"]==case)
    tgt=cases[case]["target"];question=cases[case]["question"];C=I.car(tgt);expected=C["gold"];fact=C["fact"]
    strip=I.panel_strings();state["strip"]=strip
    for c in checks:say("   PASS ·",c)
    # ---- 2 WRITE ----
    yield ev(2,"A · WRITE","The engine's init(fact, 3) is executed again for the fact of this case: one token-id forward of the instruction prefix + the fact, one all-zero-embedding forward; the result is compared bitwise with the file written at start-up.")
    src=I.source_ids(tgt);memW,fwW,dtW=I.write(tgt);tm["write"]+=dtW;raw["write"]=dict(cartridge=tgt,forwards=len(fwW))
    chk("WRITE: forward 1 = token ids of the instruction prefix + this fact only (no cache)",len(fwW)==2 and fwW[0]["kind"]=="ids"and fwW[0]["ids"]==src and not fwW[0]["has_past"])
    chk("WRITE: forward 2 = all-zero input embeddings over the same rows (no cache); no other forward",fwW[1]["kind"]=="embeds"and fwW[1]["embeds_zero"]and fwW[1]["n"]==len(src)and not fwW[1]["has_past"])
    shp=I.memory_shape(memW);T_w=shp["T"]
    chk(f"cartridge = {ARCH[0]} layers × (K, V) of shape [1, {ARCH[3]}, T, {ARCH[4]}] with T = instruction prefix + fact tokens",shp["layers"]==ARCH[0]and shp["shape"]==[1,ARCH[3],T_w,ARCH[4]]and T_w==len(src)and shp["uniform"])
    wbytes=I.memory_bytes(memW);chk("cartridge bytes = layers × 2 × KV heads × head dim × T × element size",wbytes==ARCH[0]*2*ARCH[3]*ARCH[4]*T_w*(2 if shp["dtype"]in("bfloat16","float16")else 4))
    t=time.perf_counter();stored,fi=I.read_file(tgt);tm["write_compare_load"]+=time.perf_counter()-t
    chk("start-up file of this fact: SHA-256 recomputed from disk = manifest",fi["sha_ok"]and fi["sha256"]==C["sha256"])
    cmpd=I.compare(memW,stored);raw["write"]["compare"]={k_:cmpd[k_]for k_ in("tensors_compared","all_equal","layers_differing","max_abs_diff")}
    if not cmpd["all_equal"]:
        # Diagnostic BEFORE the gate fails: a second live re-write separates re-execution nondeterminism from a stored-memory mismatch.
        memW2,_,_=I.write(tgt);c2=I.compare(memW,memW2);c3=I.compare(memW2,stored)
        cls=("re-execution is not deterministic: two live re-writes of the same fact differ from each other"if not c2["all_equal"]else
             "the re-write is reproducible (two live re-writes identical) but differs from the start-up file: stored-memory mismatch")
        raw["write_diagnostic"]=dict(classification=cls,live_vs_file=cmpd,live_vs_second_live=c2,second_live_vs_file=c3)
        say("   WRITE DIAGNOSTIC:",cls,"· layers differing",cmpd["layers_differing"],"· max |Δ|",cmpd["max_abs_diff"]);I.release(memW2);del memW2
    chk(f"WRITE comparison covers all {2*ARCH[0]} tensors ({ARCH[0]} layers × K and V), compared as raw bytes",cmpd["tensors_compared"]==2*ARCH[0]and cmpd["layers_a"]==cmpd["layers_b"]==ARCH[0])
    beq=cmpd["all_equal"];chk(f"live re-write bitwise identical to the start-up file in all {ARCH[0]} layers (K and V)",beq)
    wsha=I.memory_sha(memW);chk("memory SHA-256 of the re-write = memory SHA-256 of the file",wsha==I.memory_sha(stored))
    fb=I.file_bytes(tgt);fact_in=fact.encode("utf-8")in fb;gold_in=expected.encode("utf-8")in fb;q_in=question.encode("utf-8")in fb
    chk("the cartridge file contains neither the fact text, nor the answer word, nor the question",not fact_in and not gold_in and not q_in)
    write=dict(cartridge=tgt,file=C["file"],file_bytes=len(fb),file_sha256=fi["sha256"],sha_ok=fi["sha_ok"],forwards=fwW,T=T_w,prefix_tokens=cfg["prefix_tokens"],fact_tokens=T_w-cfg["prefix_tokens"],
               layers=shp["layers"],tensors=shp["tensors"],shape=shp["shape"],dtype=shp["dtype"],memory_bytes=wbytes,container_overhead=len(fb)-wbytes,memory_sha256=wsha,bit_equal=beq,tensors_compared=cmpd["tensors_compared"],elements_compared=int(sum(r["elements"]for r in cmpd["rows"])),
               fact_in_file=fact_in,gold_in_file=gold_in,question_in_file=q_in,seconds=dtW,startup_write_ms=float(I.build_ms(tgt)))
    I.release(memW,stored);del memW,stored
    say(f"   WRITE #{tgt:04d}: {T_w} rows · bitwise identical to {C['file']} · {1000*dtW:.0f} ms")
    # ---- 3 SSD ----
    yield ev(3,"B · SSD","All 1,453 cartridge files are listed and their sizes compared with the SHA-256 manifest written at start-up.")
    t=time.perf_counter();ps=I.pool_stat();tm["ssd_stat"]+=time.perf_counter()-t;sizes=su["sizes"]
    chk(f"SSD: {N_CART} independent cartridge files present · {N_CART} manifest entries · every size = manifest",ps["files_in_pool"]==ps["manifest"]==N_CART and not ps["size_mismatch"])
    chk("SSD: total pool bytes = sum of the manifest sizes = engine storage report",ps["total_bytes"]==sum(sizes)==su["storage"]["disk_pool_bytes"])
    ssd=dict(ps,min=min(sizes),max=max(sizes),median=float(np.median(sizes)),startup_sha_verified=bool(su["integrity"].get("manifest_1453_sha_verified")),write_ms_p50=su["write"]["ms_p50"])
    raw["ssd"]=dict(files=ps["files_in_pool"],bytes=ps["total_bytes"])
    # ---- 4 READ address ----
    yield ev(4,"C · READ — geometric address","From the question alone: one question-frame forward → layer-3 vectors → PCA128 → FAISS IVF128 → 16 candidates → locked PEAK50 selects one cartridge.")
    A=I.address(question);tm["address"]+=A["address_ms"]/1000;tm["question_encode"]+=A["encode_ms"]/1000;raw["address"]=dict(ranking=A["ranking"],touched=A["touched"])
    fids=I.frame_ids(question)
    chk("READ: exactly one model forward to read the question = instruction prefix + question only (no cache, no fact text)",len(A["forwards"])==1 and A["forwards"][0]["kind"]=="ids"and A["forwards"][0]["ids"]==fids and not A["forwards"][0]["has_past"])
    rk=A["ranking"];chk(f"READ: 1–{LOCKED_CAND[2]} distinct candidate cartridges in 0..{N_CART-1}, ranked by PEAK50",1<=len(rk)<=LOCKED_CAND[2]and len(set(rk))==len(rk)and all(0<=x<N_CART for x in rk)and[c["id"]for c in A["candidates"]]==rk)
    chk("READ: PEAK50 score = 0.5 × FRAM + 0.5 × peak for every candidate (re-derived)",all(abs(c["score"]-(0.5*c["fram"]+0.5*c["peak"]))<=1e-9 for c in A["candidates"]))
    c626=A["candidate626"];sc_={c["id"]:c["score"]for c in A["candidates"]}
    chk(f"READ: candidate626 returned min({LOCKED_CAND[2]}, touched) candidates and PEAK50 ranks exactly that set (nothing merged, added or dropped)",len(c626)==min(LOCKED_CAND[2],A["touched"])and sorted(c626)==sorted(rk))
    chk("READ: order re-derived with the engine's rule — PEAK50 score descending, ties by cartridge id",rk==sorted(c626,key=lambda ci:(-sc_[ci],ci)))
    srow=I.sealed_row(case)
    if not(srow["selected"]==rk[0]and list(srow["top5"])==rk[:5]):raw["read_diagnostic"]=dict(live_top5=rk[:5],startup_top5=list(srow["top5"]),live_candidates=c626)
    chk("READ reproduces the TEST627 engine's start-up record of this question in this runtime: same selected cartridge and same top-5",srow["selected"]==rk[0]and list(srow["top5"])==rk[:5])
    sel=rk[0];sel_ok=sel==tgt;tin=tgt in rk
    say(f"   READ: touched {A['touched']} · {len(rk)} candidates · selected #{sel:04d} · expected among candidates {tin} · {A['address_ms']:.1f} ms")
    # ---- 5 READ load + answer ----
    yield ev(5,"D · READ — load from SSD and answer","The selected file is checked against its SHA-256, loaded with torch.load, and the frozen model answers over it; the source text is not given.")
    memS,li=I.read_file(sel,device=True);tm["load"]+=li["seconds"]
    chk("READ: selected file SHA-256 recomputed from disk = manifest",li["sha_ok"])
    T=I.memory_len(memS);a1,fwA,dtA=I.ask(memS,question);tm["answer"]+=dtA;raw["answer"]=a1;qids=check_query(chk,I,fwA,question,T,"answer")
    correct=I.hit(a1,expected);cr=hit_ans(a1,expected);chk("correctness re-derived from the answer text = engine hit()",cr==correct)
    I.release(memS);del memS
    say(f"   answer {a1!r} · expected {expected} · {'CORRECT'if correct else'INCORRECT'} · {len(qids)} question tokens over the {T}-row cartridge")
    # ---- 6 control ----
    yield ev(6,"E · Control — runner-up file","The same question over the file of the second-ranked candidate (diagnostic: does the expected answer appear without the selected cartridge?).")
    if len(rk)>=2:
        ru=rk[1];memR,lr=I.read_file(ru,device=True);chk("control: runner-up file SHA-256 = manifest",lr["sha_ok"]);TR_=I.memory_len(memR)
        aR,fwR,dtR=I.ask(memR,question);tm["control"]+=dtR+lr["seconds"];check_query(chk,I,fwR,question,TR_,"control")
        control=dict(id=ru,file=lr["file"],answer=aR,gold_produced=I.hit(aR,expected),is_target=ru==tgt,T=TR_);I.release(memR);del memR
    else:control=dict(skipped="only one candidate was returned",gold_produced=None)
    raw["control"]=control
    # ---- 7 verification ----
    yield ev(7,"Verification","Weight sentinel, all-parameter guard, trainable tensors and hooks after the run.")
    fz1=I.frozen();t=time.perf_counter();g_after=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("all-parameter guard after run = before run = start-up (weights unchanged)",g_after==g_before==info["guard0"]);chk("weight sentinel after run = value at initialisation",fz1["sentinel"]==info["sentinel0"])
    chk("trainable tensors = 0, no foreign hooks and no hooks on layers/attention/q_proj after run",fz1["trainable_tensors"]==0 and fz1["foreign_hooks"]==0 and not fz1["residual_hooks"])
    tm["engine_total"]=time.perf_counter()-T0
    verdict=("SELFTEST_SYNTHETIC_"if I.kind!="ENGINE"else"")+f"CASE_{case:02d}_{'CORRECT'if correct else'INCORRECT'}_WRITE_REVERIFIED_SSD_READ_SOURCE_FREE_ANSWER"
    # ---- 8 seal ----
    yield ev(8,"Sealing","Payload, manifest, readable log and raw records are written and hashed.")
    peak=I.peak_gib();same_hw=info["gpu"]==CANON_GPU
    read=dict(question=question,frame_forwards=A["forwards"],frame_tokens=len(fids),query_rows=A["query_rows"],touched=A["touched"],candidates=A["candidates"],ranking=rk,selected=sel,selected_correct=sel_ok,
              target_in_candidates=tin,faiss_ms=A["faiss_ms"],top2_ms=A["top2_ms"],rerank_ms=A["rerank_ms"],address_ms=A["address_ms"],encode_ms=A["encode_ms"],
              file=li["file"],sha_ok=li["sha_ok"],load_ms=1000*li["seconds"],candidate626=c626,
              timing_scope={"encode_ms":"question_rep + apply (one question-frame forward)","address_ms":"candidate626 (PCA projection + FAISS search + TOP2) + features626 + rerank626, as TEST627 address_ms",
                            "faiss_ms":"PCA projection + IVF quantizer + FAISS search (part of address_ms)","top2_ms":"TOP2 candidate scoring (part of address_ms)","rerank_ms":"features626 + rerank626 (part of address_ms)",
                            "load_ms":"sha627 SHA-256 of the file + torch.load + copy to GPU, as TEST627 load_ms; may be served from the OS page cache","answer_ms":"answer(memory, question): question-template forward + greedy generation",
                            "write_ms":"init(fact, 3) only (two forwards); file writing is not repeated","note":"stage timings of one run; they are not summed into an end-to-end latency"},T=T,answer_forwards=fwA,query_tokens=len(qids),query_token_text=I.decode_each(qids),gen_forwards=len(fwA)-1,answer_ms=1000*dtA)
    live=dict(case=case,question=question,fact=fact,expected=expected,target=tgt,write=write,ssd=ssd,read=read,answer=a1,correct=bool(correct),correct_rederived=bool(cr),control=control,
              startup_row=srow,matches_startup_selection=srow["selected"]==sel,matches_startup_answer=norm_ans(str(srow["answer"]))==norm_ans(a1),
              session=[dict(r,correct=(bool(correct)if r.get("correct")is None else bool(r["correct"])))for r in(ctl.get("session")or[])])
    P={"schema":"akbascore.mam.silicon_hippocampus.qwen.test627.demo.run.v1","demo":DEMO_TITLE,"subtitle":DEMO_SUB,"name_note":NAME_NOTE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"place":AUTHOR_PLACE,
       "copyright":COPYRIGHT,"license":{"name":LICENSE_NAME,"url":LICENSE_URL},"priority_record":PRIORITY,"paradigm":PARADIGM,"discovery":DISCOVERY,"question":SCI_QUESTION,"core_message":CORE_MESSAGE,
       "earlier_records":{"v1.0":PRIOR_PUB,"v1.1":QWEN_PUB,"v1.2":V12_PUB,"note":"cited by DOI only; no value of an earlier record is used as a result here"},
       "run_id":run_id,"run_start_utc":run_start_utc,"run_end_utc":utc_now(),"verdict":verdict,
       "model":{"id":cfg["MODEL_ID"],"arch":info["arch"],"dtype":info["dtype"],"attn":info["attn"],"params":info["params"]},
       "environment":{k_:info[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},"hardware":{"reference_gpu":CANON_GPU,"this_run_gpu":info["gpu"],"same":bool(same_hw)},
       "engine":{"file":info["engine_file"],"sha256":cfg["engine_sha256"],"blob":cfg["engine_blob"],"config":cfg,"init_seconds":info["init_seconds"],"instrument":I.kind,
                 "calls":"init(ci, 3) [WRITE]; sha627 + torch.load [SSD]; question_rep + apply + candidate626 + features626 + rerank626(PEAK50) [READ address]; answer(memory, q) [READ answer]",
                 "repository":{"627.py":URL_627,"627.log":URL_627LOG,"626.py":URL_626,"626.log":URL_626LOG,"624.py":URL_624,"610.py":URL_610}},
       "instrumentation":{"forward_counter":info["forward_counter"]},"startup":su,"live":live,
       "frozen":{"weight_sha_startup":info["weight_sha0"],"weight_sha_reference":WEIGHT_SHA_EXPECTED,"full_weight_method":info["full_weight_method"],"sentinel_startup":info["sentinel0"],"sentinel_before":fz0["sentinel"],"sentinel_after":fz1["sentinel"],"guard_startup":info["guard0"],"guard_before":g_before,"guard_after":g_after,
                 "trainable_tensors":fz1["trainable_tensors"],"lora":fz1["lora"],"optimizer":fz1["optimizer"],"training_mode":fz1["training"],"foreign_hooks_before":fz0["foreign_hooks"],
                 "foreign_hooks_after":fz1["foreign_hooks"],"residual_hooks_before":fz0["residual_hooks"],"residual_hooks_after":fz1["residual_hooks"],"guard_method":info["guard_method"],"sentinel_method":info["sentinel_method"]},
       "archived":{"test627_log":TEST627_LOG,"test626_log":TEST626_LOG,"note":"Sealed reference values from the repository logs; not produced or recomputed by this run."},
       "scope":SCOPE,"checks_pre_seal":list(checks),"timing":dict(tm),"gpu":{"peak_allocated_gib":peak}}
    P=jsafe(P);allow=(info["gpu"],CANON_GPU)
    hits=claim_scan("payload",strip_words(json.dumps(P,ensure_ascii=False),strip),allow=allow)
    chk("claim-discipline and foreign-model scan of the payload: 0 hits",not hits);P["claim_scan"]={"rules":len(CLAIM_RULES)+1+len(LEAK_RULES),"hits":0,"scope":"payload JSON text; panel names excluded; earlier-record citation and the demo name exempt"}
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RAW_RECORDS.jsonl",images=f"figures_manifest_{run_id}.json",
               engine=f"{run_id}_ENGINE_STARTUP_RESULT.json",zip=f"AKBASCORE_MAM_SILICON_HIPPOCAMPUS_EVIDENCE_{run_id}.zip")
    ep=run_dir/names["engine"];src_res=I.engine_result_path()
    if src_res is not None and Path(src_res).is_file():shutil.copy(src_res,ep)
    else:ep.write_text(json.dumps({"note":"engine result file unavailable"}),encoding="utf-8")
    chk("engine start-up result copied and its SHA-256 = value recorded at start-up",(hashlib.sha256(ep.read_bytes()).hexdigest()==su["result_sha256"])if su["result_sha256"]else False)
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"copyright":COPYRIGHT,"license":LICENSE_URL,"priority_record":PRIORITY,"run_id":run_id,"payload_file":payload_name,
              "payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":verdict,"engine_sha256":cfg["engine_sha256"],"engine_chain_sha256":cfg["chain_sha256"],
              "engine_startup_result_file":ep.name,"engine_startup_result_sha256":su["result_sha256"],"test627_log_sha256":TEST627_LOG["log_sha256"],"weight_sha_startup":info["weight_sha0"],"model":MODEL_ID,
              "written_cartridge_file":write["file"],"written_cartridge_sha256":write["file_sha256"],"selected_cartridge_file":read["file"],
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not a third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"run_id":run_id,"verdict":verdict,"case":case,"question":question,"answer":a1,"expected":expected,"correct":bool(correct),"target":tgt,"selected":sel,"selected_correct":sel_ok,
             "candidates":rk,"touched":A["touched"],"write_bitwise_identical":beq,"cartridge_rows":T,"cartridge_file":read["file"],"address_ms":A["address_ms"],"load_ms":read["load_ms"],"answer_ms":read["answer_ms"],
             "control":{k_:control.get(k_)for k_ in("id","answer","gold_produced","is_target","skipped")},"weights_unchanged":True,"payload_sha256":sha,"model":MODEL_ID,
             "startup_acceptance":{p_:{k_:su["summary"][p_][k_]for k_ in("cases","selected_correct","answer_correct","errors")}for p_ in("ORIGINAL","ALTERNATE")}}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"op":"write","cartridge":tgt,"forwards":fwW,"T":T_w,"bit_equal":beq,"memory_sha256":wsha,"file":write["file"],"file_sha256":write["file_sha256"]},
           {"op":"ssd","files":ps["files_in_pool"],"bytes":ps["total_bytes"],"size_mismatch":ps["size_mismatch"]},
           {"op":"address","question":question,"forwards":A["forwards"],"candidates":A["candidates"],"touched":A["touched"],"address_ms":A["address_ms"]},
           {"op":"answer","file":read["file"],"answer":a1,"forwards":fwA},{"op":"control",**{k_:v for k_,v in control.items()}}]
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 9 figures ----
    yield ev(9,"Rendering figures",f"{N_FIGS} figures at 300 dpi (JPEG) with vector PDF copies (+ Fig. 1 in Turkish); every printed value is checked against the sealed payload.")
    ctx={"P":P,"sha":sha,"run_id":run_id,"payload_name":payload_name,"allow":allow,"strip":strip,"N":N_FIGS,"audit":[],"pdfs":[],"checks":len(checks)}
    t_r=time.perf_counter();imgs,extra=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size
        q_=ctx["pdfs"][n_-1];entries.append({"index":n_,"jpg":p.name,"pdf":q_.name,"title":cap,"width_px":w_,"height_px":h_,"dpi":OUT_DPI,"jpg_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"pdf_sha256":hashlib.sha256(q_.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"figures":entries,
                               "turkish_fig1":[{"file":Path(x).name,"sha256":hashlib.sha256(Path(x).read_bytes()).hexdigest()}for x in extra],"figure_text_audit":ctx["audit"],
                               "note":"Per-file SHA-256 is an artifact integrity seal. Figure text was checked against the sealed payload and scanned for claim discipline before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    # ---- 10 package ----
    yield ev(10,"Packaging","ZIP with figures, PDFs, payload, manifest, logs, raw records and the engine start-up result; post-seal audit.")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+list(ctx["pdfs"])+[Path(x)for x in extra]+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"],ep]
    for m in members:
        if m.suffix in(".json",".txt",".jsonl")and m!=ep:chk(f"claim-discipline and foreign-model scan of {m.name}: 0 hits",not claim_scan(m.name,strip_words(m.read_text(encoding="utf-8"),strip),allow=allow,numeric=m.suffix==".txt"))
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,ctx["pdfs"],extra,zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict)
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS · figures {render_s:.1f}s · seal {seal_s:.2f}s")
    say("VERDICT :",verdict);say("PAYLOAD :",sha);say("ZIP     :",zp);say("="*140)
    tr_jpg=next((Path(x)for x in extra if str(x).endswith(".jpg")),None)
    ledger=dict(run_id=run_id,utc=sealed_utc,case=case,question=question,answer=a1,expected=expected,correct=bool(correct),selected=sel,selected_correct=sel_ok,write_bitwise_identical=beq,
                control_gold_produced=control.get("gold_produced"),weights_unchanged=True,verdict=verdict,payload_sha256=sha)
    return dict(run_id=run_id,run_dir=run_dir,P=P,sha=sha,imgs=imgs,pdfs=ctx["pdfs"],extra=extra,tr_jpg=tr_jpg,zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,
                summary=run_dir/names["summary"],jsonl=run_dir/names["jsonl"],engine=ep,checks=len(checks)+len(post),verdict=verdict,ledger=ledger)
#<<CORE_END>>
#<<FIGURES_BEGIN>>
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827","xtick.color":"#374151","ytick.color":"#374151",
                     "xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","pdf.fonttype":42,"ps.fonttype":42,
                     "legend.fontsize":8,"legend.frameon":False,"mathtext.fontset":"dejavusans"})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F9FAFB"
BLUE,BLUE_L,ORANGE,ORANGE_L,TEAL,TEAL_L,GRAYPT,AMBER_L,AMBER,RED="#1D4ED8","#EFF6FF","#C2410C","#FFF7ED","#0F766E","#F0FDFA","#9CA3AF","#FFFBEB","#B45309","#B91C1C"
VIOLET,VIOLET_L="#6D28D9","#F5F3FF"
MONO="DejaVu Sans Mono"
def mt(s):return str(s).replace("$",r"\$")
def ascii_(s):return unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode("ascii")
def new_fig():return plt.figure(figsize=(FIG_W,FIG_H),dpi=LAYOUT_DPI,facecolor="white")
def sanitize_dashes(fig):
    """Zero-width artists are forced to a solid style: some matplotlib builds raise on a dashed pattern scaled to zero width."""
    for a in fig.findobj(lambda o:isinstance(o,(Patch,Line2D))):
        try:
            if a.get_linewidth()==0 and a.get_linestyle()not in("-","solid","None","none"," ",""):a.set_linestyle("-")
        except Exception:pass
def finish(fig,path,ctx,expect,desc,register=True):
    """Every expected value must appear in the figure text; claim scan must be clean; then JPEG (300 dpi, EXIF provenance) + vector PDF."""
    sanitize_dashes(fig);fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);bw=ws(blob);miss=[e for e in expect if ws(str(e))not in bw]
    if miss:plt.close(fig);raise AuditFail(f"FIGURE/PAYLOAD MISMATCH in {nm}: missing {miss}")
    hits=claim_scan(nm,strip_words(re.sub(r"\s*\n\s*"," ",blob),ctx.get("strip",[])),allow=ctx["allow"],numeric=True)
    if hits:plt.close(fig);raise AuditFail(f"CLAIM-DISCIPLINE HIT in {nm}: {hits[:3]}")
    pdf=Path(path).with_suffix(".pdf")
    fig.savefig(pdf,format="pdf",facecolor="white",metadata={"Title":desc,"Author":f"{AUTHOR} · {ORG}","Subject":f"{DEMO_TITLE} · run {ctx['run_id']} · payload SHA-256 {ctx['sha']}","Creator":DEMO_SHORT,
                "Keywords":f"model {MODEL_ID}; TEST627 demo engine sha256 {ENGINE_SHA256_EXPECTED}; 627.log sha256 {TEST627_LOG['log_sha256']}; license {LICENSE_URL}"})
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=OUT_DPI,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:rgb=im.convert("RGB")
    ex=Image.Exif();ex[0x010E]=ascii_(f"{desc} | {DEMO_SHORT} | run {ctx['run_id']} | payload sha256 {ctx['sha']} | engine sha256 {ENGINE_SHA256_EXPECTED}");ex[0x013B]=ascii_(f"{AUTHOR} / {ORG}");ex[0x8298]=ascii_(f"{COPYRIGHT} - {LICENSE_NAME} - {LICENSE_URL}");ex[0x0131]=ascii_(DEMO_SHORT)
    rgb.save(path,"JPEG",quality=95,optimize=True,progressive=False,subsampling=0,dpi=(OUT_DPI,OUT_DPI),exif=ex.tobytes())
    with Image.open(path)as c_:
        if c_.format!="JPEG"or c_.mode!="RGB":raise RuntimeError("JPEG validation failed")
    if register:ctx["pdfs"].append(pdf)
    ctx["audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),claim_hits=0));return str(path)
def fit_text(fig,x,y,w,h,text,fs_max=10,fs_min=6.5,color=INK,family=None,ls=1.3,weight="normal",ha="left"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else"(empty)").replace("\r","").split("\n");x0=x+w/2 if ha=="center"else x
    kw={"va":"top","ha":ha,"color":color,"linespacing":ls,"weight":weight,"multialignment":ha}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False,subsequent_indent=("  "if p.startswith("• ")else""))or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(150):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x0,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.25);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x0,y+h,mt("\n".join(ln[:keep]+["[… continued in the readable run log]"])),fontsize=fs,**kw);return fs
    fig.text(x0,y+h,"[text omitted — see run log]",fontsize=fs_min,**kw);return fs_min
def header(fig,k,title,sub,tag,tagc,figword="Fig."):
    fig.text(.03,.955,f"{figword} {k}",fontsize=12,weight="bold",color=MUTED,va="center")
    fig.text(.03+(.048 if figword=="Fig."else .056),.955,mt(title),fontsize=16.5,weight="bold",color=INK,va="center")
    fit_text(fig,.03,.902,.94,.032,sub,fs_max=9.4,fs_min=7.6,color=INK2)
    fig.text(.97,.955,mt(tag),fontsize=8.4,weight="bold",color=tagc,va="center",ha="right",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec=tagc,lw=1.0))
    fig.add_artist(Line2D([.03,.97],[.895,.895],transform=fig.transFigure,color=INK2,lw=.8))
def footer(fig,ctx,k,caption,fs_max=8.4):
    if ctx["P"]["engine"].get("instrument")!="ENGINE":
        fig.text(.5,.5,"SYNTHETIC SELF-TEST · NOT A RESULT",fontsize=44,color=RED,alpha=.22,ha="center",va="center",rotation=18,weight="bold",zorder=50)
    fig.add_artist(Line2D([.03,.97],[.132,.132],transform=fig.transFigure,color=RULE,lw=.6))
    fit_text(fig,.03,.045,.94,.08,caption,fs_max=fs_max,fs_min=6.4,color=INK2)
    fig.add_artist(Line2D([.03,.97],[.038,.038],transform=fig.transFigure,color=RULE,lw=.6))
    fig.text(.5,.025,mt(f"{DEMO_TITLE} · {DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}"),ha="center",va="center",fontsize=6.6,color=INK2)
    fig.text(.5,.010,mt(f"live run {ctx['run_id']} · {MODEL_FAMILY} · TEST627 demo engine (SHA-256 {ENGINE_SHA256_EXPECTED[:12]}…) · 627.log {TEST627_LOG['log_sha256'][:12]}… · payload {ctx['sha'][:12]}… · Fig. {k}/{ctx['N']} · {COPYRIGHT}"),
             ha="center",va="center",fontsize=6.2,color=MUTED,family=MONO)
def clean(ax,grid=True):
    for s in("top","right"):ax.spines[s].set_visible(False)
    if grid:ax.grid(True,color=GRIDC,lw=.5);ax.set_axisbelow(True)
def frame(fig,x,y,w,h,ec=INK2,fc="white",lw=.9,ls="-",z=-1):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=lw,ls=ls,zorder=z))
def arrow(fig,x0,y0,x1,y1,color=INK2,lw=1.2,ms=12,cs="arc3",ls="-"):
    fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,connectionstyle=cs,ls=ls,zorder=6))
def sblock(fig,x,y,w,h,stage,title,body,ec,fc="white",bfs=8.4,bmin=6.0,tfs=9.2,mono=False,bpad=.007):
    """Coloured stage block: solid header strip with the stage label, bold title, wrapped body."""
    frame(fig,x,y,w,h,ec,fc,lw=1.0,z=1);frame(fig,x,y+h-.03,w,.03,ec,ec,lw=1.0,z=1)
    fig.text(x+.006,y+h-.015,mt(stage),fontsize=7.4,weight="bold",color="white",va="center",zorder=3)
    fit_text(fig,x+.006,y+h-.08,w-.012,.046,title,fs_max=tfs,fs_min=7.0,color=INK,weight="bold")
    if body:fit_text(fig,x+.006,y+bpad,w-.012,h-.086-bpad,body,fs_max=bfs,fs_min=bmin,family=MONO if mono else None,color=INK2)
def chip(fig,x,y,text,col,fs=7.6):
    fig.text(x,y,mt(text),fontsize=fs,color=col,weight="bold",va="center",ha="left",family=MONO,bbox=dict(boxstyle="round,pad=0.28",fc="white",ec=col,lw=.8),zorder=7)
def disk_icon(fig,x,y,s,col):
    ax=fig.add_axes([x,y,s,s*FIG_W/FIG_H]);ax.set_zorder(4);ax.axis("off");ax.set_xlim(-1,1);ax.set_ylim(-1,1);ax.set_aspect("equal")
    ax.add_patch(Circle((0,0),.95,fc="#E5E7EB",ec=col,lw=1.0));ax.add_patch(Circle((0,0),.22,fc="white",ec=col,lw=.8));ax.plot([.25,.85],[.1,.45],color=col,lw=1.6)
def L_(ctx):return ctx["P"]["live"]
def short(s,n):s=str(s);return s if len(s)<=n else s[:n-1]+"…"
def uwrap(s,n,ind=""):
    out=[];cur=""
    for part in str(s).split("_"):
        piece=(cur+"_"+part)if cur else part
        if len(piece)>n and cur:out.append(cur+"_");cur=part
        else:cur=piece
    out.append(cur);return"\n".join(ind+x for x in out)
def okc(b):return TEAL if b else ORANGE
def yn(b):return"yes"if b else"no"
def tick(b):return"✓"if b else"✗"
def gb(b):return f"{b/1e9:.2f} GB"
# ------------------------------------------------------------------ FIG 1 · WRITE · SSD · READ (one architecture, live values)
ARCH_TXT={
"en":dict(figword="Fig.",title="The Silicon Hippocampus — how a memory is formed, stored and recalled",
 sub="{model} stays frozen · WRITE turns a fact into the model's own key/value state and saves it as a file · READ finds that file from the question alone and answers from it · live values of this run",
 tag="WRITE · SSD · READ · LIVE RUN",laneW="A · WRITE — memory formation (done once per fact, before any question; re-executed for this fact in this run)",
 laneR="B · READ — recall (for every question; the source text is not given again)",
 w1t="Source fact (text)",w1b="“{fact}”\nRead by the model once, only at WRITE time.",
 w2t="Frozen Qwen2.5-7B-Instruct",w2b="Forward 1: instruction prefix + fact ({T} tokens).\nForward 2: zero input embeddings, layer-3 state injected.\nWeights unchanged · no training · no LoRA.",
 w3t="CUT3 keeps the model's own K/V",w3b="Layers 0–3: keys/values of the fact forward. Layers 4–27: rebuilt from the layer-3 state alone.\n{L} × (K, V) = {nt} {dt} tensors [1, 4, {T}, 128].\nNot the text, not a text embedding.",
 w4t="Independent file on disk",w4b="{file} · {fbytes} bytes\nSHA-256 {fsha}…\nRe-written now: bitwise identical {eq}",
 ramt="RAM (no text, no labels; built at start-up)",ramb="Address index: FAISS IVF{lists} over {pos} positions ({ixmb} MB) + PCA{pca} basis ({pcmb} MB).\nReranking data: native layer-3 vectors ({natmb} MB).\nModel weights are in GPU memory (not counted here).",
 ssdt="SSD cartridge pool",ssdb="{n} independent files · {gbv}\nsizes = manifest (this run) {ok1}\nSHA-256 of all files (start-up) {ok2}",
 ragt="Why this is not text RAG",ragb="Text RAG: finds text → pastes the text into the prompt.\nMAM: finds a cartridge → loads the model's own stored K/V into attention memory; the source text is not sent again.\nNo vector-DB service · no SQL · no training.",
 r1t="Question only",r1b="“{q}”\nThe fact is not in the prompt.",
 r2t="Frozen Qwen reads the question",r2b="One forward: instruction prefix + question ({ft} tokens), no fact text.\n→ {rows} layer-3 question vectors.",
 r3t="Geometric address (RAM)",r3b="PCA{pca} → IVF{lists}\nnprobe {npb} · token_k {tk}\ntouched {touch} of {n}\n→ {nc} candidates",
 r4t="PEAK50 selection",r4b="Re-scores the {nc} candidates with the question vectors and picks one.\n→ #{sel}",
 r5t="Load the file from SSD",r5b="{selfile}\nSHA-256 = manifest {shaok}\n{T2} rows → GPU attention memory",
 r6t="Frozen model answers",r6b="Input: {qt} question-template tokens over the loaded cartridge.\n→ “{ans}”",
 chipA="address {a:.1f} ms",chipL="load {l:.1f} ms",chipN="answer {n:.0f} ms",tag2="this run · case {case:02d}",
 cap=("Figure {k}. One fact through the whole system in run {rid}. WRITE: the engine's init(fact, 3) produced the {L}-layer key/value cartridge again and it is bitwise identical to the file written at start-up. "
      "SSD: {n} cartridges are {n} independent files. READ: from the question alone the label-free address and the locked PEAK50 rule chose #{sel}; the file was loaded and the frozen model answered “{ans}” "
      "(expected “{exp}”: {res}). Chips: single-run timings of separate stages, not summed; load may be served from the OS page cache. Separate from this run: the engine's start-up acceptance in this runtime "
      "(original {o}/48, new wording {a}/48). Limits: the 96 questions reuse 48 facts and PEAK50 was fixed after TEST626; 1,453 files are not millions; ~{avg} MB per fact; A100 only.")),
"tr":dict(figword="Şekil",title="Silikon Hipokampüs — hafıza nasıl oluşur, saklanır ve hatırlanır",
 sub="{model} dondurulmuş kalır · WRITE bilgiyi modelin kendi anahtar/değer durumuna çevirip dosyaya yazar · READ bu dosyayı yalnızca sorudan bulur ve cevabı ondan üretir · bu çalıştırmanın canlı değerleri",
 tag="WRITE · SSD · READ · CANLI",laneW="A · WRITE — hafızanın oluşumu (her bilgi için bir kez, sorudan önce; bu bilgi için bu çalıştırmada yeniden yürütüldü)",
 laneR="B · READ — hatırlama (her soruda; kaynak metin tekrar verilmez)",
 w1t="Kaynak bilgi (metin)",w1b="“{fact}”\nModel bunu yalnızca WRITE anında bir kez okur.",
 w2t="Dondurulmuş Qwen2.5-7B-Instruct",w2b="İleri geçiş 1: talimat öneki + bilgi ({T} token).\nİleri geçiş 2: sıfır girdi, katman-3 durumu enjekte.\nAğırlıklar değişmez · eğitim yok · LoRA yok.",
 w3t="CUT3 modelin kendi K/V'sini saklar",w3b="Katman 0–3: bilgi geçişinin anahtar/değerleri. Katman 4–27: yalnızca katman-3 durumundan yeniden üretilir.\n{L} × (K, V) = {nt} {dt} tensör [1, 4, {T}, 128].\nMetin değil, metin gömmesi değil.",
 w4t="Diskte bağımsız dosya",w4b="{file} · {fbytes} bayt\nSHA-256 {fsha}…\nŞimdi yeniden yazıldı: bit düzeyinde aynı {eq}",
 ramt="RAM (metin yok, etiket yok; açılışta kuruldu)",ramb="Adres dizini: {pos} konum üzerinde FAISS IVF{lists} ({ixmb} MB) + PCA{pca} tabanı ({pcmb} MB).\nYeniden sıralama verisi: yerel katman-3 vektörleri ({natmb} MB).\nModel ağırlıkları GPU belleğinde (burada sayılmaz).",
 ssdt="SSD kartuş havuzu",ssdb="{n} bağımsız dosya · {gbv}\nboyutlar = manifest (bu çalıştırma) {ok1}\ntüm dosyaların SHA-256'sı (açılış) {ok2}",
 ragt="Neden metin RAG değil",ragb="Metin RAG: metni bulur → metni isteme yapıştırır.\nMAM: kartuşu bulur → modelin kendi saklanmış K/V'sini dikkat belleğine yükler; kaynak metin yeniden gönderilmez.\nVektör DB servisi yok · SQL yok · eğitim yok.",
 r1t="Yalnızca soru",r1b="“{q}”\nBilgi istemde yok.",
 r2t="Dondurulmuş Qwen soruyu okur",r2b="Tek geçiş: talimat öneki + soru ({ft} token), bilgi metni yok.\n→ {rows} katman-3 soru vektörü.",
 r3t="Geometrik adres (RAM)",r3b="PCA{pca} → IVF{lists}\nnprobe {npb} · token_k {tk}\ndokunulan {touch} / {n}\n→ {nc} aday",
 r4t="PEAK50 seçimi",r4b="{nc} adayı soru vektörleriyle yeniden puanlar ve birini seçer.\n→ #{sel}",
 r5t="Dosyayı SSD'den yükle",r5b="{selfile}\nSHA-256 = manifest {shaok}\n{T2} satır → GPU dikkat belleği",
 r6t="Dondurulmuş model cevaplar",r6b="Girdi: yüklenen kartuş üzerinde {qt} soru-şablonu tokenı.\n→ “{ans}”",
 chipA="adres {a:.1f} ms",chipL="yükleme {l:.1f} ms",chipN="cevap {n:.0f} ms",tag2="bu çalıştırma · vaka {case:02d}",
 cap=("Şekil {k}. Çalıştırma {rid} içinde tek bir bilginin sistemdeki yolu. WRITE: motorun init(bilgi, 3) çağrısı {L} katmanlı anahtar/değer kartuşunu yeniden üretti; açılışta yazılan dosyayla bit düzeyinde aynı. "
      "SSD: {n} kartuş {n} bağımsız dosyadır. READ: yalnızca sorudan, etiketsiz adres ve kilitli PEAK50 kuralı #{sel} kartuşunu seçti; dosya yüklendi ve dondurulmuş model “{ans}” cevabını verdi "
      "(beklenen “{exp}”: {res}). Süre etiketleri bu çalıştırmadaki ayrı aşamalardır, toplanmaz; yükleme işletim sistemi önbelleğinden gelmiş olabilir. Bu çalıştırmadan ayrı: motorun bu ortamdaki açılış kabulü "
      "(özgün {o}/48, yeni ifade {a}/48). Sınırlar: 96 soru 48 bilgiyi tekrar kullanır, PEAK50 TEST626'dan sonra sabitlendi; 1.453 dosya milyonlar değildir; bilgi başına ~{avg} MB; yalnızca A100."))}
def f_arch(ctx,k,path,lang="en"):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];su=P["startup"];Wr=L["write"];Sd=L["ssd"];Rd=L["read"];ix=su["index"];T_=ARCH_TXT[lang];fig=new_fig()
    n=cfg["n_cartridges"];S=su["summary"];avg=f"{su['storage']['avg_file_bytes']/1e6:.2f}"
    header(fig,k,T_["title"],T_["sub"].format(model=MODEL_FAMILY),T_["tag"],TEAL,figword=T_["figword"])
    # WRITE lane
    frame(fig,.03,.632,.94,.252,ec=BLUE,fc=BLUE_L,lw=1.1,z=0)
    fig.text(.038,.869,mt(T_["laneW"]),fontsize=9.6,weight="bold",color=BLUE,va="center")
    wy,wh=.641,.212;W=[(.038,.205),(.258,.215),(.488,.235),(.738,.224)]
    wb=[("1",T_["w1t"],T_["w1b"].format(fact=L["fact"])),("2",T_["w2t"],T_["w2b"].format(T=Wr["T"])),
        ("3",T_["w3t"],T_["w3b"].format(L=Wr["layers"],nt=Wr["tensors"],dt=Wr["dtype"],T=Wr["T"])),
        ("4",T_["w4t"],T_["w4b"].format(file=Wr["file"],fbytes=f"{Wr['file_bytes']:,}",fsha=Wr["file_sha256"][:16],eq=tick(Wr["bit_equal"])))]
    for(x,w),(s,t,b)in zip(W,wb):sblock(fig,x,wy,w,wh,s,t,b,BLUE,bfs=8.2)
    for i in range(3):arrow(fig,W[i][0]+W[i][1]+.002,wy+wh*.5,W[i+1][0]-.002,wy+wh*.5,BLUE)
    # middle row
    my,mh=.452,.168
    sblock(fig,.038,my,.29,mh,"RAM",T_["ramt"],T_["ramb"].format(pca=ix["pca_dims"],lists=ix["ivf_lists"],pos=f"{ix['native_positions']:,}",ixmb=f"{ix['index_bytes']/1e6:.1f}",pcmb=f"{ix['pca_basis_bytes']/1e6:.1f}",natmb=f"{ix['native_activation_bytes']/1e6:.0f}"),VIOLET,VIOLET_L,bfs=7.8)
    sblock(fig,.343,my,.314,mh,"≠ RAG",T_["ragt"],T_["ragb"],INK2,"white",bfs=8.0)
    sblock(fig,.672,my,.29,mh,"SSD",T_["ssdt"],"",AMBER,AMBER_L);disk_icon(fig,.680,my+.014,.045,AMBER)
    fit_text(fig,.732,my+.008,.224,mh-.093,T_["ssdb"].format(n=f"{n:,}",gbv=gb(Sd["total_bytes"]),ok1=tick(not Sd["size_mismatch"]),ok2=tick(Sd["startup_sha_verified"])),fs_max=8.4,fs_min=6.4,color=INK2)
    arrow(fig,.85,wy-.002,.817,my+mh+.002,AMBER,lw=1.8);arrow(fig,.31,wy-.002,.24,my+mh+.002,VIOLET,lw=1.3,ls="--")
    # READ lane
    frame(fig,.03,.142,.94,.300,ec=TEAL,fc=TEAL_L,lw=1.1,z=0)
    fig.text(.038,.157,mt(T_["laneR"]),fontsize=9.6,weight="bold",color=TEAL,va="center")
    fig.text(.962,.157,mt(T_["tag2"].format(case=L["case"])),fontsize=7.8,color=TEAL,weight="bold",va="center",ha="right")
    ry,rh=.176,.25;bw=.146;gap=(.932-6*bw)/5;X=[.038+i*(bw+gap)for i in range(6)]
    rb=[("5",T_["r1t"],T_["r1b"].format(q=L["question"])),("6",T_["r2t"],T_["r2b"].format(ft=Rd["frame_tokens"],rows=Rd["query_rows"])),
        ("7",T_["r3t"],T_["r3b"].format(pca=ix["pca_dims"],lists=ix["ivf_lists"],npb=cfg["candidates"][0],tk=cfg["candidates"][1],touch=Rd["touched"],n=f"{n:,}",nc=len(Rd["ranking"]))),
        ("8",T_["r4t"],T_["r4b"].format(nc=len(Rd["ranking"]),sel=f"{Rd['selected']:04d}")),("9",T_["r5t"],T_["r5b"].format(selfile=Rd["file"],shaok=tick(Rd["sha_ok"]),T2=Rd["T"])),
        ("10",T_["r6t"],T_["r6b"].format(qt=Rd["query_tokens"],ans=short(L["answer"],28)))]
    for i,(x,(s,t,b))in enumerate(zip(X,rb)):sblock(fig,x,ry,bw,rh,s,t,b,TEAL,"white",bfs=7.9,bmin=5.8,tfs=8.8,bpad=.04 if i in(2,4,5)else .007)
    for i in range(5):arrow(fig,X[i]+bw+.001,ry+rh*.5,X[i+1]-.001,ry+rh*.5,TEAL)
    arrow(fig,.30,my-.002,X[2]+bw*.5,ry+rh+.002,VIOLET,lw=1.4);arrow(fig,.74,my-.002,X[4]+bw*.5,ry+rh+.002,AMBER,lw=1.8)
    chip(fig,X[2]+.006,ry+.018,T_["chipA"].format(a=Rd["address_ms"]),TEAL);chip(fig,X[4]+.006,ry+.018,T_["chipL"].format(l=Rd["load_ms"]),AMBER);chip(fig,X[5]+.006,ry+.018,T_["chipN"].format(n=Rd["answer_ms"]),TEAL)
    res=("correct"if L["correct"]else"incorrect")if lang=="en"else("doğru"if L["correct"]else"yanlış")
    footer(fig,ctx,k,T_["cap"].format(k=k,rid=P["run_id"],L=Wr["layers"],n=f"{n:,}",sel=f"{Rd['selected']:04d}",ans=L["answer"],exp=L["expected"],res=res,o=S["ORIGINAL"]["answer_correct"],a=S["ALTERNATE"]["answer_correct"],avg=avg),fs_max=8.0)
    exp=[Wr["file"],Rd["file"],f"#{Rd['selected']:04d}",Wr["file_sha256"][:16]]
    return finish(fig,path,ctx,exp,"WRITE · SSD · READ architecture"+(" (Turkish)"if lang=="tr"else""),register=(lang=="en"))
def f1(ctx,k,path):return f_arch(ctx,k,path,"en")
# ------------------------------------------------------------------ FIG 2 · WRITE → SSD in detail
def f2(ctx,k,path):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];su=P["startup"];Wr=L["write"];Sd=L["ssd"];fig=new_fig();n=cfg["n_cartridges"]
    header(fig,k,"WRITE → SSD · how one memory cartridge is formed and stored",f"The fact of case {L['case']:02d} written again by the engine's init(fact, 3) in this run and compared with the file written at start-up · the pool of {n:,} files",
           "WRITE · SSD · LIVE",BLUE)
    y0,h0=.66,.215;bx=[(.03,.17),(.215,.205),(.435,.205)]
    f1_,f2_=Wr["forwards"][0],Wr["forwards"][1]
    sblock(fig,bx[0][0],y0,bx[0][1],h0,"INPUT","Instruction prefix + fact",f"“{L['fact']}”\n{Wr['prefix_tokens']} prefix + {Wr['fact_tokens']} fact tokens = {Wr['T']} rows",BLUE,bfs=8.2)
    sblock(fig,bx[1][0],y0,bx[1][1],h0,"FORWARD 1","Token-id pass over the text",f"{fw_txt(f1_)}\nkeeps K/V of layers 0–3\ncaptures the layer-3 hidden state (H3)",BLUE,bfs=8.0,mono=True)
    sblock(fig,bx[2][0],y0,bx[2][1],h0,"FORWARD 2","Zero-input pass, H3 injected",f"{fw_txt(f2_)}\nH3 replaces the layer-3 output\n→ K/V of layers 4–27",BLUE,bfs=8.0,mono=True)
    arrow(fig,bx[0][0]+bx[0][1]+.002,y0+h0*.5,bx[1][0]-.002,y0+h0*.5,BLUE);arrow(fig,bx[1][0]+bx[1][1]+.002,y0+h0*.5,bx[2][0]-.002,y0+h0*.5,BLUE)
    # cartridge strip: 28 layers, K and V
    ax=fig.add_axes([.06,.37,.57,.23]);NL=Wr["layers"];T=Wr["T"]
    for li in range(NL):
        col=BLUE if li<=CUT_EXPECTED else"#60A5FA"
        ax.add_patch(Rectangle((li,0.55),.9,.4,fc=col,ec="white",lw=.5));ax.add_patch(Rectangle((li,0.05),.9,.4,fc=col,ec="white",lw=.5,alpha=.75))
    ax.set_xlim(-.4,NL+.2);ax.set_ylim(-.35,1.25);ax.set_yticks([.25,.75]);ax.set_yticklabels(["V","K"]);ax.set_xticks([1.95,15.45]);ax.set_xticklabels(["layers 0–3","layers 4–27"])
    for s in("top","right","left"):ax.spines[s].set_visible(False)
    ax.text(1.9,1.1,"from the fact forward (layers 0–3)",fontsize=8,color=BLUE,ha="center",weight="bold");ax.text(15.5,1.1,"rebuilt from the layer-3 state alone (layers 4–27)",fontsize=8,color="#2563EB",ha="center",weight="bold")
    ax.set_title(f"The cartridge: {NL} layers × (K, V) = {Wr['tensors']} tensors, each {Wr['shape']} {Wr['dtype']} · {Wr['memory_bytes']:,} bytes",loc="left",fontsize=9.2)
    # file box
    fx,fw_=.66,.31
    frame(fig,fx,.30,fw_,.575,ec=AMBER,fc=AMBER_L,lw=1.0);fig.text(fx+.01,.858,"FILE ON DISK · THIS FACT",fontsize=8.8,weight="bold",color=AMBER,va="center")
    body=(f"{Wr['file']}\n{Wr['file_bytes']:,} bytes on disk\n= {Wr['memory_bytes']:,} tensor bytes\n+ {Wr['container_overhead']:,} container bytes\nSHA-256 (= manifest {tick(Wr['sha_ok'])})\n  {Wr['file_sha256'][:32]}\n  {Wr['file_sha256'][32:]}\n"
          f"written at start-up in {Wr['startup_write_ms']:.0f} ms\n\nTHIS RUN\nre-written by init(fact, 3): {1000*Wr['seconds']:.0f} ms\nbitwise identical in all {Wr['layers']} layers {tick(Wr['bit_equal'])}\n"
          f"fact text inside the file: {yn(Wr['fact_in_file'])}\nanswer word inside the file: {yn(Wr['gold_in_file'])}\nquestion inside the file: {yn(Wr['question_in_file'])}")
    fit_text(fig,fx+.01,.31,fw_-.02,.53,body,fs_max=8.8,fs_min=6.4,family=MONO,color=INK)
    # pool histogram
    ax2=fig.add_axes([.06,.175,.40,.075]);sz=[s/1e6 for s in su["sizes"]];ax2.hist(sz,bins=40,color="#FCD34D",edgecolor=AMBER,lw=.4);clean(ax2)
    ax2.axvline(Wr["file_bytes"]/1e6,color=BLUE,lw=1.4);ax2.set_xlabel("file size (MB) · blue line = this fact",fontsize=7.6,labelpad=1);ax2.tick_params(labelsize=7)
    ax2.set_title(f"SSD pool · {n:,} independent files",loc="left",fontsize=8.8)
    fit_text(fig,.49,.16,.48,.11,(f"{n:,} files present in this run, every size = manifest {tick(not Sd['size_mismatch'])} · total {Sd['total_bytes']:,} bytes ({gb(Sd['total_bytes'])})\n"
              f"min {Sd['min']/1e6:.2f} · median {Sd['median']/1e6:.2f} · max {Sd['max']/1e6:.2f} MB · SHA-256 of all {n:,} files verified by the engine at start-up {tick(Sd['startup_sha_verified'])}\n"
              f"start-up WRITE: {su['write']['files']:,} files · p50 {su['write']['ms_p50']:.0f} ms per file · {su['write']['seconds_total']:.0f} s in total"),fs_max=8.4,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. WRITE and SSD for case {L['case']:02d}. The frozen model reads the fact once: forward 1 keeps the keys and values of layers 0–3 and the layer-3 hidden state; forward 2 starts from all-zero input embeddings, "
           f"injects that state at layer 3 and produces the keys and values of layers 4–27. The {Wr['tensors']} tensors are the cartridge; they were saved at start-up as {Wr['file']}, and the same call in this run gives a bitwise identical result. "
           f"The file holds tensors only: neither the fact text nor the answer word is in it. Sizes: {Wr['file_bytes']:,} bytes for this fact; {gb(Sd['total_bytes'])} for {n:,} facts.")
    return finish(fig,path,ctx,[Wr["file"],f"{Wr['file_bytes']:,}",Wr["file_sha256"][:32],f"{n:,}"],"WRITE and SSD")
# ------------------------------------------------------------------ FIG 3 · READ in detail
def f3(ctx,k,path):
    P=ctx["P"];L=L_(ctx);cfg=P["engine"]["config"];Rd=L["read"];Ct=L["control"];ix=P["startup"]["index"];fig=new_fig();n=cfg["n_cartridges"]
    header(fig,k,"READ · from a question to the right file and an answer",f"Case {L['case']:02d} · “{L['question']}” · label-free geometric address in RAM → locked PEAK50 selection → file from SSD → frozen model answers",
           "READ · LIVE",TEAL)
    y0,h0=.70,.17;W_=.148;G_=(.94-6*W_)/5;X=[.03+i*(W_+G_)for i in range(6)];f0=Rd["frame_forwards"][0]
    st=[("1","Question frame",f"{fw_txt(f0)}\ninstruction prefix + question only"),("2","Layer-3 vectors",f"{Rd['query_rows']} question rows\n(engine question_rep + apply)\nencode {Rd['encode_ms']:.0f} ms (not in address)"),
        ("3",f"PCA{ix['pca_dims']} + IVF{ix['ivf_lists']}",f"nprobe {cfg['candidates'][0]} · token_k {cfg['candidates'][1]}\nFAISS {Rd['faiss_ms']:.2f} ms"),
        ("4","Touched cartridges",f"{Rd['touched']} of {n:,}\n({100*Rd['touched']/n:.1f} % of the pool)"),("5","TOP2 → candidates",f"{len(Rd['ranking'])} candidates\nTOP2 {Rd['top2_ms']:.1f} ms"),
        ("6","PEAK50 → one file",f"selected #{Rd['selected']:04d}\nPEAK50 {Rd['rerank_ms']:.1f} ms\naddress total {Rd['address_ms']:.1f} ms\n(= steps 3–6, as TEST627)")]
    for i,(s,t,b)in enumerate(st):sblock(fig,X[i],y0,W_,h0,s,t,b,TEAL,bfs=8.0,mono=True)
    for i in range(5):arrow(fig,X[i]+W_+.001,y0+h0*.5,X[i+1]-.001,y0+h0*.5,TEAL)
    # candidate scores
    ax=fig.add_axes([.07,.30,.45,.33]);C=Rd["candidates"];ys=list(range(len(C)))[::-1]
    cols=[TEAL if i==0 else"#CBD5E1"for i in range(len(C))]
    ax.barh(ys,[c["score"]for c in C],color=cols,edgecolor=[TEAL if i==0 else"#94A3B8"for i in range(len(C))],lw=.6)
    lo=min(c["score"]for c in C);hi=max(c["score"]for c in C);pad=max(1e-4,(hi-lo)*.15);ax.set_xlim(lo-pad,hi+pad*2.2)
    ax.set_yticks(ys);ax.set_yticklabels([f"#{c['id']:04d}"for c in C],fontsize=7,family=MONO);clean(ax);ax.grid(axis="y",visible=False)
    for i,c in enumerate(C):
        if c["id"]==L["target"]:ax.text(c["score"]+pad*.15,ys[i],"◆ expected (audit)",fontsize=7,color=AMBER,va="center",weight="bold")
    ax.set_xlabel("PEAK50 score = 0.5 × FRAM + 0.5 × peak (engine features626 / rerank626)",fontsize=7.8)
    ax.set_title(f"The {len(C)} candidates ranked by PEAK50 · teal = selected",loc="left",fontsize=9.2)
    # load + answer
    bx=.56;bw=.41
    frame(fig,bx,.47,bw,.18,ec=AMBER,fc=AMBER_L,lw=1.0);fig.text(bx+.01,.632,"LOAD FROM SSD",fontsize=8.6,weight="bold",color=AMBER,va="center")
    fit_text(fig,bx+.01,.478,bw-.02,.14,f"{Rd['file']} · SHA-256 = manifest {tick(Rd['sha_ok'])}\ntorch.load → {Rd['T']} rows × {ARCH[0]} layers → GPU\nload {Rd['load_ms']:.1f} ms = SHA-256 + torch.load + copy to GPU\n(may be served from the OS page cache)",fs_max=8.6,fs_min=6.4,family=MONO,color=INK)
    frame(fig,bx,.27,bw,.185,ec=TEAL,fc=TEAL_L,lw=1.0);fig.text(bx+.01,.437,"ANSWER · SOURCE TEXT NOT GIVEN",fontsize=8.6,weight="bold",color=TEAL,va="center")
    fa=Rd["answer_forwards"][0]
    fit_text(fig,bx+.01,.278,bw-.02,.145,(f"model input: {fa['n']} question-template tokens over the {Rd['T']}-row cartridge (cache {fa['past_len']})\n"
             f"then {Rd['gen_forwards']} one-token generation steps · {Rd['answer_ms']:.0f} ms\nanswer “{L['answer']}” · expected “{L['expected']}” → {'CORRECT'if L['correct']else'INCORRECT'}"),fs_max=8.6,fs_min=6.4,family=MONO,color=INK)
    frame(fig,.03,.15,.94,.10,ec=INK2,fc="white",lw=.9);fig.text(.04,.235,"CONTROL · runner-up file (diagnostic)",fontsize=8.6,weight="bold",color=INK2,va="center")
    ctl=("skipped: "+Ct["skipped"])if Ct.get("skipped")else(f"Same question over the second-ranked candidate #{Ct['id']:04d} ({Ct['file']}): answer “{Ct['answer']}” · expected answer produced: {yn(Ct['gold_produced'])}"
         +(" · this runner-up is the expected cartridge"if Ct["is_target"]else"")+f". Start-up engine row for this question: selected #{L['startup_row']['selected']} · answer “{L['startup_row']['answer']}” (same selection: {yn(L['matches_startup_selection'])}, same answer: {yn(L['matches_startup_answer'])}).")
    fit_text(fig,.04,.156,.92,.066,ctl,fs_max=8.6,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. READ for case {L['case']:02d}. The question is read in one forward pass with the instruction prefix only — no fact text. Its layer-3 vectors are projected with the label-free PCA basis and looked up in the FAISS IVF index; "
           f"{Rd['touched']} of {n:,} cartridges are touched, {len(Rd['ranking'])} become candidates and the locked PEAK50 rule selects #{Rd['selected']:04d}. Its file is verified against the manifest, loaded, and the frozen model answers from it. "
           "The expected cartridge and answer are audit metadata only; they are not used by addressing, selection or answering.")
    return finish(fig,path,ctx,[Rd["file"],f"#{Rd['selected']:04d}",str(Rd["touched"])],"READ")
# ------------------------------------------------------------------ FIG 4 · RESULTS, validation and limits
def f4(ctx,k,path):
    P=ctx["P"];L=L_(ctx);su=P["startup"];fz=P["frozen"];S=su["summary"];T7=TEST627_LOG;fig=new_fig();Rd=L["read"];Wr=L["write"]
    header(fig,k,"RESULTS · measured performance, verification and limits","Start-up acceptance of the TEST627 engine in this runtime · the sealed TEST627 log · this run — side by side, with the limits that apply to all of them","RESULTS · VALIDATION",INK2)
    def col(x,ttl,c,body):
        frame(fig,x,.395,.30,.475,ec=c,fc="white",lw=.9);fig.text(x+.01,.852,ttl,fontsize=8.6,weight="bold",color=c,va="center")
        fit_text(fig,x+.01,.402,.282,.43,body,fs_max=9.6,fs_min=6.2,family=MONO,color=INK)
    def e2e(d):return(f"               original  new wording\nselected       {d['ORIGINAL']['selected_correct']:>2}/48     {d['ALTERNATE']['selected_correct']:>2}/48\nanswered       {d['ORIGINAL']['answer_correct']:>2}/48     {d['ALTERNATE']['answer_correct']:>2}/48\n"
                      f"errors         {d['ORIGINAL']['errors']:>5}     {d['ALTERNATE']['errors']:>5}\naddress p50 ms {d['ORIGINAL']['address_p50_ms']:>6.1f}    {d['ALTERNATE']['address_p50_ms']:>6.1f}\nload p50 ms    {d['ORIGINAL']['load_p50_ms']:>6.1f}    {d['ALTERNATE']['load_p50_ms']:>6.1f}\n"
                      f"answer p50 ms  {d['ORIGINAL']['answer_p50_ms']:>6.0f}    {d['ALTERNATE']['answer_p50_ms']:>6.0f}")
    st=su["storage"];I8=su["integrity"]
    col(.03,"START-UP ACCEPTANCE · THIS RUNTIME",BLUE,f"TEST627 engine, run at start-up\nfinished {str(su.get('measured_utc',''))[:19]} UTC\n\n"+e2e(S)+f"\n\nfiles {st['cartridges']:,} · {st['disk_pool_bytes']:,} B\ntensor payload {st['total_tensor_payload_bytes']:,} B\nintegrity {sum(bool(v)for v in I8.values())}/{len(I8)}: "
        +", ".join(k_.replace("_"," ")for k_,v in I8.items()if v)[:150]+f"\nresult SHA-256 {su['result_sha256'][:24]}…")
    col(.35,"ARCHIVED RECORD · 627.log",AMBER,"repository log of 12 October 2026\nnot re-run, reproduced verbatim\n\n"+e2e(T7["e2e"])+f"\n\nfiles {T7['storage']['cartridges']:,} · {T7['storage']['disk_pool_bytes']:,} B\nintegrity {sum(T7['integrity'].values())}/{len(T7['integrity'])}\ndecision\n"
        +uwrap(T7["decision"],34,"  ")+f"\nlog SHA-256 {T7['log_sha256'][:24]}…")
    col(.67,f"THIS RUN · CASE {L['case']:02d}",TEAL,(f"one new run · {P['run_start_utc'][:19]} UTC\n\nWRITE re-check   bitwise {yn(Wr['bit_equal'])}\nSSD files        {L['ssd']['files_in_pool']:,} = manifest\nselected         #{Rd['selected']:04d} ({'expected'if Rd['selected_correct']else'NOT expected'})\n"
        f"answer           {short(L['answer'],18)}\nexpected         {short(L['expected'],18)}\nresult           {'CORRECT'if L['correct']else'INCORRECT'}\naddress / load   {Rd['address_ms']:.1f} / {Rd['load_ms']:.1f} ms\nanswer           {Rd['answer_ms']:.0f} ms\n"
        f"control leak     {yn(L['control'].get('gold_produced'))}\nweights          unchanged\ntrainable        {fz['trainable_tensors']}\nchecks           {ctx['checks']} pre-seal PASS\nverdict\n"+uwrap(P["verdict"],34,"  ")))
    frame(fig,.03,.15,.94,.23,ec=ORANGE,fc=ORANGE_L,lw=1.0);fig.text(.04,.365,"LIMITS — SHOWN, NOT HIDDEN",fontsize=8.8,weight="bold",color=ORANGE,va="center")
    T6=TEST626_LOG
    fit_text(fig,.04,.16,.92,.19,(f"• Not a blind test: the 96 questions reuse the same 48 facts. TEST626's own pre-declared rule locked {T6['locked_by_own_rule']} ({T6['e2e']} answered); PEAK50 ({T6['peak50_new_wording']} new wording) was fixed after those results were known. FRAM-only ranking chose the right file for {T6['fram_only_new_wording']}.\n"
              f"• 1,453 files do not show behaviour at millions of cartridges. Disk reads may come from the operating-system page cache (no cold-SSD claim).\n"
              f"• A cartridge is ~{st['avg_file_bytes']/1e6:.2f} MB per fact — the model's working state, not a compact text copy. RAM keeps the address index (FAISS {su['index']['index_bytes']/1e6:.1f} MB) and the reranking data (native layer-3 vectors, {su['index']['native_activation_bytes']/1e6:.0f} MB); model weights are in GPU memory.\n"
              f"• Lossless gzip: ratio {st['gzip_sample_ratio']} on {st['lossless_gzip_sample_files']} sample files only; the full-pool value is an estimate. • Synthetic capital-of-country facts; one model. • Measured on an NVIDIA A100 40 GB only; other hardware is not tested. • SHA-256 values show files are unchanged; they are not proofs.\n"
              f"• The three columns are separate measurements: no counter merges them."),fs_max=9.4,fs_min=6.4,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Results of run {P['run_id']} (verdict {P['verdict']}). The left column was produced by the locked TEST627 engine at start-up in this runtime; the middle column is the sealed repository log; the right column is this run. "
           f"Earlier public records: DOI {PRIOR_PUB['doi']} · DOI {QWEN_PUB['doi']} · DOI {V12_PUB['doi']}. Payload SHA-256 {ctx['sha']}.")
    return finish(fig,path,ctx,[T7["decision"][:20],ctx["sha"][:12],f"{S['ORIGINAL']['answer_correct']:>2}/48"],"Results and validation")
FIGURES=[("fig_01_write_ssd_read","Fig. 1 · The Silicon Hippocampus — WRITE · SSD · READ",f1),("fig_02_write_ssd","Fig. 2 · WRITE → SSD: one cartridge formed and stored",f2),
         ("fig_03_read","Fig. 3 · READ: from the question to the file and the answer",f3),("fig_04_results","Fig. 4 · Results, verification and limits",f4)]
N_FIGS=len(FIGURES)
def render_all(ctx,run_dir):
    imgs=[];extra=[]
    try:
        for i,(slug,cap,fn)in enumerate(FIGURES,1):
            p=Path(run_dir)/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
        p=Path(run_dir)/f"fig_01_TR_yaz_ssd_oku_{ctx['run_id']}.jpg";f_arch(ctx,1,p,"tr");extra=[p,p.with_suffix(".pdf")]
    finally:plt.close("all")
    return imgs,extra
#<<FIGURES_END>>
say("PART 2 / 3 complete — RUN pipeline and figures defined. Now run PART 3 / 3 in the next cell.")
MAM_SH_PART2_OK=True
# =====================================================================================================================================
# AKBASCORE MAM — THE SILICON HIPPOCAMPUS · QWEN2.5-7B · PART 3 / 3 · SELF-TEST AND GRADIO DEMONSTRATION
# Same demo as PARTS 1 and 2. Run this cell after PART 2 / 3, in the same Colab runtime. It runs a GPU-free self-test of the full
# pipeline (including fail-closed cases) and then prints a public gradio.live link. Press RUN — there is nothing to type or choose.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 12 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_SH_PART2_OK"):raise RuntimeError("PART 2 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<UI_BEGIN>>
_SYN_L="ABCDEFGHJKLMNPQRSTUVWXYZ"
class _SynMem:
    """Synthetic memory handle for the self-test (no tensors)."""
    def __init__(s,key,T,salt=""):s.key=key;s.T=T;s.salt=salt
class SelfTestInstrument:
    """Synthetic stand-in used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok",tmp=None):
        s.kind="SELF-TEST";s.mode=mode;s.calls=0;s.P=PREFIX_EXPECTED;rng=random.Random(627)
        def w(n):return"".join(rng.choice(_SYN_L)for _ in range(n)).capitalize()
        s.cars=[]
        for i in range(N_CART):
            ctry=f"ZX{624000+i:06d}"if i>=N_ORIG else w(6);city=w(10)
            s.cars.append(dict(id=i,fact=f"The current capital of {ctry} is {city}.",gold=city,country=ctry,body_len=17+i%3))
        s.case_t=[N_ORIG+7*j+3 for j in range(N_CASES)]
        s.qs=[(f"Name the present capital of {s.cars[t]['country']}."if j%2==0 else f"Which city currently serves as the capital of {s.cars[t]['country']}?")for j,t in enumerate(s.case_t)]
        s.sizes=[57344*(s.P+c["body_len"])+16437+(i%5)for i,c in enumerate(s.cars)]
        s.tmp=Path(tmp or ROOT/"_selftest_engine");s.tmp.mkdir(parents=True,exist_ok=True);s.res=s.tmp/"SYN_ENGINE_RESULT.json"
        s.res.write_text(json.dumps({"test":"627","status":"COMPLETE","note":"synthetic"}),encoding="utf-8")
        s.info=dict(model_id=MODEL_ID,arch=list(ARCH),dtype="bfloat16",attn="sdpa",gpu="SELF-TEST INSTRUMENT (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",python="n/a",
            platform="n/a",params=1,engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,init_seconds=0.0,startup_utc="n/a",
            chain_sha256={"626.py":CHAIN["626.py"]["sha256"],"624.py":CHAIN["624.py"]["sha256"],"610.py":SHA610_EXPECTED},source_blobs=dict(SOURCE_BLOBS_EXPECTED),
            weight_sha0=WEIGHT_SHA_EXPECTED,sentinel0="a"*64,sentinel_engine="a"*64,guard0="b"*64,hooks0=0,sentinel_method="synthetic",guard_method="synthetic",full_weight_method="synthetic",forward_counter="synthetic")
    def config(s):
        return dict(MODEL_ID=MODEL_ID,arch=list(ARCH),CUT=CUT_EXPECTED,prefix_tokens=s.P,n_cartridges=N_CART,n_original=N_ORIG,manifest=N_CART,rank=RANK_EXPECTED,nlist=NLIST_EXPECTED,ntok=26738,dim=3584,
                    index_ntotal=26738,candidates=list(LOCKED_CAND),reranker=list(LOCKED_RERANK),reranker_family=["FRAM","PEAK25","PEAK50"],beta=8.0,max_new=16,pool="synthetic",
                    engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,chain_sha256=dict(s.info["chain_sha256"]),lock_610=SHA610_EXPECTED,source_blobs=dict(SOURCE_BLOBS_EXPECTED),
                    source_blobs_measured=dict(SOURCE_BLOBS_EXPECTED),baseline_610=dict(H1=12,H5=12),n_questions=N_CASES)
    def startup(s):
        e=TEST627_LOG["e2e"];st=dict(TEST627_LOG["storage"]);st["disk_pool_bytes"]=sum(s.sizes);st.update(median_file_bytes=float(np.median(s.sizes)),file_container_overhead_bytes=1,tensor_dtype_bytes={"torch.bfloat16":1},
                lossless_gzip_sample_files=32,gzip_sample_estimated_pool_bytes=1,gzip_sample_estimate_not_measured_full_pool=True,gzip_compress_p50_ms=240.0,gzip_decompress_p50_ms=19.0,disk_pool_GB_decimal=sum(s.sizes)/1e9)
        integ=dict(TEST627_LOG["integrity"])
        return dict(measured="synthetic self-test",measured_utc="synthetic",gpu="none",status="INTEGRITY_FAILURE"if s.mode=="engine_status"else"COMPLETE",summary=json.loads(json.dumps(e)),integrity=integ,storage=st,locked_reranker="PEAK50",locked_candidates=dict(nprobe=2,token_k=16,budget=16),
                    limitations=["synthetic limitation line"],elapsed_seconds=1.0,result_file=s.res.name,result_sha256=hashlib.sha256(s.res.read_bytes()).hexdigest(),
                    write=dict(files=N_CART,ms_p50=131.0,ms_p95=150.0,seconds_total=190.0),sizes=list(s.sizes),
                    index=dict(native_positions=26738,ivf_lists=128,pca_dims=128,index_bytes=13970459,index_build_ms=26.0,pca_basis_bytes=1835008,pca_center_bytes=14336,owner_bytes=106952,
                               native_activation_bytes=383315968,list_min=40,list_max=900,pca_train_vectors=2048),baseline_610=dict(H1=12,H5=12))
    def engine_result_path(s):return s.res
    def cases(s):return[dict(case=j,question=s.qs[j],target=t)for j,t in enumerate(s.case_t)]
    def _sel(s,j):return s.case_t[j]+1 if(s.mode=="wrong"or j%11==5)else s.case_t[j]
    def _cands(s,j):
        t=s.case_t[j];sel=s._sel(j);ids=[sel]+[x for x in([t]+[t+k for k in range(2,17)])if x!=sel][:15]
        rng=np.random.default_rng(j);cand=[]
        for r,ci in enumerate(ids):
            fr=0.62-0.01*r+float(rng.normal(0,.001));pk=0.80-0.012*r;cand.append(dict(id=ci,score=0.5*fr+0.5*pk,fram=fr,peak=pk))
        cand.sort(key=lambda c:(-c["score"],c["id"]));return cand
    def sealed_row(s,case):
        rk=[c["id"]for c in s._cands(case)];sel=rk[0];top5=rk[:5]
        if s.mode=="read_mismatch":top5=list(reversed(top5))
        return dict(selected=sel,top5=top5,selected_correct=sel==s.case_t[case],answer=s.cars[sel]["gold"],correct=sel==s.case_t[case])
    def car(s,ci):c=s.cars[ci];return dict(id=ci,fact=c["fact"],gold=c["gold"],body_len=c["body_len"],country=c["country"],file=f"cut3_{ci:05d}.pt",bytes=s.sizes[ci],sha256=hashlib.sha256(f"f{ci}".encode()).hexdigest())
    def source_ids(s,ci):return list(range(1,s.P+1))+[1000+ci*20+j for j in range(s.cars[ci]["body_len"])]
    def _ids(s,text,base):
        s.__dict__.setdefault("_tokmap",{});out=[]
        for i,t in enumerate(re.findall(r"\n|[^\s\n]+",text)):tid=base+(sum(map(ord,t))*31+i)%50000;s._tokmap[tid]=t;out.append(tid)
        return out
    def query_ids(s,q):return s._ids(f"\nQUESTION:\n{q}\n\nANSWER:\n<|im_end|>\n<|im_start|>assistant\n",2000)
    def frame_ids(s,q):return list(range(1,s.P+1))+s._ids(q+"\n\n",60000)
    def decode_each(s,ids):return[s._tokmap.get(t,"?")for t in ids]
    def panel_strings(s):return sorted({c["gold"]for c in s.cars}|{c["country"]for c in s.cars})
    def hit(s,a,g):return hit_ans(a,g)
    def build_ms(s,ci):return 130.0+ci%7
    def pool_stat(s):
        bad=[7]if s.mode=="pool_size"else[];return dict(files_in_pool=N_CART,manifest=N_CART,size_mismatch=bad,total_bytes=sum(s.sizes))
    def pool_sha_audit(s):return dict(files=N_CART,mismatch=[],seconds=0.01)
    def sentinel(s):return"a"*64
    def guard(s):s.calls+=1;return"c"*64 if(s.mode=="guard_changes"and s.calls>1)else"b"*64
    def full_weight_sha(s):return WEIGHT_SHA_EXPECTED
    def frozen(s):
        s.__dict__["fz"]=s.__dict__.get("fz",0)+1
        return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=0,foreign_hooks=0,foreign_modules={},sentinel="a"*64,residual_hooks=(["L3:layer"]if(s.mode=="residual_hook"and s.fz>1)else[]))
    def write(s,ci):
        src=s.source_ids(ci);f=[dict(kind="ids",n=len(src),ids=src,embeds_zero=None,has_past=False,past_len=None),dict(kind="embeds",n=len(src),ids=None,embeds_zero=True,has_past=False,past_len=None)]
        if s.mode=="write_extra":f.append(dict(kind="ids",n=3,ids=[1,2,3],embeds_zero=None,has_past=False,past_len=None))
        return _SynMem(ci,len(src),"diff"if s.mode=="write_diff"else""),f,0.13
    def read_file(s,ci,device=False):
        if s.mode=="sha_mismatch"and device:return None,dict(file=f"cut3_{ci:05d}.pt",sha256="0"*64,sha_ok=False,seconds=0.01)
        return _SynMem(ci,s.P+s.cars[ci]["body_len"]),dict(file=f"cut3_{ci:05d}.pt",sha256=s.car(ci)["sha256"],sha_ok=True,bytes=s.sizes[ci],seconds=0.018)
    def file_bytes(s,ci):
        b=bytes(range(256))*64
        return b+s.cars[ci]["fact"].encode()if s.mode=="fact_in_file"else b
    def address(s,q):
        j=s.qs.index(q);t=s.case_t[j];cand=[dict(c)for c in s._cands(j)];c626=sorted(c["id"]for c in cand)
        if s.mode=="peak_rederive":cand[3]["score"]+=0.01
        if s.mode=="cand_merge":c626=c626[:-1]+[N_CART-1]
        fid=s.frame_ids(q)
        if s.mode=="frame_has_fact":fid=fid+s.source_ids(t)[s.P:]
        return dict(ranking=[c["id"]for c in cand],candidate626=c626,candidates=cand,touched=117,faiss_ms=0.8,top2_ms=18.0,rerank_ms=2.5,address_ms=24.4,encode_ms=31.0,query_rows=11,
                    forwards=[dict(kind="ids",n=len(fid),ids=fid,embeds_zero=None,has_past=False,past_len=None)])
    def ask(s,mem,q):
        ids=s.query_ids(q)
        if s.mode=="source_in_query":ids=s.source_ids(mem.key)+ids
        j=s.qs.index(q)if q in s.qs else None
        ans=s.cars[mem.key]["gold"]if j is not None else"Not provided."
        rec=[dict(kind="ids",n=len(ids),ids=ids,embeds_zero=None,has_past=True,past_len=mem.T)]+[dict(kind="ids",n=1,ids=[7],embeds_zero=None,has_past=True,past_len=mem.T+len(ids)+i)for i in range(3)]
        return ans,rec,0.24
    def compare(s,a,b):
        eq=a.key==b.key and a.T==b.T and a.salt==b.salt
        rows=[dict(layer=li,tensor=nm,shape=[1,ARCH[3],a.T,ARCH[4]],shape_equal=True,dtype_equal=True,bitwise_equal=eq or li<4,max_abs_diff=0.0 if(eq or li<4)else 0.0078125,
                   differing_elements=0 if(eq or li<4)else 11,elements=ARCH[3]*a.T*ARCH[4])for li in range(ARCH[0])for nm in("K","V")]
        return dict(layers_a=ARCH[0],layers_b=ARCH[0],tensors_compared=len(rows),all_equal=all(r["bitwise_equal"]for r in rows),layers_differing=sorted({r["layer"]for r in rows if not r["bitwise_equal"]}),
                    max_abs_diff=max(r["max_abs_diff"]for r in rows),rows=rows)
    def equal(s,a,b):return s.compare(a,b)["all_equal"]
    def memory_len(s,mem):return mem.T
    def memory_sha(s,mem):return hashlib.sha256(f"m{mem.key}{mem.T}{mem.salt}".encode()).hexdigest()
    def memory_bytes(s,mem):return mem.T*ARCH[0]*2*ARCH[3]*ARCH[4]*2
    def memory_shape(s,mem):return dict(layers=ARCH[0],tensors=2*ARCH[0],shape=[1,ARCH[3],mem.T,ARCH[4]],dtype="bfloat16",T=mem.T,uniform=True)
    def release(s,*m):pass
    def sync(s):pass
    def reset_peak(s):pass
    def peak_gib(s):return 0.0
def drain(gen):
    while True:
        try:next(gen)
        except StopIteration as s_:return s_.value
RUN_COUNTER=globals().get("RUN_COUNTER",0);GPU_LOCK=threading.Lock();SESSION_LEDGER=globals().get("SESSION_LEDGER",[])
LEDGER_PATH=ROOT/"session_ledger.jsonl"
SKIP=getattr(gr,"skip",None)or gr.update
_K=object()
FILE_KEYS=["zip","json","txt","man","engine","tr","ledger"]
DL_LABELS=["⬇ EVIDENCE PACKAGE (.zip)","⬇ FULL RUN LOG (.json)","⬇ READABLE RUN LOG (.txt)","⬇ RUN MANIFEST (.json)","⬇ ENGINE START-UP RESULT (.json)","⬇ Fig. 1 · TÜRKÇE (.jpg)","⬇ SESSION LEDGER (.jsonl)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {N_FIGS} FIGURES (JPEG, 300 dpi)"
RAW_KEYS=["txt","json","man","ledger"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)+1
_GR_MAJOR=int(re.match(r"\d+",gr.__version__).group())
try:
    from gradio.routes import API_PREFIX as _GR_API_PREFIX
except Exception:
    _GR_API_PREFIX="/gradio_api"if _GR_MAJOR>=5 else""
FILE_URL_PREFIX=f"{_GR_API_PREFIX}/file="
def dl_update(path,label):
    if path is None:
        try:return gr.DownloadButton(label=label,value=None,interactive=False)
        except Exception:return gr.update(value=None,interactive=False)
    try:return gr.DownloadButton(label=label,value=str(path),interactive=True)
    except Exception:return gr.update(value=str(path),interactive=True)
def btn_update(active):
    try:return gr.Button(value=DL_ALL_LABEL,interactive=bool(active))
    except Exception:return gr.update(value=DL_ALL_LABEL,interactive=bool(active))
def jpg_urls_json(paths):return json.dumps([{"url":FILE_URL_PREFIX+quote(str(Path(p)),safe="/"),"name":Path(p).name}for p in paths],ensure_ascii=False)
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K,nxt=_K):
    out=[SKIP()if status is _K else status,SKIP()if gal is _K else gal]
    if jpgs is _K:out+=[SKIP(),SKIP()]
    elif jpgs is None:out+=[btn_update(False),""]
    else:
        for pth in jpgs:
            if not file_ready(pth):raise RuntimeError(f"figure missing or empty: {pth}")
        out+=[btn_update(True),jpg_urls_json(jpgs)]
    if files is _K:out+=[SKIP()for _ in range(len(FILE_KEYS)*2)]
    elif files is None:out+=[dl_update(None,l)for l in DL_LABELS]+[None]*len(FILE_KEYS)
    else:
        paths=[files.get(k_)for k_ in FILE_KEYS]
        for pth in paths:
            if pth is not None and not file_ready(pth):raise RuntimeError(f"download artifact missing or empty: {pth}")
        out+=[dl_update(p,l)for p,l in zip(paths,DL_LABELS)]+[None if p is None else str(p)for p in paths]
    if raw is _K:out+=[SKIP()for _ in RAW_KEYS]
    elif raw is None:out+=[""]*len(RAW_KEYS)
    else:out+=[raw.get(k_,"")for k_ in RAW_KEYS]
    out.append(SKIP()if nxt is _K else nxt)
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total):
    pc_=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e5e7eb;height:10px"><div style="width:{pc_:.1f}%;height:10px;background:#1d4ed8"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",html.escape(e["body"])+pbar_html(e["stage"],N_STAGES),"info")
def ledger_text():return"\n".join(json.dumps(x,ensure_ascii=False)for x in SESSION_LEDGER)
def next_case():return RUN_COUNTER%N_CASES
def next_case_html(I=None):
    I=I or INSTR;c=next_case();row=I.cases()[c];t=row["target"];C=I.car(t)
    body=(f'<span class="mono">question</span> {html.escape(row["question"])}<br>'
          f'<span class="mono">WRITE</span> fact of this case (re-written live, compared with its start-up file): {html.escape(C["fact"])} → {html.escape(C["file"])}<br>'
          f'<span class="mono">READ</span> the question alone selects one of {N_CART:,} files; the fact is not given to the model<br>'
          f'<span class="mono">rotation</span> predeclared: RUN number n uses case (n − 1) mod {N_CASES} of the engine\'s {N_CASES} new-wording questions — no choice, no cherry-picking')
    return card_html(f"Next RUN · case {c:02d} of {N_CASES}",body,"info")
def engine_card_html(I):
    inf=I.info;cfg=I.config();su=I.startup();S=su["summary"];st=su["storage"];ix=su["index"];T7=TEST627_LOG
    rows=[("model",f"{inf['model_id']} · frozen · {inf['dtype']} · {arch_txt(tuple(cfg['arch']))}"),("GPU",inf["gpu"]),
          ("engine",f"TEST627 demo engine (verbatim) · SHA-256 {inf['engine_sha256'][:16]}… · chain 626.py {cfg['chain_sha256']['626.py'][:12]}… · 624.py {cfg['chain_sha256']['624.py'][:12]}… · 610.py {cfg['chain_sha256']['610.py'][:12]}…"),
          ("WRITE at start-up",f"{su['write']['files']:,} CUT3 files written by the frozen model · p50 {su['write']['ms_p50']:.0f} ms per file"),
          ("SSD",f"{st['cartridges']:,} independent files · {st['disk_pool_bytes']:,} bytes · avg {st['avg_file_bytes']/1e6:.2f} MB · SHA-256 manifest verified: {su['integrity'].get('manifest_1453_sha_verified')}"),
          ("RAM address",f"PCA{ix['pca_dims']} + FAISS IVF{ix['ivf_lists']} over {ix['native_positions']:,} positions ({ix['index_bytes']/1e6:.1f} MB) + native layer-3 vectors ({ix['native_activation_bytes']/1e6:.0f} MB)"),
          ("READ (locked)",f"nprobe {cfg['candidates'][0]} · token_k {cfg['candidates'][1]} · {cfg['candidates'][2]} candidates · {cfg['reranker'][0]}"),
          (f"start-up acceptance (this runtime, finished {str(su.get('measured_utc',''))[:19]} UTC)",f"original wording {S['ORIGINAL']['answer_correct']}/48 · new wording {S['ALTERNATE']['answer_correct']}/48 answered from the disk files · errors {S['ORIGINAL']['errors']+S['ALTERNATE']['errors']} · integrity {sum(bool(v)for v in su['integrity'].values())}/{len(su['integrity'])}"),
          ("weights",f"full-weight SHA-256 {inf['weight_sha0'][:16]}… (= reference) · sentinel {inf['sentinel0'][:16]}… · guard {inf['guard0'][:16]}…"),
          ("init",f"{inf['init_seconds']:.0f} s" if isinstance(inf['init_seconds'],float)and math.isfinite(inf['init_seconds'])else"reused")]
    body="".join(f'<span class="mono">{html.escape(a)}</span> {html.escape(str(b))}<br>'for a,b in rows)
    body+=(f'<div class="small" style="margin-top:8px"><b>SEALED TEST627 LOG</b> (627.log; not recomputed) · {T7["decision"]} · original {T7["e2e"]["ORIGINAL"]["answer_correct"]}/48 · new wording {T7["e2e"]["ALTERNATE"]["answer_correct"]}/48 · '
           f'log SHA-256 {T7["log_sha256"][:16]}…<br><b>How PEAK50 was locked</b> (626.log): TEST626\'s own pre-declared rule chose {TEST626_LOG["locked_by_own_rule"]} ({TEST626_LOG["e2e"]}); PEAK50 was fixed afterwards — the 96 questions are therefore not a blind test.</div>')
    return card_html("Engine status · start-up in this runtime · sealed references",body,"on")
HOW_HTML=card_html("What one RUN shows",
    "<b>A · WRITE</b> — the frozen model reads one fact and keeps its own key/value state (CUT3): 28 layers × (K, V). The engine wrote 1,453 such files at start-up; RUN writes the case's fact again and checks the result is bit-for-bit the same file.<br>"
    "<b>B · SSD</b> — every cartridge is a separate file on disk with a SHA-256 fingerprint; RUN checks all 1,453 files are present with the recorded sizes.<br>"
    "<b>C · READ</b> — only the question goes to the model. Its internal vectors are looked up in a geometric index in RAM (PCA128 + FAISS IVF128); 16 candidates are re-scored by PEAK50 and one file is chosen.<br>"
    "<b>D · ANSWER</b> — that file is loaded and the frozen model answers from it. The fact text is not in the prompt. A control repeats the question over the runner-up file.<br>"
    "<b>E · RESULTS</b> — four figures (JPEG 300 dpi + PDF), a sealed log and a ZIP; start-up acceptance, the sealed TEST627 log and the limits are shown together.","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler():
    global RUN_COUNTER
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        c=next_case();RUN_COUNTER+=1;run_id=f"{RUN_PREFIX}{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}-C{c:02d}"
        prune_runs(4);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        preview=[dict(case=x["case"],correct=x["correct"])for x in SESSION_LEDGER]
        gen=execute_run(INSTR,dict(run_id=run_id,run_dir=run_dir,case=c,session=preview+[dict(case=c,correct=None)]));first=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first else pack(stage_card(e));first=False
        SESSION_LEDGER.append(B["ledger"]);LEDGER_PATH.write_text(ledger_text()+"\n",encoding="utf-8")
        P=B["P"];L=P["live"];ok=L["correct"];Wr=L["write"];Rd=L["read"];Ct=L["control"];su=P["startup"];S=su["summary"]
        raw={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8"),"ledger":ledger_text()}
        ctl=("not applicable"if Ct.get("skipped")else f'#{Ct["id"]:04d} → {html.escape(str(Ct["answer"]))} · expected answer produced: {"yes" if Ct["gold_produced"] else "no"}')
        body=(f"<b>{html.escape(B['run_id'])}</b> · LIVE DEMONSTRATION RUN · case {L['case']:02d}<br>"
              f'<span class="mono">A · WRITE</span> {html.escape(L["fact"])} → {Wr["T"]} rows × {Wr["layers"]} layers · re-written now, bitwise identical to {html.escape(Wr["file"])}: <b>{Wr["bit_equal"]}</b> · fact text in file: {Wr["fact_in_file"]}<br>'
              f'<span class="mono">B · SSD</span> {L["ssd"]["files_in_pool"]:,} independent files · {L["ssd"]["total_bytes"]:,} bytes · sizes = manifest: {not L["ssd"]["size_mismatch"]}<br>'
              f'<span class="mono">C · READ</span> question “{html.escape(L["question"])}” → touched {Rd["touched"]} of {N_CART:,} → {len(Rd["ranking"])} candidates → PEAK50 selects #{Rd["selected"]:04d} ({html.escape(Rd["file"])}) · {Rd["address_ms"]:.1f} ms<br>'
              f'<span class="mono">D · ANSWER</span> <b>{html.escape(L["answer"])}</b> · expected {html.escape(L["expected"])} · <b>{"CORRECT" if ok else "INCORRECT"}</b> · model input: {Rd["query_tokens"]} question tokens over the loaded {Rd["T"]}-row cartridge · load {Rd["load_ms"]:.1f} ms · answer {Rd["answer_ms"]:.0f} ms<br>'
              f'<span class="mono">control</span> runner-up file {ctl}<br>'
              f'<span class="mono">this RUN</span> one new run, started {html.escape(P["run_start_utc"][:19])} UTC — the result above is this run only<br>'
              f'<span class="mono">separately</span> engine start-up acceptance in this runtime (finished {html.escape(str(su.get("measured_utc",""))[:19])} UTC): original {S["ORIGINAL"]["answer_correct"]}/48 · new wording {S["ALTERNATE"]["answer_correct"]}/48 · archived 627.log record: 48/48 · 48/48 — not a blind test; see limits<br>'
              f"weights unchanged (sentinel + all-parameter guard) · verdict <b>{html.escape(B['verdict'])}</b> · {N_FIGS} figures + PDFs + Turkish Fig. 1 + ZIP · {B['checks']}/{B['checks']} checks PASS<br>"
              f'<span class="mono">payload SHA-256 {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Sealed",body,"on"),[(str(p),c_)for p,c_ in B["imgs"]],
                   {"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"],"engine":B["engine"],"tr":B["tr_jpg"],"ledger":LEDGER_PATH},[p for p,_ in B["imgs"]],raw,next_case_html())
    except Exception as ex:
        print("="*140);print(f"RUN FAILED — stage: {stage_name}");print(f"{type(ex).__name__}: {ex}");traceback.print_exc();print("="*140)
        kind="AUDIT FAIL — nothing was sealed. "if isinstance(ex,AuditFail)else"";partial=getattr(ex,"partial",None)
        if partial:kind+="The raw outputs gathered so far were preserved (FULL RUN LOG). "
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run PARTS 1–3 again.</b>"if is_cuda_error(ex)else"<br>The full traceback is printed in the Colab console."
        card=card_html("Run failed",f"{html.escape(kind)}Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err")
        if partial:yield pack(card,[],{"json":partial},None,{"txt":traceback.format_exc(),"json":Path(partial).read_text(encoding="utf-8"),"man":"","ledger":ledger_text()},next_case_html())
        else:yield pack(card,[],None,None,None,next_case_html())
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
def verify_handler():
    if not GPU_LOCK.acquire(blocking=False):return"Busy: a run is in progress."
    try:
        t=time.perf_counter();g=INSTR.guard();fz=INSTR.frozen();w=INSTR.full_weight_sha();cfg=INSTR.config();pa=INSTR.pool_sha_audit()
        out=dict(model=MODEL_ID,full_weight_sha256_now=w,full_weight_sha256_reference=WEIGHT_SHA_EXPECTED,full_weight_identical_to_reference=w==WEIGHT_SHA_EXPECTED,
                 all_parameter_guard_now=g,all_parameter_guard_startup=INSTR.info["guard0"],guard_unchanged=g==INSTR.info["guard0"],sentinel_unchanged=fz["sentinel"]==INSTR.info["sentinel0"],
                 trainable_tensors=fz["trainable_tensors"],foreign_hooks=fz["foreign_hooks"],residual_hooks=fz["residual_hooks"],
                 ssd_pool_sha256_audit=dict(files=pa["files"],mismatches=len(pa["mismatch"]),mismatch_ids=pa["mismatch"][:20],seconds=round(pa["seconds"],2)),
                 engine_sha256=cfg["engine_sha256"],engine_sha256_expected=ENGINE_SHA256_EXPECTED,chain_sha256=cfg["chain_sha256"],source_blobs=cfg["source_blobs"],
                 locked_read=dict(candidates=cfg["candidates"],reranker=cfg["reranker"]),architecture=cfg["arch"],checked_utc=utc_now(),seconds=round(time.perf_counter()-t,3),
                 note="SHA-256 values show that weights and files are unchanged; they are not scientific proof.")
        return json.dumps(jsafe(out),indent=2,ensure_ascii=False)
    finally:GPU_LOCK.release()
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        for mode,c_ in(("ok",0),("ok",5),("wrong",12)):
            dd=d/f"{mode}{c_}";dd.mkdir()
            B=drain(execute_run(SelfTestInstrument(mode,d/"eng"),dict(run_id=f"SELFTEST-{mode.upper()}-C{c_:02d}",run_dir=dd,case=c_,session=[dict(case=c_,correct=None)])))
            assert len(B["imgs"])==N_FIGS and len(B["pdfs"])==N_FIGS and file_ready(B["zip"])and file_ready(B["tr_jpg"])and B["verdict"].endswith("SOURCE_FREE_ANSWER")and B["verdict"].startswith("SELFTEST_SYNTHETIC_")
            assert("_INCORRECT_"in B["verdict"])==(mode=="wrong"or c_==5)
        out.append(f"full pipeline on a synthetic instrument (correct answer, selection miss shown as INCORRECT, forced miss): {N_FIGS}/{N_FIGS} figures + Turkish Fig. 1 (JPEG 300 dpi + PDF), ZIP, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("write_diff","a WRITE result that is not bitwise identical to its file"),("write_extra","an extra model forward during WRITE"),
                         ("fact_in_file","fact text inside a cartridge file"),("pool_size","a pool file whose size differs from the manifest"),("frame_has_fact","fact tokens in the question frame"),
                         ("peak_rederive","a PEAK50 score that does not re-derive"),("cand_merge","a ranked set that differs from the candidate626 set"),
                         ("read_mismatch","a live READ that does not reproduce the engine's start-up record"),("sha_mismatch","a selected file whose SHA-256 differs from the manifest"),
                         ("source_in_query","fact tokens in the answering input"),("residual_hook","a hook left on a decoder layer after the run"),("engine_status","an engine start-up status other than COMPLETE")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestInstrument(mode,d/"eng"),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2,case=3,session=[])))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (nothing sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
    finally:QUIET=False
    if len(pack(HOW_HTML))!=N_OUTPUTS or len(pack(HOW_HTML,[],None,None,None,""))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
    out.append(f"callback output count = {N_OUTPUTS}");out.append(f"file route {FILE_URL_PREFIX}");shutil.rmtree(d,ignore_errors=True);return out
say("="*140);say(DEMO_TITLE+" — "+MODEL_FAMILY+" · PART 3 / 3 · SELF-TEST AND INTERFACE");say("[1/3] SERVICE SELF-TEST")
for c_ in service_selftest():say(" PASS ·",c_)
#<<UI_END>>
# ---------------- interface ----------------
say("[2/3] INTERFACE")
CSS="""
:root{--kz-on:#0f766e;--kz-fg:#111827;--kz-card:#ffffff;--kz-bd:#d1d5db;--kz-a:#1d4ed8;--kz-err:#b91c1c;--kz-warn:#b45309;--kz-hero:#0b1220;--kz-gold:#b45309}
.dark{--kz-on:#2dd4bf;--kz-fg:#f3f4f6;--kz-card:#111827;--kz-bd:#4b5563;--kz-a:#60a5fa;--kz-err:#f87171;--kz-warn:#fbbf24;--kz-hero:#0b1220;--kz-gold:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:920px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{background:#0b1220;color:#f9fafb!important;border-radius:6px;padding:18px 18px 14px;margin:4px 0 8px}
.hero *{color:#f9fafb!important}
.rec{font-size:11.5px;letter-spacing:1.6px;text-transform:uppercase;opacity:.85;border:1px solid rgba(255,255,255,.35);display:inline-block;padding:2px 8px;border-radius:3px}
.brand{font-size:clamp(24px,6.4vw,34px);font-weight:800;letter-spacing:.4px;line-height:1.1;margin-top:10px}
.title{font-size:clamp(15px,4vw,19px);font-weight:600;margin-top:4px}
.para{font-size:clamp(14px,3.8vw,17px);margin-top:10px;color:#fde68a!important;font-weight:600}
.q{font-size:13.5px;margin-top:8px;font-style:italic;opacity:.95}
.by{font-size:14px;margin-top:10px}
.by b{color:#fde68a!important}
.lic{font-size:12px;opacity:.85;margin-top:6px}
.lic a{text-decoration:underline}
.msg{margin:6px 0 0;padding-left:18px;font-size:13.5px}
.card{background:var(--kz-card);border:1px solid var(--kz-bd);border-left:4px solid var(--kz-bd);border-radius:4px;padding:10px 12px;margin:6px 0}
.card.on{border-left-color:var(--kz-on)}.card.err{border-left-color:var(--kz-err)}.card.warn{border-left-color:var(--kz-warn)}.card.info{border-left-color:var(--kz-a)}
.card .h{font-weight:700;font-size:15px;margin-bottom:4px}
.small{font-size:13.5px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;opacity:.8;margin-right:4px}
#kz_run button,#kz_run{font-size:clamp(15px,4.4vw,18px)!important;font-weight:700!important;min-height:54px!important}
"""
DL_ALL_JS="""(u)=>{let items=[];try{items=JSON.parse(u||"[]")}catch(e){items=[]}
let i=0;const next=()=>{if(i>=items.length)return;const it=items[i++];const a=document.createElement('a');a.href=it.url;a.download=it.name;a.rel='noopener';
document.body.appendChild(a);a.click();setTimeout(()=>{a.remove();next()},900)};next();return u;}"""
HERO=(f'<div class="kz hero"><span class="rec">Public technical demonstration · priority record</span>'
      f'<div class="brand">AKBASCORE MAM</div><div class="title">The Silicon Hippocampus · {html.escape(DEMO_SUB)} · {html.escape(MODEL_FAMILY)}</div>'
      f'<div class="para">{html.escape(PARADIGM)}.</div><div class="q">{html.escape(SCI_QUESTION)}</div>'
      f'<div class="by"><b>{html.escape(DISCOVERY)}.</b><br>{html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(AUTHOR_PLACE)}</div>'
      '<ul class="msg">'+"".join(f"<li>{html.escape(m)}</li>"for m in CORE_MESSAGE)+"</ul>"
      f'<div class="lic">{html.escape(NAME_NOTE)}</div><div class="lic">{html.escape(PRIORITY)}</div>'
      f'<div class="lic">{html.escape(LICENSE_NAME)} · {html.escape(COPYRIGHT)} · <a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a> · '
      f'<a href="{URL_627}" target="_blank" rel="noopener">627.py</a> · <a href="{URL_627LOG}" target="_blank" rel="noopener">TEST627 log</a> · <a href="{URL_626LOG}" target="_blank" rel="noopener">TEST626 log</a> · '
      f'<a href="{URL_624}" target="_blank" rel="noopener">624.py</a> · <a href="{URL_610}" target="_blank" rel="noopener">610.py</a> · '
      f'DOI <a href="{PRIOR_PUB["url"]}" target="_blank" rel="noopener">{html.escape(PRIOR_PUB["doi"])}</a> · DOI <a href="{QWEN_PUB["url"]}" target="_blank" rel="noopener">{html.escape(QWEN_PUB["doi"])}</a> · '
      f'DOI <a href="{V12_PUB["url"]}" target="_blank" rel="noopener">{html.escape(V12_PUB["doi"])}</a></div></div>')
SCOPE_HTML=card_html("Scope of this demonstration",
    "<b>Demonstrated live in every RUN:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["demonstrated"])+"<br><br><b>Produced at start-up in this runtime:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["startup"])
    +"<br><br><b>Archived (sealed logs):</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["archived"])
    +"<br><br><b>Not established:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["not_established"])+"<br><br><b>Technical notes:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["notes"]),"info")
READY_HTML=card_html("Ready",f"Press <b>RUN</b>. Nothing to type or choose: each RUN takes the next predeclared case, re-runs WRITE for its fact, checks the SSD pool, READs the answer from the question alone, runs the control, verifies the frozen weights, seals the payload and renders {N_FIGS} figures (JPEG 300 dpi + vector PDF, Fig. 1 also in Turkish). About 20–40 seconds.","info")
def _tb(**kw):
    for extra in(([{"buttons":["copy"]}]if _GR_MAJOR>=6 else[])+[{"show_copy_button":True},{}]):
        try:return gr.Textbox(**extra,**kw)
        except Exception:pass
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)
if"demo"in globals():
    try:demo.close()
    except Exception:pass
_TITLE="AKBASCORE MAM · The Silicon Hippocampus · Qwen2.5-7B · Mustafa Akbaş"
if _GR_MAJOR>=6:_blocks=gr.Blocks(title=_TITLE)
else:
    try:_blocks=gr.Blocks(css=CSS,title=_TITLE)
    except TypeError:_blocks=gr.Blocks(title=_TITLE)
with _blocks as demo:
    gr.HTML(HERO);gr.HTML(HOW_HTML);gr.HTML(engine_card_html(INSTR))
    nxt=gr.HTML(next_case_html())
    run_btn=gr.Button("RUN · WRITE → SSD → READ → answer without the source text → control → seal",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_FIGS} figures (tap to open · JPEG 300 dpi; vector PDFs and the Turkish Fig. 1 in the ZIP)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · figures, PDFs, payload, logs, engine result","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)","ENGINE START-UP RESULT (.json)","Fig. 1 · Türkçe (.jpg)","SESSION LEDGER (.jsonl)")]
    with gr.Tabs():
        with gr.Tab("READABLE RUN LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
        with gr.Tab("SESSION LEDGER"):raw_led=_tb(lines=10,max_lines=30,label="one line per sealed run in this session")
    with gr.Accordion("Verification · full-weight SHA-256 · weight guard · SHA-256 of all 1,453 cartridge files (about 10–30 s)",open=False):
        ver_btn=gr.Button("Run verification now")
        ver_out=_tb(lines=14,max_lines=40,label="verification record")
    gr.HTML(SCOPE_HTML)
    gr.HTML(f'<div class="kz small" style="opacity:.8;margin:8px 0 18px">{html.escape(DISCOVERY)} · {html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(LICENSE_NAME)} · '
            f'<a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a></div>')
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man,raw_led,nxt]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    run_btn.click(run_handler,inputs=None,outputs=OUTS)
    ver_btn.click(verify_handler,inputs=None,outputs=ver_out)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
say("[3/3] LAUNCH (public share link) — open the printed gradio.live link and press RUN.")
_LAUNCH=dict(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
try:demo.launch(**_LAUNCH)
except TypeError as _ex:
    if"css"not in str(_ex):raise
    _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)
