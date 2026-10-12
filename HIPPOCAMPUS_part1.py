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
