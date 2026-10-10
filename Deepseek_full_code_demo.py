# =====================================================================================================================================
# AKBASCORE MAM · PERSISTENT NUMERICAL MEMORY FOR FROZEN LANGUAGE MODELS — DEEPSEEK-V2-LITE-CHAT · PART 1 / 3 · ENGINE
# Third model family (multi-head latent attention) · Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 10 October 2026
# Run PART 1 → PART 2 → PART 3 as three consecutive Google Colab cells in the SAME runtime (GPU: A100 40 GB; about 31 GB of weights).
# PART 1 embeds 596.py (the working DeepSeek MLA CUT3 engine and its own demo backend) VERBATIM, verifies its git blob SHA-1, executes it
# in its own module namespace, loads frozen DeepSeek-V2-Lite-Chat at the pinned revision and attaches read-only measurement.
# PART 2 adds the fail-closed measurement pipeline and figures. PART 3 starts the Gradio demonstration.
# AKBASCORE RESEARCH SOFTWARE LICENSE · Copyright © 2026 Mustafa Akbaş · https://github.com/ceceli33/titan-cognitive-core-v2
#
# ENGINE         : 596.py, embedded byte-for-byte as ENGINE_SRC and executed unchanged. It verifies its own sources (560.py panel facts,
#                  590.py CUT3 MLA functions) by git blob, builds the locked 24-case panel and exposes demo_load_case / demo_ask /
#                  demo_verify_all. This program calls only that API; the memory algorithm is not modified, recalibrated or re-implemented.
# MECHANISM      : each source fact is read once by the frozen model and stored as an independent numerical cartridge
#                  (layer-3 residual H3 + layers 0–3 MLA cache: latent c_KV 512 + rotary key 64). Cartridges are appended one by one:
#                  layers 4–26 are computed only for the new tokens, attending to the existing memory; the rotary part of layers 0–3 is
#                  re-phased to the new positions. Questions are answered from the numerical memory; no source text at query time.
# INSTRUMENTATION: (1) a recording-only forward pre-hook, attached only while an engine API call runs; (2) a write/append observer,
#                  attached only during load_case(): it temporarily wraps the engine's own write_cartridge / init_cartridge /
#                  append_cartridge in the engine namespace, calls them unchanged and compares every previously consolidated cache row
#                  bitwise; (3) an all-parameter weight guard plus the engine's own weight fingerprint; (4) re-derivation of every value.
# =====================================================================================================================================
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,gc,types
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
DEMO_TITLE="AKBASCORE MAM · Persistent numerical memory for frozen language models"
DEMO_SHORT="AKBASCORE MAM · DeepSeek-V2-Lite-Chat incremental MLA DC3 demonstration"
MODEL_FAMILY="DeepSeek-V2-Lite-Chat"
AUTHOR="Mustafa Akbaş";ORG="AkbasCore AI Teknoloji";DATE_TXT="10 October 2026";DATE_ISO="2026-10-10";AUTHOR_PLACE="Mersin, Türkiye"
COPYRIGHT="Copyright © 2026 Mustafa Akbaş";LICENSE_NAME="AKBASCORE RESEARCH SOFTWARE LICENSE"
LICENSE_URL="https://github.com/ceceli33/titan-cognitive-core-v2"
REPO_RAW="https://github.com/ceceli33/titan-cognitive-core-v2/blob/main/"
SOURCE_URL=REPO_RAW+"596.py";LOCK_URL=REPO_RAW+"597.py";LOG_URL=REPO_RAW+"597.log";SCALE_LOG_URL=REPO_RAW+"590.py"
PARADIGM="A new numerical-memory paradigm for frozen language models"
DISCOVERY="Discovered and developed by Mustafa Akbaş"
# ---- earlier records (cited as references only; their values are NOT DeepSeek results) ----
PRIOR_PUBS=[dict(short="Mistral-7B (DC6)",title="AKBASCORE MAM v1.0 — Source-Free Persistent Memory for Frozen LLMs via DC6 Consolidation (Mistral-7B)",
                 doi="10.5281/zenodo.23245358",url="https://doi.org/10.5281/zenodo.23245358",date="8 October 2026"),
            dict(short="Qwen2.5-7B (DC3)",title="AKBASCORE MAM v1.1 — Cross-Model Replication: Source-Free Persistent Memory for Frozen LLMs on Qwen2.5-7B (DC3)",
                 doi="10.5281/zenodo.23257347",url="https://doi.org/10.5281/zenodo.23257347",date="9 October 2026")]
PRIOR_PUB=PRIOR_PUBS[0]
CROSS_TITLE="Cross-Model Replication — Third Model Family: DeepSeek-V2-Lite-Chat (MLA, DC3)"
CROSS_TEXT=("After Mistral-7B (DC6) and Qwen2.5-7B (DC3), the AKBASCORE MAM architecture has been implemented on DeepSeek-V2-Lite-Chat, "
            "a model family with multi-head latent attention, using a model-specific consolidation boundary and the same write → init → append → ask cycle.")
CROSS_RESULT="On the shared 72-question panel the DeepSeek result is 69/72; the three failed cases are reported in full."
CROSS_BOUNDARY=("Three model families are experimental evidence that the architecture transfers across models; they are not evidence that it works in every language model. "
                "Earlier records are cited as references only; every number in this demonstration comes from DeepSeek-V2-Lite-Chat.")
CROSS_REF=" · ".join(f"{p['title']} · DOI {p['doi']} · {p['url']}" for p in PRIOR_PUBS)
CROSS_ALLOWED=(CROSS_TEXT,CROSS_REF)+tuple(p["title"]for p in PRIOR_PUBS)+tuple(p["short"]for p in PRIOR_PUBS)
PRIORITY=("Public technical demonstration of the AKBASCORE MAM architecture on a third transformer model family. "
          "Mustafa Akbaş (AkbasCore AI Teknoloji) discovered and developed this numerical-memory mechanism (first public record: DOI "
          f"{PRIOR_PUBS[0]['doi']}, {PRIOR_PUBS[0]['date']}; second model family: DOI {PRIOR_PUBS[1]['doi']}, {PRIOR_PUBS[1]['date']}); "
          "this DeepSeek-V2-Lite-Chat record is dated 10 October 2026.")
CORE_MESSAGE=["Independent numerical cartridges are written by the frozen model — each source fact is read once.",
              "Cartridges are consolidated append-only into one shared numerical memory.",
              "No earlier source text is replayed during consolidation.",
              "Previously consolidated memory tokens are not recomputed.",
              "Model weights are not changed.",
              "The question is answered without giving the source text to the model again."]
ENGINE_FILE="596.py"
ENGINE_SHA256_EXPECTED="f0c02654eddbf04756b9a33ee2b6433a5523c91dec00b3253ab415648f06e85f"
ENGINE_BLOB_EXPECTED="85067b4f1cd0a1a941c0b88752f813360600c6b1"
ENGINE_SOURCES_EXPECTED={"560.py":"35d95a929a4e043a9af346c67fc103da47a792b5","590.py":"09ae7a6862956128465efc7e76cdca1e478a1a52"}
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
WEIGHT_SHA_EXPECTED=None
MODEL_ID="deepseek-ai/DeepSeek-V2-Lite-Chat";MODEL_REVISION="85864749cd611b4353ce1decdb286193298f64c7"
ARCH=(27,2048,512,64);CUT_EXPECTED=3;PREFIX_EXPECTED=28;N_CASES=24;N_CART=5;POSITIONS=("FIRST","MIDDLE","LAST")
TARGET_SLOT={"FIRST":0,"MIDDLE":2,"LAST":4}
CANON_GPU="NVIDIA A100-SXM4-40GB"
CACHE_SHORT="MLA cache (latent 512 + rotary 64)"
MEM_LABELS=("‖latent c_KV‖ per row (512-d)","‖rotary key k_R‖ per row (64-d)")
def arch_txt(a=ARCH):return f"{a[0]} layers · hidden {a[1]} · multi-head latent attention, cache per token and layer: latent {a[2]} + rotary {a[3]}"
def arch_short(a=ARCH):return f"{a[0]} L · {a[1]} · MLA {a[2]}+{a[3]}"
def upper_txt(cut=CUT_EXPECTED,nl=ARCH[0]):return f"layers {cut+1}–{nl-1}"
def correct_rule(ans,gold):return norm_ans(ans)==norm_ans(gold)
CORRECT_RULE_TXT="exact normalized name match (the 596.py scoring rule: casefold, punctuation removed, whole answer = gold name)"
# ---- reference records (archived Qwen-free, Mistral-free DeepSeek records; never produced or recomputed by this program) ----
TEST596=dict(name="TEST596",engine_file=ENGINE_FILE,engine_blob=ENGINE_BLOB_EXPECTED,model=MODEL_ID,revision=MODEL_REVISION,panel_sha=EXPECTED_PANEL_SHA,
    gpu=CANON_GPU,torch="2.11.0+cu130",transformers="5.18.0",cut=3,prefix=28,cases=24,positions=3,total=72,arm="INCR_DC3_MLA",
    score={"FIRST":24,"MIDDLE":23,"LAST":22,"TOTAL":69},
    failures=[(11,"LAST","Mirev","Kareth"),(14,"LAST","Corven","Marvek"),(23,"MIDDLE","Karev","Luneth")],
    source="reference record stored in 596.py (REFERENCE['TEST596']); the three failures were reproduced in TEST597 (597.log)")
TEST597=dict(name="TEST597",log_sha256="1ea5991a5230b2172e456f578ad264e36840d86a59860b753c06814770a8de37",program_blob="87b1642008505a50d5ef8659257de1b375367186",
    seconds=147.03,failing_slots=3,joint=(3,3),incr_on_failing=(0,3),incr_nine_slots=(6,9),native_indep=(0,3),
    failures_reproduced=True,nine_replay_match=True,readout_fix_verified=False,
    indep_answers=[(11,"LAST","The current capital of Sorvik is Mirev."),(14,"LAST","Marvek"),(23,"MIDDLE","The current capital of Talrek is Laven.")],
    decision="NO_VERIFIED_READOUT_FIX_DEMO_WITH_69_72_DISCLOSURE")
FAILED={(c,p):(g,a)for c,p,g,a in TEST596["failures"]}
def reference_outcome(case,position):
    """TEST596 reference outcome of one case/position: the recorded wrong answer for the three failures, otherwise correct (answer = gold)."""
    f=FAILED.get((int(case),str(position)))
    return dict(correct=f is None,answer=None if f is None else f[1])
ARM_ORDER=("INCR_DC3_MLA",)
ARM_DESC={"INCR_DC3_MLA":"append-only incremental consolidation at layer 3, multi-head latent attention cache (this engine; the arm run live)",
          "JOINT":"facts processed together as text (upper bound, not a memory) — TEST597, three failed slots only",
          "NATIVE_INDEP_MLA":"independently written native MLA cache, rotary re-phased, no consolidation — TEST597, three failed slots only"}
SCOPE={
 "demonstrated":[
  "Five source facts are written independently by frozen DeepSeek-V2-Lite-Chat (pinned revision): each fact is read once and stored as a numerical cartridge (layer-3 residual H3 + layers 0–3 MLA cache: latent c_KV 512 + rotary key 64).",
  "The cartridges are appended one by one into one consolidated numerical memory. For each append, layers 4–26 are computed only for the new tokens, attending to the existing memory; the rotary part of layers 0–3 is re-phased to the new positions with the model's own rotary embedding.",
  "Measured in every run: the consolidation forward receives all-zero input embeddings plus the stored H3 (no token ids); no earlier cartridge's source is read again; every previously consolidated cache row is bitwise identical after each append.",
  "The question is answered from the numerical memory: the query forward receives only the question-template tokens over the installed memory.",
  "Model weights are unchanged (all-parameter guard before and after the run; the engine's own weight fingerprint unchanged)."],
 "archived":[
  "TEST596 reference on the shared locked panel (24 five-cartridge cases × FIRST/MIDDLE/LAST = 72): append-only incremental MLA DC3 69/72 (FIRST 24/24, MIDDLE 23/24, LAST 22/24). Reference values; single live runs do not recompute them. The optional live replay reports its own score and compares it with this reference.",
  "The three failed cases, kept in full: case 11 LAST (expected Mirev, answered Kareth), case 14 LAST (expected Corven, answered Marvek), case 23 MIDDLE (expected Karev, answered Luneth).",
  "TEST597 reproduced the three failures. On these three slots the joint-text control answered 3/3 and native independent MLA cartridges 0/3. No verified readout fix was found, so no score above 69/72 is claimed.",
  "Earlier records of the same architecture: Mistral-7B (DC6), DOI "+PRIOR_PUBS[0]["doi"]+"; Qwen2.5-7B (DC3), DOI "+PRIOR_PUBS[1]["doi"]+". Cited as references only; they are not DeepSeek results."],
 "not_established":[
  "Error-free readout on this model: 3 of 72 panel questions are answered with a distractor name (69/72).",
  "Memories larger than five cartridges are not established for DeepSeek in this record.",
  "Strict cache-only sufficiency: the cartridge is augmented (H3 residual + layers 0–3 MLA cache).",
  "Questions outside the locked panel template: free-text questions are shown but not scored.",
  "Multi-step inference that combines several cartridges.",
  "Generality beyond the three tested model families: each family needs its own consolidation boundary; other models are not tested."],
 "notes":[
  "Writing cartridge k reads only fact k (inside the DeepSeek chat template) once; this is the write event of that cartridge. The engine writes the five cartridges first, then initialises and appends.",
  "The 28-token instruction prefix is written with the first cartridge; later cartridges contribute their body rows only.",
  "Expected answers come from panel construction and are audit metadata; they are not an input to writing, consolidation or answering.",
  "Scoring follows the engine: "+CORRECT_RULE_TXT+".",
  "The engine is 596.py (git blob "+ENGINE_BLOB_EXPECTED[:12]+"…), executed unchanged; its demo API (demo_load_case, demo_ask, demo_verify_all) is called directly.",
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
ALLOWED_TESTS={"596","597"}
CLAIM_RULES=[("capacity-overclaim",r"(?i)\bunlimited\b|\binfinite\b"),("universal-claim",r"(?i)\buniversal\b"),("human-like-claim",r"(?i)human-like"),
    ("model-independence-claim",r"(?i)model-independent"),("all-transformers-claim",r"(?i)all transformers|every transformer|all (?:large )?language models"),("solved-claim",r"(?i)\bsolved\b"),
    ("perfect-claim",r"(?i)\bperfect\b"),("world-first-claim",r"(?i)world'?s first|first in the world|independently proven"),
    ("legal-status-claim",r"(?i)\bpatent(ed|-pending)\b|non-obvious|\bpatentab"),("metaphor",r"(?i)\bbrain\b|\bquantum\b|\bneuron"),
    ("complexity-claim",r"O\(1\)"),("compression-wording",r"(?i)\bcompress")]
# ---- foreign-model leak scanner: no value of the earlier Mistral or Qwen records may appear as a DeepSeek value. Only citations are exempt. ----
LEAK_RULES=[("mistral-name",r"(?i)mistral"),("qwen-name",r"(?i)qwen"),("mistral-cut",r"\bH6\b|\bDC6\b|\b(?:layers?|L)\s?0[–-]6\b|\blayers 7[–-]31\b"),
    ("earlier-tests",r"TEST\s?5[67]\d\b"),("earlier-arch-words",r"\b3[2] (?:layers|L\b|Q)|\b28 (?:layers|L\b|Q)|\b[48] KV\b|\b2[79]-token|\blayers 4[–-]27\b"),
    ("perfect-panel-claim",r"(?i)DeepSeek[^.]{0,60}72/72")]
LEAK_RULES_NUM=[("earlier-hidden",r"(?<![\d.])(?:4096|3584)(?![\d.]|\s?ms\b)")]
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
    return re.sub(r"(?<![A-Za-z])(?:"+"|".join(map(re.escape,ws))+r")(?![A-Za-z])","",text)
def cpu_selftest():
    n=0
    assert correct_rule("Melket","Melket") and correct_rule("melket.","Melket") and not correct_rule("The capital is Melket.","Melket");n+=1
    ok_txt=("DeepSeek-V2-Lite-Chat TEST596 TEST597 69/72 append-only DC3 H3 layers 0–3 layers 4–26 27 layers hidden 2048 28-token prefix MLA latent 512 rotary 64 "
            +PARADIGM+" "+DISCOVERY+" "+PRIORITY+" "+CROSS_TITLE+" "+CROSS_TEXT+" "+CROSS_RESULT+" "+CROSS_BOUNDARY+" "+CROSS_REF+" "
            +" ".join(SCOPE["archived"]+SCOPE["not_established"]+SCOPE["demonstrated"]+SCOPE["notes"]))
    assert claim_scan("t",ok_txt,numeric=True)==[],claim_scan("t",ok_txt,numeric=True);n+=1
    for b in("unlimited memory","universal memory","human-like memory","model-independent","works on all transformers","works in all language models","memory solved","perfect recall",
             "world's first","patent-pending","brain","quantum","O(1)","compressed","TEST528","TEST560","TEST575","TEST572","Mistral","Qwen","H6 residual","DC6","layers 0–6",
             "32 layers","28 layers","4 KV heads","8 KV heads","27-token prefix","29-token prefix","layers 4–27","DeepSeek scored 72/72"):
        assert claim_scan("t",b),b;n+=1
    assert claim_scan("t","hidden 4096",numeric=True) and claim_scan("t","hidden 3584",numeric=True) and not claim_scan("t","hidden 3584") and not claim_scan("t","t=0.40963 sha 4096ab12cd34ef56",numeric=True);n+=1
    assert strip_words("Zorvan BRAIN carried",["BRAIN"])=="Zorvan  carried";n+=1
    assert sum(TEST596["score"][p]for p in POSITIONS)==TEST596["score"]["TOTAL"]==69 and len(TEST596["failures"])==72-69;n+=1
    assert {p:24-sum(1 for c,q,_,_ in TEST596["failures"]if q==p)for p in POSITIONS}=={p:TEST596["score"][p]for p in POSITIONS};n+=1
    assert reference_outcome(11,"LAST")=={"correct":False,"answer":"Kareth"} and reference_outcome(0,"FIRST")["correct"];n+=1
    return n
#<<CONST_END>>
#<<ENGINE_BEGIN>>
for _m,_p in[("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("gradio","gradio"),("matplotlib","matplotlib"),("PIL","pillow")]:
    if importlib.util.find_spec(_m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",_p])
import numpy as np,torch,transformers,gradio as gr
QUIET=False
def say(*a):
    if not QUIET:print(*a,flush=True)
def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="milliseconds")
say("="*140);say(DEMO_TITLE+" — "+MODEL_FAMILY+" · PART 1 / 3 · ENGINE");say(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");say(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");say("="*140)
say(f"[1/5] CPU self-test PASS ({cpu_selftest()} checks)")
# ---------------- 596.py — embedded verbatim (do not edit; its git blob SHA-1 and SHA-256 are verified below) ----------------
ENGINE_SRC=r'''# AKBASCORE MAM - DEEPSEEK TEST596 - TEST560/575 CANONICAL PANEL TRANSFER
# 24 worlds x FIRST/MIDDLE/LAST, 96 cartridges, 5 per case, panel SHA locked.
# CUT3 MLA WRITE/INIT/APPEND extracted without algorithmic changes from TEST590.
# Demonstration backend, not an independent validation. GPU run required. No Gradio.
# JOINT and INDEP optional controls, default runs full 72 incremental cases.
import os,sys,time,gc,re,json,hashlib,random,urllib.request,ast,traceback
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,transformers
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST='DEEPSEEK_DEMO_V2';MODEL='deepseek-ai/DeepSeek-V2-Lite-Chat';REV='85864749cd611b4353ce1decdb286193298f64c7'
DTYPE=torch.bfloat16;SEED=577;CUT=3;NL=27;H=2048;MAX_NEW=24;BEAM_WIDTH=4;BEAM_STEPS=48
PANEL_SEED=550550;EXPECTED_PANEL_SHA='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45'
SOURCES={'560.py':'35d95a929a4e043a9af346c67fc103da47a792b5','590.py':'09ae7a6862956128465efc7e76cdca1e478a1a52'}
RUN_JOINT=os.getenv('AKBAS_RUN_JOINT','0')=='1';RUN_INDEP=os.getenv('AKBAS_RUN_INDEP','0')=='1'
START_CASE=int(os.getenv('AKBAS_START_CASE','0'));END_CASE=int(os.getenv('AKBAS_END_CASE','24'))
assert 0<=START_CASE<END_CASE<=24
random.seed(SEED);torch.manual_seed(SEED);torch.set_grad_enabled(False);T0=time.time()
GATES={};HOOK_COUNT={'init':0,'append':0};ROWS=[]
def gate(name,ok):
    GATES[name]=bool(ok);print('GATE',name,'PASS' if ok else 'FAIL',flush=True)
def git_blob(s):
    b=s.encode('utf-8');return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def fetch_verified(name):
    u='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/'+name
    req=urllib.request.Request(u,headers={'User-Agent':'AKBASCORE-TEST596'})
    with urllib.request.urlopen(req,timeout=60) as resp:s=resp.read().decode('utf-8')
    actual=git_blob(s);assert actual==SOURCES[name],f'{name} SOURCE BLOB MISMATCH {actual}'
    print('SOURCE_VERIFIED',name,actual,flush=True);return s
print('='*120);print('TEST596 - DEEPSEEK MLA CUT3 / EXACT TEST560-575 PANEL TRANSFER');print('='*120)
source560=fetch_verified('560.py');source590=fetch_verified('590.py')
# Extract BASE literal without executing Mistral code or loading Mistral weights.
base_node=next(n for n in ast.parse(source560).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE' for t in n.targets))
BASE=ast.literal_eval(base_node.value)
assert len(BASE)==24
print('SOURCE_PANEL_BASE:',len(BASE),'worlds')
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 REQUIRED'
print('GPU:',torch.cuda.get_device_name(0),'TORCH:',torch.__version__,'TRANSFORMERS:',transformers.__version__)
def replace_hidden(out,new):
    if isinstance(out,tuple):return (new,)+out[1:]
    if isinstance(out,list):return [new]+out[1:]
    return new
old=globals().get('model',None)
reuse=isinstance(old,torch.nn.Module) and getattr(old,'_akbas_verified_revision',None)==REV and len(getattr(getattr(old,'model',None),'layers',[]))==NL and all(p.device.type=='cuda' and (not p.is_floating_point() or p.dtype==DTYPE) for p in old.parameters())
if reuse:model=old;print('MODEL_REUSED YES')
else:
    if isinstance(old,torch.nn.Module):globals().pop('model',None);del old;gc.collect();torch.cuda.empty_cache()
    print('MODEL_LOADING PINNED REVISION')
    cfg=AutoConfig.from_pretrained(MODEL,revision=REV,trust_remote_code=False);cfg._attn_implementation='eager'
    model,li=AutoModelForCausalLM.from_pretrained(MODEL,config=cfg,revision=REV,trust_remote_code=False,device_map={'':0},low_cpu_mem_usage=True,output_loading_info=True,attn_implementation='eager',dtype=DTYPE)
    for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):assert not li.get(k,[]),f'LOAD ERROR {k}'
    model._akbas_verified_revision=REV
model.eval()
for pp in model.parameters():pp.requires_grad_(False)
base=model.model;LAYERS=base.layers;DEVICE=model.get_input_embeddings().weight.device
assert len(LAYERS)==NL and base.config.hidden_size==H and DEVICE.type=='cuda'
tok=globals().get('tok',None)
if tok is None or getattr(tok,'name_or_path',None)!=MODEL:tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,trust_remote_code=False)
EOS_IDS={int(tok.eos_token_id)};eos_cfg=getattr(model.generation_config,'eos_token_id',None)
for x in (eos_cfg if isinstance(eos_cfg,(list,tuple)) else [eos_cfg]):
    if x is not None:EOS_IDS.add(int(x))
def ids(s):return tok(s,add_special_tokens=False,return_tensors='pt').input_ids.to(DEVICE)
def run(input_ids=None,inputs_embeds=None,past=None,attention_mask=None,position_ids=None,cache_position=None):
    with torch.inference_mode():return model(input_ids=input_ids,inputs_embeds=inputs_embeds,past_key_values=past,attention_mask=attention_mask,position_ids=position_ids,cache_position=cache_position,use_cache=True,return_dict=True,logits_to_keep=1)
def layers_of(c):
    assert hasattr(c,'layers') and len(c.layers)==NL
    out=[]
    for i,l in enumerate(c.layers):
        k=getattr(l,'keys',None);r=getattr(l,'values',None)
        assert torch.is_tensor(k) and torch.is_tensor(r) and k.ndim==4 and r.ndim==4
        assert k.shape[:3]==r.shape[:3] and k.shape[-1]==512 and r.shape[-1]==64,f'MLA CACHE L{i}'
        out.append((k,r))
    return out
def clone_layers(c):return tuple((k.detach().clone(),r.detach().clone()) for k,r in layers_of(c))
def make_cache(data):
    c=DynamicCache()
    for i,(k,r) in enumerate(data):c.update(k.detach().clone(),r.detach().clone(),i)
    return c
def length(data):return int(data[0][0].shape[-2])
def validate(data,n):
    assert len(data)==NL
    for i,(k,r) in enumerate(data):
        assert tuple(k.shape)==(1,1,n,512),f'LATENT L{i}: {tuple(k.shape)}'
        assert tuple(r.shape)==(1,1,n,64),f'ROTARY L{i}: {tuple(r.shape)}'
        assert torch.isfinite(k).all() and torch.isfinite(r).all()
    return True
def fingerprint():
    h=hashlib.sha256()
    for i in (0,3,6,12,20,26):
        pp=next(LAYERS[i].parameters()).detach().reshape(-1);n=pp.numel()
        for j in range(8):
            a=j*(n-128)//7;h.update(pp[a:a+128].float().cpu().numpy().tobytes())
    return h.hexdigest()
FP0=fingerprint();gate('MODEL_FROZEN',all(not pp.requires_grad for pp in model.parameters()))
# Exact 24-world fact/query definitions and panel RNG protocol of TEST560/575.
W=[]
for wi,(s,current,near,former,role_target) in enumerate(BASE):
    facts=[('CURRENT',f'The current capital of {s} is {current}.',current),('NEAR',f'The largest city of {s} is {near}.',near),('FORMER',f'The former capital of {s} was {former}.',former),('ROLE',f'The current capital of {role_target} is {s}.',s)]
    queries=[('CURRENT',f'What is the current capital of {s}?',current),('FORMER',f'What was the former capital of {s}?',former),('NEAR',f'What is the largest city of {s}?',near),('ROLE',f'What is the current capital of {role_target}?',s)]
    W.append({'id':wi,'facts':facts,'queries':queries})
CAR=[];KEY_TO_IDX={}
for wi,w in enumerate(W):
    for typ,fact,gold in w['facts']:
        ci=len(CAR);CAR.append({'idx':ci,'world':wi,'type':typ,'fact':fact,'gold':gold});KEY_TO_IDX[(wi,typ)]=ci
rng=random.Random(PANEL_SEED);pool=list(range(96));rng.shuffle(pool);PANEL=[];cursor=0
QUERY_CYCLE=['CURRENT','FORMER','NEAR','ROLE'];POSITIONS=['FIRST','MIDDLE','LAST']
for wi in range(24):
    typ=QUERY_CYCLE[wi%4];target=KEY_TO_IDX[(wi,typ)];others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]['world']==wi or any(CAR[x]['world']==CAR[ci]['world'] for x in others):continue
        others.append(ci)
    PANEL.append({'case':wi,'target':target,'others':others})
PANEL_SHA=sha_obj(PANEL);print('PANEL_SHA:',PANEL_SHA)
gate('PANEL_SHA_560_575',PANEL_SHA==EXPECTED_PANEL_SHA);assert GATES['PANEL_SHA_560_575']
def order_keys(z,slot):
    d=list(z['others']);t=z['target']
    return [t]+d if slot=='FIRST' else d[:2]+[t]+d[2:] if slot=='MIDDLE' else d+[t]
# DeepSeek-native chat wrapping; facts, distractors, order, questions and scoring are from TEST560/575.
SYSTEM='Answer the question using only the information provided. Give only the requested name and nothing else.'
PREFIX=tok.apply_chat_template([{'role':'user','content':SYSTEM+'\nINFORMATION:\n'}],tokenize=False,add_generation_prompt=False)
PREFIX_IDS=ids(PREFIX)[0].tolist();P=len(PREFIX_IDS)
for c in CAR:
    full=ids(PREFIX+c['fact']+'\n')[0].tolist()
    assert full[:P]==PREFIX_IDS,f'PREFIX MISMATCH ci={c["idx"]}'
    c['full']=full;c['body']=full[P:];c['body_len']=len(full)-P
def question(q):return tok.apply_chat_template([{'role':'user','content':q+' Answer with the name only.'}],tokenize=False,add_generation_prompt=True)
print('CARTRIDGES',len(CAR),'PREFIX',P,'BODY_LENGTHS',sorted(set(len(c['body']) for c in CAR)),flush=True)
def rephase_rotary(r,oldpos,newpos):
    assert r.shape[-1]==64 and oldpos.numel()==newpos.numel()==r.shape[-2]
    if torch.equal(oldpos,newpos):return r.detach().clone()
    n=oldpos.numel();emb=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE)
    with torch.inference_mode():
        f0=base.rotary_emb(emb,oldpos.unsqueeze(0)).to(torch.complex64)
        f1=base.rotary_emb(emb,newpos.unsqueeze(0)).to(torch.complex64)
    phase=f1*torch.conj(f0);z=torch.view_as_complex(r.float().reshape(1,1,n,32,2).contiguous())
    return torch.view_as_real(z*phase.unsqueeze(1)).reshape_as(r.float()).to(r.dtype)

def write_cartridge(ci):
    c=CAR[ci];x=torch.tensor([c['full']],device=DEVICE,dtype=torch.long);n=x.shape[1];pos=torch.arange(n,device=DEVICE);state={};handles=[]
    def capture(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out;state['h']=z.detach().clone()
    try:
        handles.append(LAYERS[CUT].register_forward_hook(capture))
        o=run(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
        data=clone_layers(o.past_key_values)
    finally:
        for h in handles:h.remove()
    assert state['h'].shape==(1,n,H);validate(data,n)
    return {'h':state['h'],'early':data[:CUT+1],'pos':pos.detach().clone(),'body_len':n-P,'native':data}
def init_cartridge(ci):
    cp=CP[ci];n=cp['h'].shape[1];seen=[0];handles=[]
    def inject(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out;assert z.shape==cp['h'].shape;seen[0]+=1
        return replace_hidden(out,cp['h'])
    try:
        handles.append(LAYERS[CUT].register_forward_hook(inject));pos=torch.arange(n,device=DEVICE)
        o=run(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
        upper=clone_layers(o.past_key_values)
    finally:
        for h in handles:h.remove()
    result=tuple((k.clone(),r.clone()) for k,r in cp['early'])+upper[CUT+1:];validate(result,n);HOOK_COUNT['init']+=seen[0]
    delta=max(float((a.float()-b.float()).abs().max().item()) for pa,pb in zip(result,cp['native']) for a,b in zip(pa,pb))
    return result,seen[0],delta
def append_cartridge(memory,ci):
    cp=CP[ci];oldn=length(memory);q=cp['body_len'];newpos=torch.arange(oldn,oldn+q,device=DEVICE);oldpos=cp['pos'][P:]
    body_h=cp['h'][:,P:,:];assert body_h.shape==(1,q,H)
    cache=make_cache(memory);calls=[0];handles=[]
    def inject(module,args,out):
        z=out[0] if isinstance(out,(tuple,list)) else out;assert z.shape==body_h.shape;calls[0]+=1
        return replace_hidden(out,body_h)
    try:
        handles.append(LAYERS[CUT].register_forward_hook(inject))
        o=run(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past=cache,attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=newpos.unsqueeze(0),cache_position=newpos)
        grown=list(clone_layers(o.past_key_values))
    finally:
        for h in handles:h.remove()
    for li in range(CUT+1):
        oldk,oldr=memory[li];newk,newr=cp['early'][li]
        body_k=newk[:,:,P:,:].detach().clone();body_r=rephase_rotary(newr[:,:,P:,:],oldpos,newpos)
        grown[li]=(torch.cat((oldk,body_k),dim=-2),torch.cat((oldr,body_r),dim=-2))
    result=tuple(grown);validate(result,oldn+q);HOOK_COUNT['append']+=calls[0]
    unchanged=all(torch.equal(a,na[:,:,:oldn,:]) and torch.equal(b,nb[:,:,:oldn,:]) for (a,b),(na,nb) in zip(memory,result))
    return result,calls[0],unchanged
def joint_reference(order,stage_size):
    seq=list(PREFIX_IDS)
    for ci in order[:stage_size]:seq+=CAR[ci]['body']
    x=torch.tensor([seq],device=DEVICE,dtype=torch.long);n=x.shape[1];pos=torch.arange(n,device=DEVICE)
    o=run(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    result=clone_layers(o.past_key_values);validate(result,n);return result
def prefix_forward(memory,qtext):
    cache=make_cache(memory);n=length(memory);qids=ids(qtext);ql=qids.shape[1];pos=torch.arange(n,n+ql,device=DEVICE)
    o=run(input_ids=qids,past=cache,attention_mask=torch.ones((1,n+ql),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    return clone_layers(o.past_key_values),o.logits[:,-1,:].float()
def step_forward(cache_data,token):
    n=length(cache_data);pos=torch.tensor([n],device=DEVICE)
    o=run(input_ids=torch.tensor([[int(token)]],device=DEVICE,dtype=torch.long),past=make_cache(cache_data),attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos)
    return clone_layers(o.past_key_values),o.logits[:,-1,:].float()
def greedy_from_prefix(cache_data,logits,max_new=MAX_NEW):
    generated=[];cache=cache_data
    for _ in range(max_new):
        t=int(logits.argmax(-1).item())
        if t in EOS_IDS:break
        generated.append(t);cache,logits=step_forward(cache,t)
    return tok.decode(generated,skip_special_tokens=True).strip(),generated
def beam_candidates(cache_data,logits,width=BEAM_WIDTH,steps=BEAM_STEPS):
    beams=[{'tokens':[],'logp':0.0,'cache':cache_data,'logits':logits,'done':False}]
    for step in range(steps):
        expanded=[]
        for b in beams:
            if b['done']:expanded.append(b);continue
            lp=torch.log_softmax(b['logits'],dim=-1);vals,idx=torch.topk(lp,width,dim=-1)
            for v,t in zip(vals[0].tolist(),idx[0].tolist()):
                t=int(t);expanded.append({'tokens':b['tokens']+[t],'logp':b['logp']+float(v),'cache':b['cache'],'logits':b['logits'],'done':t in EOS_IDS})
        expanded.sort(key=lambda x:x['logp'],reverse=True);selected=expanded[:width];new=[]
        for b in selected:
            if b['done']:new.append(b)
            else:
                c,l=step_forward(b['cache'],b['tokens'][-1]);b['cache']=c;b['logits']=l;new.append(b)
        beams=new
        if all(b['done'] for b in beams):break
    candidates=[];seen=set()
    for b in beams:
        raw=tok.decode([t for t in b['tokens'] if t not in EOS_IDS],skip_special_tokens=True).strip()
        if raw in seen:continue
        seen.add(raw);candidates.append({'text':raw,'tokens':b['tokens'],'logp':b['logp'],'length':len(b['tokens']),'ended':b['done']})
    return candidates


# WRITE is independent; CP created on demand to avoid 96-cartridge GPU OOM.
CP={}
def get_cp(ci):
    if ci not in CP:CP[ci]=write_cartridge(ci)
    return CP[ci]
# original engine expects CP[ci] - prefill selected 5 before each case.
def normalize(s):return ' '.join(re.sub(r'[^\w\s-]',' ',str(s).casefold()).split())
def select_name(greedy,candidates):
    # No gold, no list of known names, no oracle; use autonomous greedy output.
    # Completed beam retained as diagnostic, not used to change answer without verified name readout.
    return normalize(greedy)
def ask_name(memory,q):
    cache,logits=prefix_forward(memory,question(q))
    greedy,_=greedy_from_prefix(cache,logits)
    return {'answer':select_name(greedy,[]),'raw':greedy}
def indep_reference(keys):
    # Source-only independent cartridge caches, positioned in shared memory; no cross-cartridge interaction.
    n=P+sum(CAR[ci]['body_len'] for ci in keys)
    out=[]
    for li in range(NL):
        kk=[];rr=[];off=P
        for j,ci in enumerate(keys):
            cp=CP[ci];k,r=cp['native'][li]
            if j==0:kk.append(k[:,:,:P,:]);rr.append(r[:,:,:P,:])
            q=cp['body_len'];oldpos=cp['pos'][P:];newpos=torch.arange(off,off+q,device=DEVICE)
            kk.append(k[:,:,P:,:]);rr.append(rephase_rotary(r[:,:,P:,:],oldpos,newpos));off+=q
        out.append((torch.cat(kk,dim=-2),torch.cat(rr,dim=-2)))
    result=tuple(out);validate(result,n);return result

# ===== DEMO ADAPTER: no changes above this boundary to TEST596 CUT3 engine =====
from datetime import datetime,timezone
from threading import RLock
from collections import Counter
_DEMO_LOCK=RLock()
_ACTIVE=None
_DEMO_ROWS=[]
_DEMO_EVENTS=[]
REFERENCE={'TEST596':{'correct':69,'total':72,'by_slot':{'FIRST':24,'MIDDLE':23,'LAST':22},'failures':[(11,'LAST'),(14,'LAST'),(23,'MIDDLE')]},'TEST597':{'joint':(3,3),'incremental':(6,9),'independent':(0,3),'readout_fix_verified':False}}
_DEMO_DIR=Path('/content/AKBASCORE_DEEPSEEK_DEMO' if Path('/content').is_dir() else '/tmp/AKBASCORE_DEEPSEEK_DEMO');_DEMO_DIR.mkdir(parents=True,exist_ok=True)
def _utc():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def _sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,default=str).encode('utf-8')).hexdigest()
def _event(kind,**kw):
    e={'utc':_utc(),'kind':kind,**kw};_DEMO_EVENTS.append(e);return e
def _clear_active():
    global _ACTIVE
    _ACTIVE=None;CP.clear();gc.collect();torch.cuda.empty_cache()
def _assert_clean():
    assert all(not l._forward_hooks and not l._forward_pre_hooks for l in LAYERS),'RESIDUAL HOOKS'
    assert fingerprint()==FP0,'WEIGHT FINGERPRINT DRIFT'
    assert all(not p.requires_grad for p in model.parameters()),'TRAINABLE WEIGHTS'
    assert not model.training,'MODEL TRAINING MODE'
    assert PANEL_SHA==EXPECTED_PANEL_SHA,'PANEL SHA MISMATCH'
def demo_cases():
    return [{'case':wi,'type':CAR[z['target']]['type'],'question':next(x[1] for x in W[wi]['queries'] if x[0]==CAR[z['target']]['type']),'slots':list(POSITIONS)} for wi,z in enumerate(PANEL)]
def demo_status():
    return {'backend':'AKBASCORE MAM DeepSeek demo V2','model':MODEL,'revision':REV,'model_frozen':all(not p.requires_grad for p in model.parameters()),'cut':CUT,'layers':NL,'hidden':H,'cache':'MLA latent 512 + rotary 64','panel_sha':PANEL_SHA,'panel_verified':PANEL_SHA==EXPECTED_PANEL_SHA,'cartridges':len(CAR),'cases':len(PANEL),'source_blobs':dict(SOURCES),'reference_results':REFERENCE,'engine':'TEST596 unchanged numerical WRITE/INIT/APPEND/greedy ASK','scope':'same logical TEST560/575 panel, DeepSeek-specific tokenization; not TEST575 five-arm equivalence','active':None if _ACTIVE is None else {k:v for k,v in _ACTIVE.items() if k not in ('memory','keys')},'run_count':len(_DEMO_ROWS)}
def demo_load_case(case_id,slot='MIDDLE',progress=None):
    global _ACTIVE
    with _DEMO_LOCK:
        _assert_clean();wi=int(case_id);slot=str(slot).upper()
        if wi not in range(24) or slot not in POSITIONS:raise ValueError('case must be 0..23; slot FIRST/MIDDLE/LAST')
        _clear_active();z=PANEL[wi];keys=order_keys(z,slot);typ=CAR[z['target']]['type']
        q=next(x[1] for x in W[wi]['queries'] if x[0]==typ);gold=CAR[z['target']]['gold']
        try:
            for ci in keys:CP[ci]=write_cartridge(ci)
            if progress:progress({'step':1,'total':len(keys),'event':'WRITE/INIT','cartridge':0,'memory_tokens':0})
            mem,calls,delta=init_cartridge(keys[0]);assert calls==1 and delta==0.0,'INIT DIVERGENCE'
            steps=[{'stage':1,'cartridge':keys[0],'memory_tokens':length(mem),'old_prefix_unchanged':True}]
            for j,ci in enumerate(keys[1:],2):
                oldn=length(mem);grown,calls,ok=append_cartridge(mem,ci)
                assert calls==1 and ok,'APPEND NON-DESTRUCTIVE GATE FAILED'
                assert length(grown)==oldn+CAR[ci]['body_len'],'APPEND LENGTH GATE FAILED'
                steps.append({'stage':j,'cartridge':ci,'memory_tokens':length(grown),'old_prefix_unchanged':ok})
                if progress:progress({'step':j,'total':len(keys),'event':'APPEND','cartridge':j-1,'memory_tokens':length(grown)})
                del mem;mem=grown
            assert length(mem)==P+sum(CAR[ci]['body_len'] for ci in keys),'TOTAL LENGTH'
            _ACTIVE={'case':wi,'slot':slot,'keys':keys,'memory':mem,'question':q,'gold':gold,'type':typ,'steps':steps,'memory_tokens':length(mem),'loaded_utc':_utc()}
            _assert_clean();_event('case_loaded',case=wi,slot=slot,steps=steps)
            return {'case':wi,'slot':slot,'question':q,'type':typ,'memory_tokens':length(mem),'steps':steps,'source_text_supplied_at_query':False,'weight_fingerprint_unchanged':True,'append_only_verified':True}
        except BaseException:
            _clear_active();raise
def demo_ask(question_text=None,diagnostic_beam=False):
    with _DEMO_LOCK:
        if _ACTIVE is None:raise RuntimeError('Call demo_load_case first')
        _assert_clean();a=_ACTIVE
        q=a['question'] if question_text is None else str(question_text).strip()
        if not q or len(q)>1000:raise ValueError('Question length must be 1..1000')
        # The source text is not replayed. This calls TEST596 original ASK path.
        output=ask_name(a['memory'],q)
        row={'utc':_utc(),'case':a['case'],'slot':a['slot'],'arm':'INCR_DC3_MLA','question':q,'answer':output['raw'],'normalized':output['answer'],'memory_tokens':a['memory_tokens'],'is_locked_panel_question':q==a['question']}
        if row['is_locked_panel_question']:
            row['gold']=a['gold'];row['correct']=output['answer']==normalize(a['gold'])
        else:
            row['gold']=None;row['correct']=None
        if diagnostic_beam:
            cache,logits=prefix_forward(a['memory'],question(q));cands=beam_candidates(cache,logits)
            row['beam_candidates']=[{'raw':c['text'],'logp':round(float(c['logp']),7),'ended':bool(c['ended']),'tokens':int(c['length'])} for c in cands]
            row['beam_diagnostic_only']=True
        _DEMO_ROWS.append(row);_assert_clean();_event('question_answered',case=a['case'],slot=a['slot'],correct=row['correct'])
        return dict(row)
def demo_run_case(case_id,slot='MIDDLE',diagnostic_beam=False,progress=None):
    info=demo_load_case(case_id,slot,progress=progress);ans=demo_ask(diagnostic_beam=diagnostic_beam)
    return {'load':info,'answer':ans,'status':demo_status()}
def demo_verify_all(progress=None,save=True):
    # Re-run exact locked TEST596 72 queries. This is independent of the published 69/72 reference record.
    with _DEMO_LOCK:
        _assert_clean();rows=[];t0=time.time()
        for wi in range(24):
            for slot in POSITIONS:
                info=demo_load_case(wi,slot);r=demo_ask();rows.append(r)
                if progress:progress({'done':len(rows),'total':72,'case':wi,'slot':slot,'correct':r['correct']})
        by_slot={s:sum(r['correct'] for r in rows if r['slot']==s) for s in POSITIONS}
        correct=sum(r['correct'] for r in rows);failures=[{'case':r['case'],'slot':r['slot'],'expected':r['gold'],'actual':r['answer']} for r in rows if not r['correct']]
        match=(correct==69 and by_slot==REFERENCE['TEST596']['by_slot'] and {(r['case'],r['slot']) for r in rows if not r['correct']}==set(REFERENCE['TEST596']['failures']))
        record={'utc':_utc(),'model':MODEL,'revision':REV,'panel_sha':PANEL_SHA,'scope':'72-query locked panel re-run, no gold-based answer selection','correct':correct,'total':72,'by_slot':by_slot,'failures':failures,'matches_test596_reference':match,'weight_fingerprint_unchanged':fingerprint()==FP0,'seconds':round(time.time()-t0,2),'rows':rows}
        record['payload_sha256']=_sha(record)
        if save:
            p=_DEMO_DIR/f'verification_{int(time.time())}.json';p.write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf-8');record['saved_to']=str(p)
        _assert_clean();_event('panel_verification',correct=correct,match=match)
        return record
def demo_export():
    with _DEMO_LOCK:
        _assert_clean()
        payload={'utc':_utc(),'engine':'TEST596 DeepSeek CUT3 MLA, numerical functions unchanged','reference':REFERENCE,'status':demo_status(),'rows':list(_DEMO_ROWS),'events':list(_DEMO_EVENTS),'note':'Reference scores are historical, not a substitute for live verification. Custom questions are unscored.'}
        payload['sha256']=_sha(payload)
        p=_DEMO_DIR/f'demo_record_{int(time.time())}.json';p.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
        return {'path':str(p),'sha256':payload['sha256'],'rows':len(_DEMO_ROWS)}
def demo_selftest(gpu=False):
    with _DEMO_LOCK:
        _assert_clean()
        assert len(PANEL)==24 and len(CAR)==96 and len(demo_cases())==24
        assert all(len(order_keys(z,s))==5 and len(set(order_keys(z,s)))==5 for z in PANEL for s in POSITIONS)
        assert REFERENCE['TEST596']['correct']==69 and sum(REFERENCE['TEST596']['by_slot'].values())==69
        result={'static':'PASS','panel_sha':PANEL_SHA,'weight_fingerprint_unchanged':True,'gpu_case':None}
        if gpu:
            # Non-oracle runtime smoke test, known TEST596 passing case 0/FIRST.
            r=demo_run_case(0,'FIRST');result['gpu_case']={'case':0,'slot':'FIRST','correct':r['answer']['correct'],'answer':r['answer']['answer']}
            assert r['answer']['correct'],'SMOKE TEST CASE 0 FAILED'
        return result
print('DEEPSEEK_DEMO_BACKEND_READY; no Gradio; functions: demo_cases, demo_status, demo_load_case, demo_ask, demo_run_case, demo_verify_all, demo_export, demo_selftest',flush=True)
print('REFERENCE_TEST596: 69/72; TEST597: no verified readout fix; not claiming 72/72',flush=True)
'''
_ENGINE_BYTES=ENGINE_SRC.encode("utf-8")
ENGINE_SHA256=hashlib.sha256(_ENGINE_BYTES).hexdigest();ENGINE_BLOB=git_blob_sha1(_ENGINE_BYTES)
if ENGINE_BLOB!=ENGINE_BLOB_EXPECTED or ENGINE_SHA256!=ENGINE_SHA256_EXPECTED:
    raise RuntimeError(f"Embedded 596.py differs from the repository file (blob {ENGINE_BLOB} / SHA-256 {ENGINE_SHA256}). Re-paste PART 1 unchanged.")
ENGINE_EXEC_SHA256=ENGINE_SHA256
say(f"[2/5] Engine source verified · {ENGINE_FILE} · git blob {ENGINE_BLOB} · SHA-256 {ENGINE_SHA256}")
STARTUP_UTC=utc_now()
if"ENG"in globals()and getattr(globals()["ENG"],"__engine_blob__",None)==ENGINE_BLOB and globals().get("DEMO_ENGINE")is not None:
    say("[3/5] Engine already initialised in this runtime — reusing it (no second model load).");ENGINE_INIT_SECONDS=globals().get("ENGINE_INIT_SECONDS",float("nan"))
else:
    say("[3/5] Executing 596.py unchanged in its own module namespace (verifies 560.py/590.py, loads frozen DeepSeek-V2-Lite-Chat ≈31 GB, a few minutes) …")
    say("      (the lines up to DEEPSEEK_DEMO_BACKEND_READY are printed by 596.py itself)");_t=time.perf_counter()
    ENG=types.ModuleType("mam_test596_deepseek_engine");ENG.__file__=ENGINE_FILE+" (embedded verbatim)";sys.modules[ENG.__name__]=ENG
    exec(compile(ENGINE_SRC,ENGINE_FILE,"exec"),ENG.__dict__)
    ENG.__engine_blob__=ENGINE_BLOB;ENG.__engine_sha256__=ENGINE_SHA256
    # ---------------- demo adapter: maps the engine's own demo API onto the interface of the AKBASCORE demo family ----------------
    class DeepSeekDC3:
        """Calls only ENG.demo_load_case / ENG.demo_ask / ENG.demo_verify_all / ENG.demo_cases / ENG.demo_status (596.py, unchanged)."""
        def __init__(s):
            s.tok=ENG.tok;s.model=ENG.model;s.nl=ENG.NL;s.h=ENG.H;s.CUT=ENG.CUT;s.prefix_ids=list(ENG.PREFIX_IDS);s.p=ENG.P;s.car=ENG.CAR;s.panel=ENG.PANEL;s.panel_sha=ENG.PANEL_SHA
        @property
        def _memory(s):return None if ENG._ACTIVE is None else ENG._ACTIVE["memory"]
        def _keys(s,case,position):return list(ENG.order_keys(ENG.PANEL[int(case)],str(position)))
        def cases(s):return[{"case":c["case"],"type":c["type"],"question":c["question"],"positions":list(c["slots"])}for c in ENG.demo_cases()]
        def status(s):
            st=ENG.demo_status()
            return{"title":"AKBASCORE MAM","developer":ORG,"author":AUTHOR,"date":DATE_ISO,"model":st["model"],"revision":st["revision"],"cut":st["cut"],"layers":st["layers"],"hidden":st["hidden"],
                   "cache":st["cache"],"panel_sha":st["panel_sha"],"panel_verified":st["panel_verified"],"model_frozen":st["model_frozen"],"cartridges":st["cartridges"],"cases":st["cases"],
                   "reference_test":"TEST596","reference_score":f"{ENG.REFERENCE['TEST596']['correct']}/{ENG.REFERENCE['TEST596']['total']}","run_count":st["run_count"]}
        def load_case(s,case=0,position="MIDDLE"):
            case=int(case);position=str(position).upper()
            with torch.no_grad():
                t0=time.perf_counter();r=ENG.demo_load_case(case,position);dt=time.perf_counter()-t0
            keys=s._keys(case,position);tgt=ENG.PANEL[case]["target"]
            return{"case":case,"position":position,"cartridges":[{"id":ci,"type":ENG.CAR[ci]["type"],"fact":ENG.CAR[ci]["fact"],"is_target":ci==tgt}for ci in keys],"question":r["question"],
                   "expected":ENG.CAR[tgt]["gold"],"events":[{"step":x["stage"],"cartridge_id":x["cartridge"],"cache_tokens":x["memory_tokens"],"old_prefix_unchanged":x["old_prefix_unchanged"]}for x in r["steps"]],
                   "elapsed_seconds":round(dt,3),"weights_updated":not r["weight_fingerprint_unchanged"],"source_replay_during_consolidation":bool(r["source_text_supplied_at_query"]),
                   "prior_memory_recomputed":not r["append_only_verified"],"engine_result":r}
        def ask(s,question=None):
            with torch.no_grad():
                t0=time.perf_counter();row=ENG.demo_ask(question);dt=time.perf_counter()-t0
            return{"question":row["question"],"answer":row["answer"],"normalized":row["normalized"],"reference_question":bool(row["is_locked_panel_question"]),"expected":row["gold"],
                   "correct":row["correct"],"elapsed_seconds":round(dt,3),"source_text_supplied_to_query":False,"model_weights_changed":False,"engine_row":row}
        def run_reference_panel(s,progress=None):
            def _p(d):
                if progress:progress(d["done"],72,{"case":d["case"],"position":d["slot"],"answer":"…","correct":d["correct"]})
            with torch.no_grad():rec=ENG.demo_verify_all(progress=_p,save=False)
            rows=[{"case":r["case"],"position":r["slot"],"answer":r["answer"],"expected":r["gold"],"correct":bool(r["correct"])}for r in rec["rows"]]
            return{"score":rec["correct"],"total":rec["total"],"matches_TEST596":bool(rec["matches_test596_reference"]),"panel_sha":rec["panel_sha"],"rows":rows,"elapsed_seconds":rec["seconds"],
                   "engine_payload_sha256":rec["payload_sha256"]}
    DEMO_ENGINE=DeepSeekDC3();torch.cuda.synchronize();ENGINE_INIT_SECONDS=time.perf_counter()-_t
# ---------------- measurement adapter (read-only with respect to the engine; never alters writing, consolidation or answering) ----------------
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
    """No forward hooks or pre-hooks left on any decoder layer or attention module (same check as the engine's _assert_clean)."""
    out=[]
    for li,layer in enumerate(model.model.layers):
        for label,module in(("layer",layer),("attention",getattr(layer,"self_attn",None))):
            if module is not None and(module._forward_hooks or module._forward_pre_hooks):out.append(f"L{li}:{label}")
    return out
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values())+list(ENG.__dict__.values()))
def _tensor_bytes(x):
    x=x.detach().contiguous().cpu()
    if x.dtype==torch.bfloat16:x=x.view(torch.uint16)
    return x.numpy().tobytes()
def memory_sha(layers,rows=None):
    """SHA-256 over the raw bytes of every cache tensor (latent and rotary; optionally only the first `rows` positions)."""
    h=hashlib.sha256()
    for k,v in layers:
        if rows is not None:k=k[:,:,:rows,:];v=v[:,:,:rows,:]
        h.update(_tensor_bytes(k));h.update(_tensor_bytes(v))
    return h.hexdigest()
class Instrument:
    """Thin adapter around the engine's demo API, plus measurement. Every scientific quantity comes from the engine itself."""
    def __init__(self,D,init_seconds):
        self.D=D;self.kind="ENGINE";m=D.model;c=m.config
        self.info=dict(model_id=ENG.MODEL,revision=ENG.REV,arch=[ENG.NL,ENG.H,ARCH[2],ARCH[3]],dtype=str(next(m.parameters()).dtype).replace("torch.",""),
            attn=getattr(c,"_attn_implementation",None),heads=getattr(c,"num_attention_heads",None),kv_lora_rank=getattr(c,"kv_lora_rank",None),qk_rope_head_dim=getattr(c,"qk_rope_head_dim",None),
            routed_experts=getattr(c,"n_routed_experts",None),experts_per_token=getattr(c,"num_experts_per_tok",None),shared_experts=getattr(c,"n_shared_experts",None),
            gpu=torch.cuda.get_device_name(0),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,python=sys.version.split()[0],platform=platform.platform(),
            params=sum(p.numel()for p in m.parameters()),rope=str(getattr(c,"rope_parameters",None)or getattr(c,"rope_theta",None)),engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256,
            engine_blob=ENGINE_BLOB,engine_exec_sha256=ENGINE_EXEC_SHA256,init_seconds=init_seconds,startup_utc=STARTUP_UTC,weight_sha0=None,
            sentinel_method="engine weight fingerprint (596.py fingerprint(): SHA-256 over 8 slices × 128 values of the first parameter of layers 0/3/6/12/20/26)",
            guard_method="all-parameter guard: SHA-256 over (name, shape, dtype, sum, |sum|, sum of squares) of every parameter tensor",
            full_weight_method="model weights pinned by Hugging Face revision "+ENG.REV,
            forward_counter="recording-only forward pre-hook on the causal-LM module, attached only during an engine API call",
            observer="write/append observer: wraps the engine's own write_cartridge/init_cartridge/append_cartridge during load_case() only; calls them unchanged; compares prior cache rows bitwise")
        self.info["sentinel0"]=self.sentinel();self.info["sentinel_engine_load"]=ENG.FP0;self.info["guard0"]=self.guard();self.info["hooks0"]=hook_stats(m)[0]
    # ---- read-only views of engine state ----
    def config(self):
        D=self.D
        return dict(MODEL_ID=ENG.MODEL,REVISION=ENG.REV,CUT=ENG.CUT,MAX_NEW=ENG.MAX_NEW,SEED=ENG.SEED,PANEL_SEED=ENG.PANEL_SEED,EXPECTED_PANEL_SHA=ENG.EXPECTED_PANEL_SHA,panel_sha=ENG.PANEL_SHA,
                    prefix_tokens=ENG.P,n_cartridges=len(ENG.CAR),n_cases=len(ENG.PANEL),positions=list(ENG.POSITIONS),system=ENG.SYSTEM,arch=[ENG.NL,ENG.H,ARCH[2],ARCH[3]],
                    source_blobs=dict(ENG.SOURCES),engine_gates=dict(ENG.GATES),
                    engine_sha256=hashlib.sha256(ENGINE_SRC.encode("utf-8")).hexdigest(),engine_blob=git_blob_sha1(ENGINE_SRC.encode("utf-8")),engine_exec_sha256=ENGINE_EXEC_SHA256)
    def status(self):return self.D.status()
    def cases(self):return self.D.cases()
    def keys(self,case,position):return list(self.D._keys(int(case),str(position)))
    def car(self,ci):c=ENG.CAR[ci];return dict(id=ci,world=c["world"],type=c["type"],fact=c["fact"],gold=c["gold"],body_len=int(c["body_len"]))
    def target(self,case):return int(ENG.PANEL[case]["target"])
    def source_ids(self,ci):return list(ENG.CAR[ci]["full"])
    def query_ids(self,q):return[int(t)for t in ENG.ids(ENG.question(q))[0].tolist()]
    def decode_each(self,ids):return[ENG.tok.decode([t])for t in ids]
    def panel_strings(self):return sorted({w for row in ENG.BASE for w in row})
    # ---- integrity ----
    def sentinel(self):return ENG.fingerprint()
    def guard(self):
        h=hashlib.sha256()
        with torch.inference_mode():
            for name,p in self.D.model.named_parameters():
                x=p.detach().float();vals=(float(x.sum().item()),float(x.abs().sum().item()),float((x*x).sum().item()))
                h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode());h.update(np.asarray(vals,dtype=np.float64).tobytes());del x
        torch.cuda.empty_cache();return h.hexdigest()
    def full_weight_sha(self):return None
    def frozen(self):
        m=self.D.model;tot,fo=hook_stats(m)
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
    def load(self,case,position):
        D=self.D;G=ENG.__dict__;rec,_rec=self._recorder();writes={};steps=[]
        orig={k:G[k]for k in("write_cartridge","init_cartridge","append_cartridge")}
        def w_obs(ci):
            a=len(rec);t=time.perf_counter();res=orig["write_cartridge"](ci);torch.cuda.synchronize();writes[int(ci)]=dict(fw=(a,len(rec)),seconds=time.perf_counter()-t);return res
        def i_obs(ci):
            a=len(rec);t=time.perf_counter();res=orig["init_cartridge"](ci);torch.cuda.synchronize();mem=res[0]
            steps.append(dict(kind="init",cartridge_id=int(ci),old_len=0,new_len=int(mem[0][0].shape[-2]),fw=(a,len(rec)),prior_rows_bitwise_unchanged=None,prior_sha_before=None,prior_sha_after=None,
                              memory_sha=memory_sha(mem),seconds=time.perf_counter()-t,engine_hook_calls=int(res[1]),engine_init_delta=float(res[2])));return res
        def a_obs(memory,ci):
            a=len(rec);oldn=int(memory[0][0].shape[-2]);sb=memory_sha(memory);t=time.perf_counter();res=orig["append_cartridge"](memory,ci);torch.cuda.synchronize();dt_=time.perf_counter()-t;mem=res[0]
            same=all(torch.equal(mem[l][0][:,:,:oldn],memory[l][0])and torch.equal(mem[l][1][:,:,:oldn],memory[l][1])for l in range(len(memory)))
            steps.append(dict(kind="append",cartridge_id=int(ci),old_len=oldn,new_len=int(mem[0][0].shape[-2]),fw=(a,len(rec)),prior_rows_bitwise_unchanged=bool(same),
                              prior_sha_before=sb,prior_sha_after=memory_sha(mem,oldn),memory_sha=memory_sha(mem),seconds=dt_,engine_hook_calls=int(res[1]),engine_unchanged_flag=bool(res[2])));return res
        h=D.model.register_forward_pre_hook(_rec,with_kwargs=True);G["write_cartridge"]=w_obs;G["init_cartridge"]=i_obs;G["append_cartridge"]=a_obs
        try:torch.cuda.synchronize();t=time.perf_counter();setup=D.load_case(case,position);torch.cuda.synchronize();dt_=time.perf_counter()-t
        finally:
            h.remove()
            for k,v in orig.items():G[k]=v
        for s in steps:
            w=writes.get(s["cartridge_id"])
            if w is None:raise AuditFail(f"no write event recorded for cartridge #{s['cartridge_id']}")
            s["forwards"]=rec[w["fw"][0]:w["fw"][1]]+rec[s["fw"][0]:s["fw"][1]];s["write_seconds"]=w["seconds"];del s["fw"]
        setup.pop("engine_result",None)
        return setup,dict(steps=steps,forwards=rec),dt_
    def ask(self,q=None):
        D=self.D;rec,_rec=self._recorder();h=D.model.register_forward_pre_hook(_rec,with_kwargs=True)
        try:torch.cuda.synchronize();t=time.perf_counter();res=D.ask(q);torch.cuda.synchronize();dt_=time.perf_counter()-t
        finally:h.remove()
        res.pop("engine_row",None)
        return res,rec,dt_
    def memory_len(self):M=self.D._memory;return int(M[0][0].shape[-2])if M is not None else 0
    def memory_sha_now(self):return memory_sha(self.D._memory)
    def memory_profile(self):
        """Display-only statistics of the consolidated MLA memory (L2 norm of every latent and rotary row, per layer)."""
        M=self.D._memory;kn=[];vn=[];nb=0
        with torch.inference_mode():
            for k,v in M:
                kn.append([float(x)for x in k.float().norm(dim=-1).mean(dim=1)[0].tolist()]);vn.append([float(x)for x in v.float().norm(dim=-1).mean(dim=1)[0].tolist()])
                nb+=k.numel()*k.element_size()+v.numel()*v.element_size()
        return dict(layers=len(M),T=len(kn[0]),k_norm=kn,v_norm=vn,bytes=int(nb),kv_heads=int(M[0][0].shape[1]),head_dim=int(M[0][0].shape[-1]),rope_dim=int(M[0][1].shape[-1]),
                    dtype=str(M[0][0].dtype).replace("torch.",""))
    def replay(self,progress=None):return self.D.run_reference_panel(progress)
    def sync(self):torch.cuda.synchronize()
    def reset_peak(self):torch.cuda.reset_peak_memory_stats()
    def peak_gib(self):return torch.cuda.max_memory_allocated()/2**30
say("[4/5] Measurement adapter · engine weight fingerprint · all-parameter guard …")
INSTR=Instrument(DEMO_ENGINE,ENGINE_INIT_SECONDS)
_cfg=INSTR.config();_st=INSTR.status();_fz=INSTR.frozen();_ref=ENG.REFERENCE["TEST596"]
ENGINE_LOCK=[("596.py git blob = repository file",_cfg["engine_blob"]==ENGINE_BLOB_EXPECTED),("596.py SHA-256 = embedded reference",_cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED),
 ("engine sources verified: 560.py and 590.py git blobs",_cfg["source_blobs"]==ENGINE_SOURCES_EXPECTED),("engine gates all PASS",bool(_cfg["engine_gates"])and all(_cfg["engine_gates"].values())),
 ("model = deepseek-ai/DeepSeek-V2-Lite-Chat at the pinned revision",_cfg["MODEL_ID"]==MODEL_ID and _cfg["REVISION"]==MODEL_REVISION),
 (f"architecture {ARCH[0]} layers · hidden {ARCH[1]}",tuple(INSTR.info["arch"])==ARCH==tuple(_cfg["arch"])and len(DEMO_ENGINE.model.model.layers)==ARCH[0]),
 ("consolidation cut = layer 3",_cfg["CUT"]==CUT_EXPECTED),("chat-template instruction prefix = 28 tokens",_cfg["prefix_tokens"]==PREFIX_EXPECTED),
 ("locked panel SHA-256",_cfg["panel_sha"]==EXPECTED_PANEL_SHA==_cfg["EXPECTED_PANEL_SHA"]),("96 cartridges · 24 cases · FIRST/MIDDLE/LAST",_cfg["n_cartridges"]==96 and _cfg["n_cases"]==N_CASES and tuple(_cfg["positions"])==POSITIONS),
 ("engine reference = TEST596 69/72 with the three recorded failures",_ref["correct"]==69 and _ref["total"]==72 and _ref["by_slot"]=={p:TEST596["score"][p]for p in POSITIONS}
     and {tuple(x)for x in _ref["failures"]}=={(c,p)for c,p,_,_ in TEST596["failures"]}),
 ("adapter status: TEST596 reference 69/72",_st.get("reference_test")=="TEST596"and _st.get("reference_score")=="69/72"),
 ("engine weight fingerprint = value at model load",_fz["sentinel"]==INSTR.info["sentinel0"]==INSTR.info["sentinel_engine_load"]),("trainable tensors = 0",_fz["trainable_tensors"]==0),
 ("model in eval mode, no LoRA, no optimizer",(not _fz["training"])and not _fz["lora"]and not _fz["optimizer"]),("no foreign forward/backward hooks",_fz["foreign_hooks"]==0),
 ("no residual hooks on decoder layers or attention",not _fz["residual_hooks"])]
_bad=[n for n,ok in ENGINE_LOCK if not ok]
if _bad:raise RuntimeError(f"ENGINE LOCK FAILED: {_bad}")
say(f"[5/5] Engine lock PASS · {len(ENGINE_LOCK)} checks · init {ENGINE_INIT_SECONDS:.1f}s · guard {INSTR.info['guard0'][:24]}… · fingerprint {INSTR.info['sentinel0'][:24]}…")
say("PART 1 / 3 complete. Now run PART 2 / 3 in the next cell.");say("="*140)
MAM_DSK_PART1_OK=True
#<<ENGINE_END>>
# =====================================================================================================================================
# AKBASCORE MAM · PERSISTENT NUMERICAL MEMORY FOR FROZEN LANGUAGE MODELS — DEEPSEEK-V2-LITE-CHAT · PART 2 / 3 · MEASUREMENT PIPELINE AND FIGURES
# Same demo as PART 1. Run this cell after PART 1 / 3, in the same Colab runtime; then run PART 3 / 3.
# Defines the fail-closed measurement run (write → append-only consolidation → source-free question → verification → sealed payload,
# logs, ZIP), six figures (JPEG 300 dpi + vector PDF) and the optional live replay of the locked panel. Nothing here changes the engine.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 10 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_DSK_PART1_OK"):raise RuntimeError("PART 1 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<CORE_BEGIN>>
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path("/content/AKBASCORE_MAM_DEEPSEEK_DC3")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_DEEPSEEK_DC3")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
N_STAGES=8
def ev(stage,title,body):return dict(stage=stage,title=title,body=body)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
RUN_PREFIX="MAMDSK-";REPLAY_PREFIX="MAMDSKREPLAY-"
def prune_runs(keep=4,prefix=RUN_PREFIX):
    runs=sorted([p for p in ROOT.glob(prefix+"*")if p.is_dir()],key=lambda p:p.stat().st_mtime)
    for p in runs[:max(0,len(runs)-keep)]:shutil.rmtree(p,ignore_errors=True)
def segments(steps,p):
    """Row ranges of the consolidated memory: shared prefix (written with cartridge 1) and the body rows of cartridges 1..5."""
    seg=[("prefix",0,p,None)];prev=p
    for n_,s in enumerate(steps,1):seg.append((f"C{n_}",prev,s["new_len"],s["cartridge_id"]));prev=s["new_len"]
    return seg
def post_seal_audit(imgs,pdfs,zp,member_names,pp,sha,tp,mp,verdict):
    checks=[]
    def ok(name,cond):
        checks.append(name)
        if not cond:raise AuditFail("POST-SEAL AUDIT FAILED: "+name)
    ok(f"{N_FIGS}/{N_FIGS} figures created (JPEG)",len(imgs)==N_FIGS and all(file_ready(p)for p,_ in imgs));ok(f"{N_FIGS}/{N_FIGS} vector PDF copies",len(pdfs)==N_FIGS and all(file_ready(p)for p in pdfs))
    bad=[]
    for p,_ in imgs:
        with Image.open(p)as im:
            if not(im.format=="JPEG"and im.mode=="RGB"and im.size[0]>=3600):bad.append(p.name)
    ok(f"{N_FIGS}/{N_FIGS} figures are JPEG/RGB at ≥3600 px width",not bad)
    with zipfile.ZipFile(zp)as z:
        ok("ZIP testzip()",z.testzip()is None);nm=z.namelist()
        ok("ZIP is flat (no sub-folders)",all("/"not in n for n in nm));ok("ZIP contents = expected package",sorted(nm)==sorted(member_names))
        jp=sorted(n for n in nm if n.lower().endswith(".jpg"));ok("figure numbering 01..%02d"%N_FIGS,[n[4:6]for n in jp]==[f"{i:02d}"for i in range(1,N_FIGS+1)])
    ok("payload SHA-256 recomputation",hashlib.sha256(pp.read_bytes()).hexdigest()==sha);ok("payload JSON parses",isinstance(json.loads(pp.read_bytes().decode("utf-8")),dict))
    m=json.loads(mp.read_text(encoding="utf-8"));ok("manifest hash and verdict match payload",m["payload_sha256"]==sha and m["verdict"]==verdict)
    ok("readable log exists and is non-empty",file_ready(tp));ok("ZIP exists and is non-empty",file_ready(zp))
    return checks
def fw_txt(f):
    if f["kind"]=="ids":return f"token ids n={f['n']}"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
    return f"embeddings n={f['n']} ({'all zero'if f['embeds_zero']else'NON-ZERO'})"+(f" · cache {f['past_len']}"if f["has_past"]else" · no cache")
def make_txt(P,sha,names,sealed_utc):
    o=[];a=o.append;S="="*140;Dd="-"*140;E_=P["environment"];L=P["live"];cfg=P["engine"]["config"];fz=P["frozen"];ans=L["answer"]
    a(S);a(f"{DEMO_TITLE} — READABLE RUN LOG");a(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");a(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");a(S)
    a(PRIORITY);a(f"{PARADIGM}.")
    a(f"Derived from {names['payload']} (SHA-256 {sha}). Manifest: {names['manifest']}.")
    a("This is a LIVE run of the frozen DeepSeek-V2-Lite-Chat incremental MLA DC3 engine (596.py, unchanged). It is not TEST596; the TEST596 reference values are reproduced verbatim, never recomputed.")
    a(CROSS_TITLE+". "+CROSS_TEXT);a(CROSS_RESULT);a(CROSS_BOUNDARY);a("Earlier records (references only): "+CROSS_REF)
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("VERDICT",P["verdict"]),
        ("MODEL",f"{cfg['MODEL_ID']} · revision {cfg['REVISION']} · frozen · {P['model']['dtype']} · attention {P['model']['attn']}"),("ARCHITECTURE",arch_txt(tuple(P["model"]["arch"]))),
        ("GPU",f"{E_['gpu']} (TEST596 reference GPU: {CANON_GPU}; same: {P['hardware']['same']})"),("TORCH / TRANSFORMERS / GRADIO",f"{E_['torch']} / {E_['transformers']} / {E_['gradio']}"),
        ("ENGINE FILE",f"{P['engine']['file']} · git blob {P['engine']['blob']} · SHA-256 {P['engine']['sha256']} (executed unchanged)"),
        ("ENGINE SOURCES",", ".join(f"{k} blob {v}"for k,v in cfg["source_blobs"].items())+" (verified by the engine)"),
        ("CONSOLIDATION CUT",f"layer {cfg['CUT']} (cartridge = H{cfg['CUT']} residual + layers 0–{cfg['CUT']} {CACHE_SHORT}; {upper_txt(cfg['CUT'],cfg['arch'][0])} consolidate)"),
        ("PANEL",f"locked 24-case panel (shared with the earlier records) · SHA-256 {cfg['panel_sha']}"),("TEST596 REFERENCE",f"INCR_DC3_MLA {TEST596['score']['TOTAL']}/72 · {TEST596['source']}"),
        ("TEST597 LOG SHA-256",TEST597["log_sha256"]),("SCORING",CORRECT_RULE_TXT)):a(f"{k_:<30}: {v}")
    a("");a("CASE");a(Dd)
    a(f"case {L['case']:02d} · target position {L['position']} (slot {L['target_slot']+1} of 5) · question: {L['question']}");a(f"expected answer (audit metadata): {L['expected']}")
    for n_,c in enumerate(L["cartridges"],1):a(f"C{n_} · cartridge #{c['id']:02d} · {c['type']:<8} · {c['body_len']:>2} body tokens · {c['fact']}"+("   ← TARGET"if c["is_target"]else""))
    a("");a("WRITE + APPEND-ONLY CONSOLIDATION · load_case()  [LIVE]");a(Dd)
    for n_,s in enumerate(L["steps"],1):
        a(f"step {n_} · {s['kind']:<6} · cartridge #{s['cartridge_id']:02d} · memory {s['old_len']} → {s['new_len']} tokens (+{s['new_len']-s['old_len']}) · {1000*s['seconds']:.1f} ms")
        for f in s["forwards"]:a(f"         forward: {fw_txt(f)}")
        if s["kind"]=="append":a(f"         previously consolidated rows 0…{s['old_len']-1}: bitwise unchanged = {s['prior_rows_bitwise_unchanged']} · SHA {s['prior_sha_before'][:16]}… → {s['prior_sha_after'][:16]}…")
    a(f"final memory: {L['memory']['T']} tokens × {L['memory']['layers']} layers · MLA latent {L['memory']['head_dim']} + rotary {L['memory']['rope_dim']} · {L['memory']['bytes']/2**20:.2f} MiB ({L['memory']['dtype']}) · SHA {L['memory_sha'][:24]}…")
    a(f"source token-id forwards during load: {L['source_reads']} (one per cartridge, each its own fact; the engine performs the five writes before consolidation) · consolidation forwards with all-zero input embeddings: {L['zero_embed_forwards']}")
    a("");a("SOURCE-FREE QUESTION · ask()  [LIVE]");a(Dd)
    a(f"question: {ans['question']}");a(f"answer  : {ans['answer']}");a(f"expected: {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'} (re-derived: {L['correct_rederived']})")
    a(f"query input = question template only: {L['query_tokens']} token ids over the installed memory ({L['query_past']} tokens) · generation forwards {L['gen_forwards']} · {1000*ans['elapsed_seconds']:.0f} ms")
    a(f"repeat ask(): identical answer = {L['repeat_identical']} · memory SHA unchanged by querying = {L['memory_unchanged_by_query']}")
    ro=L["reference_outcome"];a(f"TEST596 reference for this case and position: {'correct' if ro['correct'] else 'FAILED with answer '+str(ro['answer'])} · live outcome identical: {L['matches_reference']} (reported, not a gate)")
    if L.get("custom"):cu=L["custom"];a(f"free-text question (unscored): {cu['question']} → {cu['answer']} · query tokens {cu['query_tokens']}")
    a("");a("TEST596 REFERENCE · DEEPSEEK-V2-LITE-CHAT (archived; not produced by this run)");a(Dd)
    v=TEST596["score"];a(f"INCR_DC3_MLA  FIRST {v['FIRST']:>2}/24 · MIDDLE {v['MIDDLE']:>2}/24 · LAST {v['LAST']:>2}/24 · TOTAL {v['TOTAL']}/72 · {ARM_DESC['INCR_DC3_MLA']}")
    for c_,p_,g_,w_ in TEST596["failures"]:a(f"failed case {c_:02d} {p_:<6} · expected {g_} · answered {w_}")
    a(f"TEST597 (three failed slots, failures reproduced: {TEST597['failures_reproduced']}): JOINT {TEST597['joint'][0]}/{TEST597['joint'][1]} · INCR_DC3_MLA {TEST597['incr_on_failing'][0]}/{TEST597['incr_on_failing'][1]} · NATIVE_INDEP_MLA {TEST597['native_indep'][0]}/{TEST597['native_indep'][1]} · readout fix verified: {TEST597['readout_fix_verified']}")
    a(f"TEST597 log SHA-256 {TEST597['log_sha256']} · program git blob {TEST597['program_blob']} · decision {TEST597['decision']}")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("sentinel_startup","sentinel_before","sentinel_after","guard_startup","guard_before","guard_after"):a(f"{k_:<18}: {fz[k_]}")
    a(f"trainable tensors={fz['trainable_tensors']} · lora={fz['lora']} · optimizer={fz['optimizer']} · training_mode={fz['training_mode']} · foreign hooks outside API calls {fz['foreign_hooks_before']}→{fz['foreign_hooks_after']} · residual hooks on layers/attention {len(fz['residual_hooks_before'])}→{len(fz['residual_hooks_after'])}")
    a(fz["full_weight_method"]);a(fz["guard_method"]);a(fz["sentinel_method"]);a(P["instrumentation"]["forward_counter"]);a(P["instrumentation"]["observer"])
    for t_,k_ in(("DEMONSTRATED IN THIS RUN","demonstrated"),("ARCHIVED","archived"),("NOT ESTABLISHED","not_established"),("TECHNICAL NOTES","notes")):
        a("");a(t_);a(Dd)
        for t in P["scope"][k_]:a("• "+t)
    a("");a("PRE-SEAL CHECKS");a(Dd)
    for c_ in P["checks_pre_seal"]:a("PASS · "+c_)
    a("");a("TIMING (seconds)");a(Dd)
    for k_,v in P["timing"].items():a(f"{k_:<24}: {v:.3f}")
    a(f"peak GPU memory {P['gpu']['peak_allocated_gib']:.2f} GiB");a("");a(f"Raw forward records and the memory profile are in {names['payload']} and {names['jsonl']}.");a(S)
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
def _execute_run(I,ctl,state):
    run_id=ctl["run_id"];run_dir=Path(ctl["run_dir"]);case=int(ctl["case"]);position=str(ctl["position"]).upper();custom=(ctl.get("custom")or"").strip()
    info=I.info;checks=[];tm=Counter();T0=time.perf_counter();state["checks"]=checks;raw=dict(setup=None,trace=None,answer=None,repeat=None,custom=None);state["raw"]=raw
    def chk(name,cond):
        checks.append(name)
        if not cond:raise AuditFail(name)
    run_start_utc=utc_now();say("="*140);say(f"RUN {run_id} · case {case:02d} · target position {position}");say("="*140)
    # ---- 1 integrity ----
    yield ev(1,"Integrity","Engine source hash, locked panel, consolidation cut, weight sentinel, all-parameter guard, hooks.")
    I.reset_peak();cfg=I.config();st0=I.status();fz0=I.frozen();t=time.perf_counter();g_before=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("596.py git blob and SHA-256 = embedded reference · engine sources verified",cfg["engine_blob"]==ENGINE_BLOB_EXPECTED and cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED and cfg["source_blobs"]==ENGINE_SOURCES_EXPECTED)
    chk(f"model at pinned revision, architecture, consolidation cut (layer {CUT_EXPECTED}) and {PREFIX_EXPECTED}-token prefix as in TEST596",cfg["MODEL_ID"]==MODEL_ID and cfg["REVISION"]==MODEL_REVISION and tuple(cfg["arch"])==ARCH and cfg["CUT"]==CUT_EXPECTED and cfg["prefix_tokens"]==PREFIX_EXPECTED)
    chk("locked panel (SHA-256) · 96 cartridges · 24 cases",cfg["panel_sha"]==EXPECTED_PANEL_SHA and cfg["n_cartridges"]==96 and cfg["n_cases"]==N_CASES)
    chk("engine gates all PASS (model frozen, panel SHA)",bool(cfg["engine_gates"])and all(cfg["engine_gates"].values()))
    chk("weight sentinel before run = value at initialisation",fz0["sentinel"]==info["sentinel0"]);chk("all-parameter guard before run = value at startup",g_before==info["guard0"])
    chk("trainable tensors = 0, eval mode, no LoRA, no optimizer",fz0["trainable_tensors"]==0 and not fz0["training"]and not fz0["lora"]and not fz0["optimizer"])
    chk("no foreign forward/backward hooks outside API calls · no residual hooks on layers/attention",fz0["foreign_hooks"]==0 and not fz0["residual_hooks"])
    chk("case in 0..23 and position in FIRST/MIDDLE/LAST",0<=case<N_CASES and position in POSITIONS)
    keys=I.keys(case,position);tgt=I.target(case);slot=TARGET_SLOT[position]
    chk("five cartridges; target cartridge at the selected position",len(keys)==N_CART and keys[slot]==tgt and len(set(keys))==N_CART)
    cars=[dict(I.car(ci),is_target=ci==tgt)for ci in keys];expected=I.car(tgt)["gold"];strip=I.panel_strings();state["strip"]=strip
    for c in checks:say("   PASS ·",c)
    # ---- 2 write + append-only consolidation ----
    yield ev(2,"Write + append-only consolidation","load_case(): five independent cartridges are written and appended one by one; every model forward is recorded and prior rows are compared bitwise.")
    setup,trace,dt=I.load(case,position);raw["setup"]=setup;raw["trace"]=trace;tm["load_case"]+=dt;steps=trace["steps"];P_=cfg["prefix_tokens"]
    chk("load_case(): five consolidation steps in the selected order (one init + four appends)",len(steps)==N_CART and [s["cartridge_id"]for s in steps]==keys and steps[0]["kind"]=="init"and all(s["kind"]=="append"for s in steps[1:]))
    chk("engine setup echoes the same cartridges, question and expected answer",[c["id"]for c in setup["cartridges"]]==keys and setup["expected"]==expected and setup["position"]==position and setup["case"]==case)
    s0=steps[0];f0=s0["forwards"];src0=I.source_ids(keys[0])
    chk("step 1: write = one token-id forward of cartridge 1's own fact (no cache); consolidation = one all-zero-embedding forward over the same rows",
        len(f0)==2 and f0[0]["kind"]=="ids"and f0[0]["ids"]==src0 and not f0[0]["has_past"]and f0[1]["kind"]=="embeds"and f0[1]["embeds_zero"]and f0[1]["n"]==len(src0)and not f0[1]["has_past"]and s0["new_len"]==len(src0))
    for n_,s in enumerate(steps[1:],2):
        f=s["forwards"];src=I.source_ids(s["cartridge_id"]);q=len(src)-P_
        chk(f"step {n_}: write = one token-id forward of cartridge {n_}'s own fact only (no cache)",len(f)==2 and f[0]["kind"]=="ids"and f[0]["ids"]==src and not f[0]["has_past"])
        chk(f"step {n_}: consolidation forward = all-zero embeddings for the {q} new rows only, over the existing {s['old_len']}-token memory",f[1]["kind"]=="embeds"and f[1]["embeds_zero"]and f[1]["n"]==q and f[1]["has_past"]and f[1]["past_len"]==s["old_len"]==steps[n_-2]["new_len"])
        chk(f"step {n_}: previously consolidated rows 0…{s['old_len']-1} bitwise unchanged after the append",s["prior_rows_bitwise_unchanged"]is True and s["prior_sha_before"]==s["prior_sha_after"])
        chk(f"step {n_}: memory grows by exactly the new rows (+{q})",s["new_len"]==s["old_len"]+q)
    ids_fw=[f for f in trace["forwards"]if f["kind"]=="ids"];emb_fw=[f for f in trace["forwards"]if f["kind"]=="embeds"]
    chk("no source text replayed: exactly five token-id forwards during load, each the own fact of the cartridge being written, each once",
        len(ids_fw)==N_CART and [f["ids"]for f in ids_fw]==[I.source_ids(ci)for ci in keys]and len(trace["forwards"])==2*N_CART)
    chk("adapter flags: weights not updated · no source replay in consolidation · prior memory not recomputed",setup["weights_updated"]is False and setup["source_replay_during_consolidation"]is False and setup["prior_memory_recomputed"]is False)
    chk("engine event log: cache sizes match the measured memory growth",[e["cache_tokens"]for e in setup["events"]]==[s["new_len"]for s in steps])
    T=I.memory_len();chk("final memory length = prefix + all five body lengths",T==steps[-1]["new_len"]==P_+sum(c["body_len"]for c in cars))
    mem_sha=I.memory_sha_now();chk("final memory SHA-256 = SHA-256 recorded after the last append",mem_sha==steps[-1]["memory_sha"])
    prof=I.memory_profile();chk(f"memory profile: {ARCH[0]} layers × T rows · MLA latent {ARCH[2]} + rotary {ARCH[3]}",prof["layers"]==ARCH[0] and prof["T"]==T and prof["kv_heads"]==1 and prof["head_dim"]==ARCH[2] and prof["rope_dim"]==ARCH[3])
    chk("engine step flags: one H3 injection per step, init identical to the cartridge's own cache, append-only flag True",all(s["engine_hook_calls"]==1 for s in steps)and steps[0]["engine_init_delta"]==0.0 and all(s["engine_unchanged_flag"]for s in steps[1:]))
    say(f"   memory {T} tokens · 5 steps · prior rows bitwise unchanged at every append · {1000*dt:.0f} ms")
    # ---- 3 source-free question ----
    question=setup["question"]
    yield ev(3,"Source-free question","ask(): the question template is processed over the numerical memory; the source text is not given to the model.")
    a1,fw_a,dt=I.ask(None);raw["answer"]=a1;tm["ask"]+=dt;qids=I.query_ids(question)
    chk("query forward input = question-template tokens only, over the installed memory",len(fw_a)>=1 and fw_a[0]["kind"]=="ids"and fw_a[0]["ids"]==qids and fw_a[0]["has_past"]and fw_a[0]["past_len"]==T)
    chk("generation forwards: one new token each, over the growing cache",all(f["kind"]=="ids"and f["n"]==1 and f["has_past"]and f["past_len"]==T+len(qids)+n_ for n_,f in enumerate(fw_a[1:])))
    chk("ask(): reference question of this case; engine flags: no source text, weights unchanged",a1["question"]==question and a1["reference_question"]is True and a1["expected"]==expected and a1["source_text_supplied_to_query"]is False and a1["model_weights_changed"]is False)
    cr=correct_rule(a1["answer"],expected);chk("correctness re-derived from the answer text (exact normalized name match) = engine result",cr==a1["correct"])
    a2,fw_a2,dt=I.ask(None);raw["repeat"]=a2;tm["ask_repeat"]+=dt
    chk("repeat ask(): same forward pattern and identical answer (greedy decoding)",a2["answer"]==a1["answer"]and fw_a2[0]["ids"]==qids and len(fw_a2)==len(fw_a))
    sha_q=I.memory_sha_now();chk("memory unchanged by querying (SHA-256 before = after)",sha_q==mem_sha)
    say(f"   answer {a1['answer']!r} · expected {expected} · {'CORRECT'if a1['correct']else'INCORRECT'} · {len(qids)} query tokens over {T}-token memory")
    # ---- 4 optional free-text question ----
    yield ev(4,"Free-text question","Optional, unscored: a question typed by the visitor is answered from the same numerical memory.")
    cu=None
    if custom and custom!=question:
        a3,fw_a3,dt=I.ask(custom);raw["custom"]=a3;tm["ask_custom"]+=dt;cq=I.query_ids(custom);strip=strip+[v for x in(custom,str(a3["answer"]))for v in(x," ".join(x.split()),json.dumps(x,ensure_ascii=False)[1:-1])];state["strip"]=strip
        chk("free-text query forward input = question-template tokens only, over the installed memory",fw_a3[0]["kind"]=="ids"and fw_a3[0]["ids"]==cq and fw_a3[0]["past_len"]==T)
        chk("free-text question is not scored",a3["reference_question"]is False and a3["correct"]is None)
        chk("memory unchanged after the free-text question",I.memory_sha_now()==mem_sha)
        cu=dict(question=custom,answer=a3["answer"],query_tokens=len(cq),gen_forwards=len(fw_a3)-1,seconds=a3["elapsed_seconds"],forwards=fw_a3)
    # ---- 5 verification ----
    yield ev(5,"Verification","Weight sentinel, all-parameter guard, trainable tensors and hooks after the run.")
    fz1=I.frozen();t=time.perf_counter();g_after=I.guard();tm["guard"]+=time.perf_counter()-t
    chk("all-parameter guard after run = before run = startup (weights unchanged)",g_after==g_before==info["guard0"]);chk("weight sentinel after run = value at initialisation",fz1["sentinel"]==info["sentinel0"])
    chk("trainable tensors = 0, no foreign hooks and no residual hooks after run",fz1["trainable_tensors"]==0 and fz1["foreign_hooks"]==0 and not fz1["residual_hooks"])
    tm["engine_total"]=time.perf_counter()-T0
    verdict=f"CASE_{case:02d}_{position}_{'CORRECT'if a1['correct']else'INCORRECT'}_APPEND_ONLY_VERIFIED_SOURCE_FREE_QUERY"
    # ---- 6 seal ----
    yield ev(6,"Sealing","Payload, manifest, readable log and raw records are written and hashed.")
    peak=I.peak_gib();same_hw=info["gpu"]==CANON_GPU;ref_out=reference_outcome(case,position);ref_match=(bool(a1["correct"])==ref_out["correct"])and(ref_out["correct"]or norm_ans(a1["answer"])==norm_ans(ref_out["answer"]))
    live=dict(case=case,position=position,target_slot=slot,cartridges=cars,question=question,expected=expected,setup=setup,steps=steps,forwards_load=trace["forwards"],
              source_reads=len(ids_fw),zero_embed_forwards=sum(1 for f in emb_fw if f["embeds_zero"]),memory=prof,memory_sha=mem_sha,segments=segments(steps,P_),
              answer=a1,answer_repeat=a2,correct=bool(a1["correct"]),correct_rederived=bool(cr),repeat_identical=a2["answer"]==a1["answer"],memory_unchanged_by_query=sha_q==mem_sha,
              query_tokens=len(qids),query_token_text=I.decode_each(qids),query_past=T,gen_forwards=len(fw_a)-1,forwards_ask=fw_a,custom=cu,
              reference_outcome=ref_out,matches_reference=ref_match,
              session=[dict(r,correct=(bool(a1["correct"])if r.get("correct")is None else bool(r["correct"])))for r in(ctl.get("session")or[])])
    P={"schema":"akbascore.mam.deepseek.mla.dc3.demo.run.v1","demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"place":AUTHOR_PLACE,"copyright":COPYRIGHT,
       "license":{"name":LICENSE_NAME,"url":LICENSE_URL},"priority_record":PRIORITY,"paradigm":PARADIGM,"discovery":DISCOVERY,"core_message":CORE_MESSAGE,
       "cross_model":{"title":CROSS_TITLE,"statement":CROSS_TEXT,"result":CROSS_RESULT,"boundary":CROSS_BOUNDARY,"earlier_records":PRIOR_PUBS,"note":"references only; no value of the earlier records is used as a result here"},
       "run_id":run_id,"run_start_utc":run_start_utc,"run_end_utc":utc_now(),"verdict":verdict,
       "model":{"id":cfg["MODEL_ID"],"revision":cfg["REVISION"],"arch":info["arch"],"heads":info.get("heads"),"kv_lora_rank":info.get("kv_lora_rank"),"qk_rope_head_dim":info.get("qk_rope_head_dim"),
                "routed_experts":info.get("routed_experts"),"experts_per_token":info.get("experts_per_token"),"shared_experts":info.get("shared_experts"),"dtype":info["dtype"],"attn":info["attn"],"params":info["params"],"rope":info.get("rope")},
       "environment":{k_:info[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},"hardware":{"reference_gpu":CANON_GPU,"this_run_gpu":info["gpu"],"same":bool(same_hw)},
       "engine":{"file":info["engine_file"],"sha256":cfg["engine_sha256"],"blob":cfg["engine_blob"],"exec_sha256":cfg["engine_exec_sha256"],"config":cfg,"status_before":st0,"init_seconds":info["init_seconds"],
                 "instrument":I.kind,"adapter":"DeepSeekDC3: calls the engine's own demo_load_case / demo_ask / demo_verify_all (596.py, unchanged)","source":SOURCE_URL,"lock_program":LOCK_URL,"log":LOG_URL},
       "instrumentation":{"forward_counter":info["forward_counter"],"observer":info["observer"]},"live":live,
       "frozen":{"weights_pinned":info["full_weight_method"],"full_weight_method":info["full_weight_method"],"sentinel_startup":info["sentinel0"],"sentinel_before":fz0["sentinel"],"sentinel_after":fz1["sentinel"],"guard_startup":info["guard0"],"guard_before":g_before,"guard_after":g_after,
                 "trainable_tensors":fz1["trainable_tensors"],"lora":fz1["lora"],"optimizer":fz1["optimizer"],"training_mode":fz1["training"],"foreign_hooks_before":fz0["foreign_hooks"],
                 "foreign_hooks_after":fz1["foreign_hooks"],"residual_hooks_before":fz0["residual_hooks"],"residual_hooks_after":fz1["residual_hooks"],"guard_method":info["guard_method"],"sentinel_method":info["sentinel_method"]},
       "archived":{"test596":TEST596,"test597":TEST597,"arm_description":ARM_DESC,"note":"DeepSeek reference values; not produced or recomputed by this run. The joint-text and independent controls were run in TEST597 on the three failed slots only, not in this demo."},
       "scope":SCOPE,"checks_pre_seal":list(checks),"timing":dict(tm),"gpu":{"peak_allocated_gib":peak}}
    P=jsafe(P);allow=(info["gpu"],CANON_GPU)
    hits=claim_scan("payload",strip_words(json.dumps(P,ensure_ascii=False),strip),allow=allow)
    chk("claim-discipline and foreign-model scan of the payload: 0 hits",not hits);P["claim_scan"]={"rules":len(CLAIM_RULES)+1+len(LEAK_RULES),"hits":0,"scope":"payload JSON text; panel names and free text excluded; earlier-record citation exempt"}
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RAW_RECORDS.jsonl",images=f"figures_manifest_{run_id}.json",zip=f"AKBASCORE_MAM_DEEPSEEK_DC3_EVIDENCE_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"copyright":COPYRIGHT,"license":LICENSE_URL,"priority_record":PRIORITY,"run_id":run_id,"payload_file":payload_name,
              "payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":verdict,"engine_sha256":cfg["engine_sha256"],"engine_blob":cfg["engine_blob"],"test596_reference":f"{TEST596['score']['TOTAL']}/72","test597_log_sha256":TEST597["log_sha256"],
              "model":MODEL_ID,"revision":MODEL_REVISION,
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not a third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"run_id":run_id,"verdict":verdict,"case":case,"position":position,"question":question,"answer":a1["answer"],"expected":expected,"correct":a1["correct"],
             "cartridges":[{"slot":n_+1,"id":c["id"],"type":c["type"],"target":c["is_target"]}for n_,c in enumerate(cars)],
             "memory_growth":[s["new_len"]for s in steps],"prior_rows_bitwise_unchanged":[s["prior_rows_bitwise_unchanged"]for s in steps[1:]],"source_reads_during_load":len(ids_fw),
             "query_tokens":len(qids),"memory_tokens":T,"weights_unchanged":True,"engine_sha256":cfg["engine_sha256"],"payload_sha256":sha,"model":MODEL_ID,
             "reference_outcome_TEST596":ref_out,"live_outcome_matches_reference":ref_match,
             "reference_test596":dict(TEST596["score"]),"reference_failures":TEST596["failures"]}
    (run_dir/names["summary"]).write_text(json.dumps(jsafe(summary),indent=2,ensure_ascii=False),encoding="utf-8")
    lines=[{"op":"step","n":n_+1,"kind":s["kind"],"cartridge_id":s["cartridge_id"],"old_len":s["old_len"],"new_len":s["new_len"],"prior_rows_bitwise_unchanged":s["prior_rows_bitwise_unchanged"],
            "memory_sha":s["memory_sha"],"forwards":s["forwards"]}for n_,s in enumerate(steps)]
    lines+=[{"op":"ask","question":question,"answer":a1["answer"],"forwards":fw_a},{"op":"memory_profile","k_norm":prof["k_norm"],"v_norm":prof["v_norm"]}]
    if cu:lines.append({"op":"ask_free_text","question":cu["question"],"answer":cu["answer"],"forwards":cu["forwards"]})
    (run_dir/names["jsonl"]).write_text("\n".join(json.dumps(jsafe(x),ensure_ascii=False)for x in lines)+"\n",encoding="utf-8")
    (run_dir/names["txt"]).write_text(make_txt(P,sha,names,sealed_utc),encoding="utf-8");seal_s=time.perf_counter()-t_seal
    # ---- 7 figures ----
    yield ev(7,"Rendering figures",f"{N_FIGS} figures at 300 dpi (JPEG) with vector PDF copies; every printed value is checked against the sealed payload.")
    ctx={"P":P,"sha":sha,"run_id":run_id,"payload_name":payload_name,"allow":allow,"strip":strip,"N":N_FIGS,"audit":[],"pdfs":[]}
    t_r=time.perf_counter();imgs=render_all(ctx,run_dir);render_s=time.perf_counter()-t_r;entries=[]
    for n_,(p,cap)in enumerate(imgs,1):
        with Image.open(p)as im:w_,h_=im.size
        q_=ctx["pdfs"][n_-1];entries.append({"index":n_,"jpg":p.name,"pdf":q_.name,"title":cap,"width_px":w_,"height_px":h_,"dpi":OUT_DPI,"jpg_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"pdf_sha256":hashlib.sha256(q_.read_bytes()).hexdigest()})
    imf=run_dir/names["images"]
    imf.write_text(json.dumps({"run_id":run_id,"payload_sha256":sha,"verdict":verdict,"count":len(entries),"render_seconds":render_s,"figures":entries,"figure_text_audit":ctx["audit"],
                               "note":"Per-file SHA-256 is an artifact integrity seal. Figure text was checked against the sealed payload and scanned for claim discipline before saving."},indent=2,ensure_ascii=False),encoding="utf-8")
    # ---- 8 package ----
    yield ev(8,"Packaging","ZIP with figures, PDFs, payload, manifest, logs and raw records; post-seal audit.")
    zp=run_dir/names["zip"];members=[p for p,_ in imgs]+list(ctx["pdfs"])+[imf,mp,pp,run_dir/names["txt"],run_dir/names["summary"],run_dir/names["jsonl"]]
    for m in members:
        if m.suffix in(".json",".txt",".jsonl"):chk(f"claim-discipline and foreign-model scan of {m.name}: 0 hits",not claim_scan(m.name,strip_words(m.read_text(encoding="utf-8"),strip),allow=allow,numeric=m.suffix==".txt"))
    with zipfile.ZipFile(zp,"w",compression=zipfile.ZIP_DEFLATED)as z:
        for m in members:z.write(m,arcname=m.name)
    post=post_seal_audit(imgs,ctx["pdfs"],zp,[m.name for m in members],pp,sha,run_dir/names["txt"],mp,verdict)
    say(f"{run_id}: {len(checks)} pre-seal + {len(post)} post-seal checks = {len(checks)+len(post)}/{len(checks)+len(post)} PASS · figures {render_s:.1f}s · seal {seal_s:.2f}s")
    say("VERDICT :",verdict);say("PAYLOAD :",sha);say("ZIP     :",zp);say("="*140)
    ledger=dict(run_id=run_id,utc=sealed_utc,case=case,position=position,question=question,answer=a1["answer"],expected=expected,correct=bool(a1["correct"]),memory_tokens=T,
                prior_rows_unchanged=all(s["prior_rows_bitwise_unchanged"]for s in steps[1:]),weights_unchanged=True,verdict=verdict,payload_sha256=sha)
    return dict(run_id=run_id,run_dir=run_dir,P=P,sha=sha,imgs=imgs,pdfs=ctx["pdfs"],zip=zp,payload=pp,txt=run_dir/names["txt"],manifest=mp,images_manifest=imf,summary=run_dir/names["summary"],
                jsonl=run_dir/names["jsonl"],checks=len(checks)+len(post),verdict=verdict,ledger=ledger)
# ---------------- optional LIVE replay of the locked panel (72 runs; reported separately from the TEST596 reference) ----------------
def replay_report(I,run_id,run_dir,progress=None):
    info=I.info;t0=time.perf_counter();start=utc_now();g0=I.guard();s0=I.sentinel()if hasattr(I,"sentinel")else info["sentinel0"]
    if g0!=info["guard0"]:raise AuditFail("all-parameter guard before replay differs from startup")
    R=I.replay(progress);g1=I.guard()
    if g1!=g0:raise AuditFail("all-parameter guard changed during replay")
    rows=R["rows"];score=sum(int(r["correct"])for r in rows)
    if score!=R["score"]or len(rows)!=72:raise AuditFail("replay score does not re-derive from its rows")
    per={p:sum(int(r["correct"])for r in rows if r["position"]==p)for p in POSITIONS}
    for r in rows:
        ro=reference_outcome(r["case"],r["position"]);r["reference_correct"]=ro["correct"];r["reference_answer"]=ro["answer"]
        r["matches_sealed"]=(bool(r["correct"])==ro["correct"])and(ro["correct"]or norm_ans(r["answer"])==norm_ans(ro["answer"]))
    same=sum(int(r["matches_sealed"])for r in rows);fails=sorted((r["case"],r["position"])for r in rows if not r["correct"])
    out={"schema":"akbascore.mam.deepseek.mla.dc3.live_replay.v1","model":MODEL_ID,"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"license":LICENSE_URL,"run_id":run_id,"start_utc":start,"end_utc":utc_now(),
         "status":"LIVE REPLAY — separate from the TEST596 reference","score":score,"total":72,"per_position":per,"matches_TEST596_total":score==TEST596["score"]["TOTAL"],
         "failed_cases":fails,"failed_cases_match_TEST596":fails==sorted((c,p)for c,p,_,_ in TEST596["failures"]),"engine_reports_match":R.get("matches_TEST596"),"outcomes_identical_to_TEST596":same,"panel_sha":R["panel_sha"],
         "engine_sha256":ENGINE_SHA256_EXPECTED,"engine_blob":ENGINE_BLOB_EXPECTED,"guard_before":g0,"guard_after":g1,"weights_unchanged":g1==g0==info["guard0"],"gpu":info["gpu"],"seconds":round(time.perf_counter()-t0,2),
         "rows":rows,"reference":{"name":"TEST596","INCR_DC3_MLA":TEST596["score"],"failures":TEST596["failures"]}}
    pb=canon(out);sha=hashlib.sha256(pb).hexdigest();jp=run_dir/f"{run_id}_LIVE_REPLAY.json";jp.write_bytes(pb)
    S="="*120;txt=[S,f"{DEMO_TITLE} — {MODEL_FAMILY} · LIVE REPLAY OF THE LOCKED PANEL (INCR_DC3)",f"{DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}",S,
         "This replay is a LIVE measurement in this runtime. It is reported separately from the TEST596 reference and does not replace it.",
         f"run {run_id} · {start} → {out['end_utc']} · {out['seconds']} s · GPU {info['gpu']}",f"LIVE score {score}/72 · FIRST {per['FIRST']}/24 · MIDDLE {per['MIDDLE']}/24 · LAST {per['LAST']}/24",
         f"TEST596 reference INCR_DC3_MLA: {TEST596['score']['TOTAL']}/72 · failed cases now: {fails} · same failed cases as TEST596: {out['failed_cases_match_TEST596']} · outcomes identical to TEST596: {same}/72",f"weights unchanged (all-parameter guard before = after = startup): {out['weights_unchanged']}",
         f"payload SHA-256 {sha}","-"*120]+[f"case {r['case']:02d} {r['position']:<6} expected {r['expected']:<10} answer {r['answer']!r} {'PASS'if r['correct']else'FAIL'} · TEST596 {'PASS' if r['reference_correct'] else 'FAIL '+str(r['reference_answer'])} {'=' if r['matches_sealed'] else '≠'}"for r in rows]+[S]
    tp=run_dir/f"{run_id}_LIVE_REPLAY.txt";tp.write_text("\n".join(txt),encoding="utf-8")
    for p_ in(jp,tp):
        h_=claim_scan(p_.name,strip_words(p_.read_text(encoding="utf-8"),I.panel_strings()),allow=(info["gpu"],CANON_GPU),numeric=p_.suffix==".txt")
        if h_:raise AuditFail(f"claim-discipline hit in {p_.name}: {h_[:3]}")
    return dict(out=out,sha=sha,json=jp,txt=tp)
#<<CORE_END>>
#<<FIGURES_BEGIN>>
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827","xtick.color":"#374151","ytick.color":"#374151",
                     "xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,"axes.titleweight":"bold","axes.titlelocation":"left","pdf.fonttype":42,"ps.fonttype":42,
                     "legend.fontsize":8,"legend.frameon":False})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F9FAFB"
BLUE,BLUE_L,ORANGE,ORANGE_L,TEAL,TEAL_L,GRAYPT,AMBER_L,AMBER,RED="#1D4ED8","#DBEAFE","#C2410C","#FFEDD5","#0F766E","#CCFBF1","#9CA3AF","#FEF3C7","#B45309","#B91C1C"
SEGC=["#9CA3AF","#1D4ED8","#0F766E","#7C3AED","#B45309","#BE185D"]
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
def finish(fig,path,ctx,expect,desc):
    """Every expected value must appear in the figure text; claim scan must be clean; then JPEG (300 dpi, EXIF provenance) + vector PDF."""
    sanitize_dashes(fig);fig.canvas.draw();texts=[t.get_text()for t in fig.findobj(Text)if t.get_text().strip()];blob="\n".join(texts);nm=Path(path).name
    ws=lambda z:re.sub(r"\s+","",z);bw=ws(blob);miss=[e for e in expect if ws(e)not in bw]
    if miss:plt.close(fig);raise AuditFail(f"FIGURE/PAYLOAD MISMATCH in {nm}: missing {miss}")
    hits=claim_scan(nm,strip_words(re.sub(r"\s*\n\s*"," ",blob),ctx.get("strip",[])),allow=ctx["allow"],numeric=True)
    if hits:plt.close(fig);raise AuditFail(f"CLAIM-DISCIPLINE HIT in {nm}: {hits[:3]}")
    pdf=Path(path).with_suffix(".pdf")
    fig.savefig(pdf,format="pdf",facecolor="white",metadata={"Title":desc,"Author":f"{AUTHOR} · {ORG}","Subject":f"{DEMO_TITLE} · run {ctx['run_id']} · payload SHA-256 {ctx['sha']}","Creator":DEMO_SHORT,
                "Keywords":f"model {MODEL_ID} revision {MODEL_REVISION}; engine 596.py git blob {ENGINE_BLOB}; engine sha256 {ENGINE_SHA256}; TEST596 reference 69/72; license {LICENSE_URL}"})
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=OUT_DPI,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf)as im:rgb=im.convert("RGB")
    ex=Image.Exif();ex[0x010E]=ascii_(f"{desc} | {DEMO_SHORT} | run {ctx['run_id']} | payload sha256 {ctx['sha']} | engine sha256 {ENGINE_SHA256}");ex[0x013B]=ascii_(f"{AUTHOR} / {ORG}");ex[0x8298]=ascii_(f"{COPYRIGHT} - {LICENSE_NAME} - {LICENSE_URL}");ex[0x0131]=ascii_(DEMO_SHORT)
    rgb.save(path,"JPEG",quality=95,optimize=True,progressive=False,subsampling=0,dpi=(OUT_DPI,OUT_DPI),exif=ex.tobytes())
    with Image.open(path)as c_:
        if c_.format!="JPEG"or c_.mode!="RGB":raise RuntimeError("JPEG validation failed")
    ctx["pdfs"].append(pdf);ctx["audit"].append(dict(file=nm,texts=len(texts),expected_values=len(expect),claim_hits=0));return str(path)
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
def header(fig,k,title,sub,tag,tagc):
    fig.text(.03,.955,f"Fig. {k}",fontsize=12,weight="bold",color=MUTED,va="center")
    fig.text(.078,.955,mt(title),fontsize=16.5,weight="bold",color=INK,va="center")
    fig.text(.03,.917,mt(sub),fontsize=9.4,color=INK2,va="center")
    fig.text(.97,.955,tag,fontsize=8.4,weight="bold",color=tagc,va="center",ha="right",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec=tagc,lw=1.0))
    fig.add_artist(Line2D([.03,.97],[.895,.895],transform=fig.transFigure,color=INK2,lw=.8))
def footer(fig,ctx,k,caption):
    fig.add_artist(Line2D([.03,.97],[.132,.132],transform=fig.transFigure,color=RULE,lw=.6))
    fit_text(fig,.03,.045,.94,.08,caption,fs_max=8.6,fs_min=6.6,color=INK2)
    fig.add_artist(Line2D([.03,.97],[.038,.038],transform=fig.transFigure,color=RULE,lw=.6))
    fig.text(.5,.025,mt(f"{DEMO_TITLE} · {DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}"),ha="center",va="center",fontsize=6.6,color=INK2)
    fig.text(.5,.010,mt(f"live run {ctx['run_id']} · {MODEL_FAMILY} · 596.py engine (blob {ENGINE_BLOB[:12]}…; not TEST596 itself) · TEST596 reference 69/72 · payload {ctx['sha'][:12]}… · Fig. {k}/{ctx['N']} · {COPYRIGHT}"),
             ha="center",va="center",fontsize=6.2,color=MUTED,family=MONO)
def clean(ax,grid=True):
    for s in("top","right"):ax.spines[s].set_visible(False)
    if grid:ax.grid(True,color=GRIDC,lw=.5);ax.set_axisbelow(True)
def frame(fig,x,y,w,h,ec=INK2,fc="white",lw=.9,ls="-",z=-1):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=lw,ls=ls,zorder=z))
def block(fig,x,y,w,h,stage,title,body,ec=INK2,fc="white",mono=True,bfs=8.4):
    frame(fig,x,y,w,h,ec,fc);frame(fig,x,y+h-.032,w,.032,ec,PANEL if fc=="white"else fc,lw=.9)
    fig.text(x+.007,y+h-.016,stage,fontsize=7,weight="bold",color=MUTED,va="center")
    fig.text(x+.007,y+h-.052,mt(title),fontsize=9.6,weight="bold",color=INK,va="center")
    if body:fit_text(fig,x+.007,y+.008,w-.014,h-.078,body,fs_max=bfs,fs_min=6.4,family=MONO if mono else None,color=INK2)
def arrow(fig,x0,y0,x1,y1,color=INK2,lw=1.1,ms=11,cs="arc3"):fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,connectionstyle=cs,zorder=6))
def L_(ctx):return ctx["P"]["live"]
def short(s,n):s=str(s);return s if len(s)<=n else s[:n-1]+"…"
# ------------------------------------------------------------------ FIG 1 · write → append → ask
def f1(ctx,k,path):
    L=L_(ctx);cfg=ctx["P"]["engine"]["config"];st=L["steps"];cars=L["cartridges"];T=L["memory"]["T"];P_=cfg["prefix_tokens"];fig=new_fig();a1=L["answer"];AR=cfg["arch"];UP=upper_txt(cfg["CUT"],AR[0])
    header(fig,k,"Write → append-only consolidation → source-free answer",f"{MODEL_FAMILY} · case {L['case']:02d} · target at {L['position']} · the five cartridges, token counts and the answer below are taken from this run.","ARCHITECTURE · VALUES FROM THIS RUN",INK2)
    for x,s in((.03,"(a) SOURCE FACTS · READ ONCE"),(.255,"(b) WRITE · INDEPENDENT CARTRIDGES"),(.505,"(c) APPEND · CONSOLIDATED MEMORY"),(.765,"(d) QUESTION · NO SOURCE TEXT")):fig.text(x,.865,s,fontsize=8,weight="bold",color=INK2,va="center")
    y0=.79;hh=.098;gap=.012
    for n_,(c,s)in enumerate(zip(cars,st)):
        y=y0-n_*(hh+gap);col=SEGC[n_+1];tg=c["is_target"]
        frame(fig,.03,y-hh,.205,hh,ec=col,fc=(BLUE_L if tg else"white"),lw=1.4 if tg else .9)
        fig.text(.036,y-.018,f"C{n_+1} · #{c['id']:02d} · {c['type']}"+("  · TARGET"if tg else""),fontsize=7.8,weight="bold",color=col,va="center")
        fit_text(fig,.036,y-hh+.006,.193,hh-.034,c["fact"],fs_max=8.2,fs_min=6.4,color=INK)
        frame(fig,.255,y-hh,.225,hh,ec=col,fc="white",lw=.9)
        fig.text(.261,y-.018,f"cartridge C{n_+1} · written in step {n_+1}",fontsize=7.6,weight="bold",color=col,va="center")
        wr=s["forwards"][0]["n"];body=c["body_len"]
        fig.text(.261,y-.044,mt(f"1 frozen forward · {wr} token ids · no cache"),fontsize=7.3,family=MONO,color=INK2,va="center")
        fig.text(.261,y-.066,mt(f"H{cfg['CUT']} residual  {(wr if n_==0 else body)} × {AR[1]}"),fontsize=7.3,family=MONO,color=INK2,va="center")
        fig.text(.261,y-.087,mt(f"MLA cache L0–{cfg['CUT']}  {(wr if n_==0 else body)} rows"),fontsize=7.3,family=MONO,color=INK2,va="center")
        arrow(fig,.235,y-hh/2,.255,y-hh/2,color=col);arrow(fig,.48,y-hh/2,.505,y-hh/2,color=col)
    ax=fig.add_axes([.515,.255,.225,.535]);ax.set_zorder(3);ax.set_xlim(0,T);ax.set_ylim(-.5,4.5);ax.invert_yaxis();clean(ax,grid=False)
    seg=L["segments"]
    for n_,s in enumerate(st):
        old=s["old_len"];new=s["new_len"]
        if n_==0:ax.barh(n_,P_,left=0,color=SEGC[0],height=.62);ax.barh(n_,new-P_,left=P_,color=SEGC[1],height=.62)
        else:
            for m_,(lab,a_,b_,_)in enumerate(seg[:n_+1]):ax.barh(n_,b_-a_,left=a_,color=SEGC[m_],height=.62,alpha=.28)
            ax.barh(n_,new-old,left=old,color=SEGC[n_+1],height=.62)
        ax.text(new+.6,n_,f"{new}",fontsize=7.6,va="center",color=INK)
    ax.set_yticks([]);ax.spines["left"].set_visible(False);ax.set_xlabel("memory rows (tokens)",fontsize=8)
    ax.set_title("memory after step 1 … 5 (faded = earlier rows, never recomputed)",fontsize=7.8,weight="normal",pad=3)
    fig.text(.515,.205,mt(f"{UP}: computed only for new rows,\nattending to the existing memory · layers 0–{cfg['CUT']}: rotary part re-phased"),fontsize=7.4,color=INK2,va="top")
    block(fig,.765,.43,.205,.36,"QUERY","Question over the memory",f"{L['question']}\n\nmodel input: {L['query_tokens']} template tokens only\ninstalled memory: {T} numerical rows\n(prefix + C1 … C5)\nsource facts given to the model: none\nweights: frozen",mono=False,bfs=8.6)
    arrow(fig,.742,.6,.765,.6)
    frame(fig,.765,.205,.205,.2,ec=BLUE if L["correct"]else ORANGE,fc=BLUE_L if L["correct"]else ORANGE_L,lw=1.3)
    fig.text(.775,.385,"ANSWER (generated by the frozen model)",fontsize=7.4,weight="bold",color=INK2,va="center")
    fig.text(.775,.33,mt(short(a1["answer"],22)),fontsize=15,weight="bold",color=INK,va="center")
    fig.text(.775,.272,mt(f"expected {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'}"),fontsize=8.6,color=BLUE if L["correct"]else ORANGE,weight="bold",va="center")
    fig.text(.775,.235,"expected answer = audit metadata",fontsize=7,color=MUTED,va="center")
    footer(fig,ctx,k,f"Figure {k}. The executed memory path of {MODEL_FAMILY} ({AR[0]} layers, hidden {AR[1]}) for case {L['case']:02d}. (a) Each source fact is read once. (b) The frozen model writes an independent numerical cartridge per fact: the layer-{cfg['CUT']} residual H{cfg['CUT']} and the layers 0–{cfg['CUT']} MLA cache (latent c_KV and rotary key). The engine writes all five cartridges before consolidating. "
           f"(c) Cartridges are appended in order; for every append, {UP} are computed only for the new rows over the existing memory, so earlier rows are never recomputed (bitwise check, Fig. 2). "
           f"(d) The question template ({L['query_tokens']} tokens) is processed over the {T}-token numerical memory; no source text is given to the model. Weights are frozen throughout.")
    return finish(fig,path,ctx,[f"{T}",f"{L['query_tokens']} template tokens",mt(short(a1["answer"],22)),f"expected {L['expected']}",f"× {AR[1]}",UP],"Write, append, ask")
# ------------------------------------------------------------------ FIG 2 · append-only growth (measured)
def f2(ctx,k,path):
    L=L_(ctx);st=L["steps"];cars=L["cartridges"];T=L["memory"]["T"];fig=new_fig()
    header(fig,k,"Append-only consolidation, measured forward by forward","Every model forward during load_case() and a bitwise comparison of all previously consolidated cache rows after each append.","LIVE MEASUREMENT",BLUE)
    ax=fig.add_axes([.06,.31,.42,.52]);ax.set_zorder(3);clean(ax)
    for n_,s in enumerate(st):
        old=s["old_len"];add=s["new_len"]-old;col=SEGC[n_+1]
        ax.bar(n_,old,color="#E5E7EB",edgecolor=INK2,lw=.5,width=.62);ax.bar(n_,add,bottom=old,color=col,width=.62)
        ax.text(n_,s["new_len"]+1.2,f"{s['new_len']}",ha="center",fontsize=8,color=INK)
        if n_:ax.text(n_,old/2,"✓",ha="center",va="center",fontsize=13,color=TEAL,weight="bold")
    ax.set_xticks(range(5));ax.set_xticklabels([f"step {n_+1}\nC{n_+1}"+("\nTARGET"if c["is_target"]else"")for n_,c in enumerate(cars)],fontsize=7.8)
    ax.set_ylabel("memory rows (tokens)");ax.set_ylim(0,T*1.12);ax.set_title("memory length after each step",fontsize=8.6)
    ax.legend(handles=[Patch(facecolor="#E5E7EB",edgecolor=INK2,lw=.5,label="earlier rows (✓ bitwise unchanged)"),Patch(color=SEGC[1],label="rows added in this step")],loc="upper left")
    tx=.53;fig.text(tx,.83,"Forward record (instrumentation)",fontsize=10,weight="bold",color=INK)
    fig.text(tx,.80,mt(f"{'step':<5}{'write forward':<27}{'consolidation forward':<38}{'prior rows':<12}"),fontsize=7.7,family=MONO,color=MUTED)
    for n_,s in enumerate(st):
        f=s["forwards"];y=.772-n_*.083
        w_=f"ids {f[0]['n']} · no cache"
        c_=(f"zero-emb {f[1]['n']} · cache {f[1]['past_len']}"if f[1]["has_past"]else f"zero-emb {f[1]['n']} · no cache")
        pr="init"if s["kind"]=="init"else("✓ identical"if s["prior_rows_bitwise_unchanged"]else"✗ CHANGED")
        fig.text(tx,y,mt(f"{n_+1:<5}{w_:<27}{c_:<38}{pr:<12}"),fontsize=7.9,family=MONO,color=INK,weight="bold"if cars[n_]["is_target"]else"normal")
        if s["kind"]=="append":fig.text(tx,y-.028,mt(f"     SHA rows 0…{s['old_len']-1}: {s['prior_sha_before'][:12]}… = {s['prior_sha_after'][:12]}…"),fontsize=7.1,family=MONO,color=INK2)
        else:fig.text(tx,y-.028,mt(f"     prefix + body of C1 written together ({s['new_len']} rows)"),fontsize=7.1,family=MONO,color=INK2)
    frame(fig,tx,.17,.44,.17,ec=TEAL,fc=TEAL_L,lw=1.0)
    fit_text(fig,tx+.008,.178,.424,.155,f"Measured in this run:\n• source token-id forwards during load: {L['source_reads']} — one per cartridge, each its own fact, each once\n"
             f"• consolidation forwards with all-zero input embeddings: {L['zero_embed_forwards']} (stored H{ctx['P']['engine']['config']['CUT']} injected; no token ids)\n"
             f"• previously consolidated rows bitwise unchanged after all 4 appends: {all(s['prior_rows_bitwise_unchanged']for s in st[1:])}\n• final memory {T} rows · SHA-256 {L['memory_sha'][:20]}…",fs_max=8.4,fs_min=6.6,color=INK)
    fig.text(.06,.21,mt(f"total load_case() time {1000*L['setup']['elapsed_seconds']:.0f} ms (engine-reported) · per step: "+" · ".join(f"{1000*s['seconds']:.0f}"for s in st)+" ms"),fontsize=7.8,family=MONO,color=INK2)
    cfg=ctx["P"]["engine"]["config"]
    footer(fig,ctx,k,f"Figure {k}. Live record of the five consolidation steps ({MODEL_FAMILY}). Step 1 writes cartridge C1 together with the {cfg['prefix_tokens']}-token instruction prefix; steps 2–5 append C2–C5. In each step the instrumentation sees exactly one token-id forward "
           f"(the write of the new cartridge's own fact, no cache) and one consolidation forward whose input embeddings are all zero (the stored H{cfg['CUT']} is injected) and whose length equals the new rows only, over the existing memory. "
           f"The engine performs the five writes first; each write is shown in the row of its cartridge. After every append all earlier latent and rotary rows of all {L['memory']['layers']} layers are compared bitwise with the memory before the append.")
    return finish(fig,path,ctx,[f"{T}",f"{L['source_reads']}",L["memory_sha"][:20],"✓ identical"],"Append-only consolidation")
# ------------------------------------------------------------------ FIG 3 · consolidated memory map
def f3(ctx,k,path):
    L=L_(ctx);M=L["memory"];T=M["T"];seg=L["segments"];cfg=ctx["P"]["engine"]["config"];fig=new_fig()
    header(fig,k,"The consolidated numerical memory",f"{MODEL_FAMILY} · {M['layers']} layers × {T} token rows · MLA latent {M['head_dim']} + rotary {M['rope_dim']} · {M['bytes']/2**20:.2f} MiB ({M['dtype']}) · values read from the memory of this run.","LIVE MEASUREMENT",BLUE)
    for n_,(key,ttl)in enumerate((("k_norm",MEM_LABELS[0]),("v_norm",MEM_LABELS[1]))):
        A=np.asarray(M[key],dtype=float);mu=A.mean(axis=1,keepdims=True);sd=A.std(axis=1,keepdims=True);sd[sd==0]=1;Z=(A-mu)/sd
        ax=fig.add_axes([.06,.585-n_*.29,.80,.235]);ax.set_zorder(3)
        im=ax.imshow(Z,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5,interpolation="nearest",origin="lower")
        for lab,a_,b_,_ in seg[1:]:ax.axvline(a_-.5,color=INK,lw=.8)
        ax.axhline(cfg["CUT"]+.5,color=AMBER,lw=1.4,ls=(0,(4,2)))
        NLm=M["layers"];mid=NLm//2;ax.set_yticks([0,cfg["CUT"],mid,NLm-1]);ax.set_yticklabels(["L0",f"L{cfg['CUT']}",f"L{mid}",f"L{NLm-1}"],fontsize=7);ax.set_title(mt(ttl+" · z-scored within each layer"),fontsize=8.4)
        ax.set_xlim(-.5,T-.5);ax.set_xticks([]);cb=fig.colorbar(im,cax=fig.add_axes([.87,.585-n_*.29,.008,.235]));cb.ax.tick_params(labelsize=6.5);cb.outline.set_linewidth(.5)
    fig.text(.86,.852,mt(f"dashed line: layers 0–{cfg['CUT']} below come from the cartridges · {upper_txt(cfg['CUT'],M['layers'])} above were computed by the appends"),fontsize=7.6,color=AMBER,ha="right",va="center",weight="bold")
    sx=fig.add_axes([.06,.235,.80,.032]);sx.set_zorder(3);sx.set_xlim(-.5,T-.5);sx.set_ylim(0,1);sx.axis("off")
    for n_,(lab,a_,b_,ci)in enumerate(seg):
        c_=L["cartridges"][n_-1]if n_ else None;tg=bool(c_ and c_["is_target"])
        sx.add_patch(Rectangle((a_-.5,0),b_-a_,1,color=SEGC[n_],alpha=1 if(tg or n_==0)else .75,lw=0))
        sx.text((a_+b_-1)/2,.5,(lab+("·T"if tg else"")),ha="center",va="center",fontsize=7.6,color="white",weight="bold")
    fig.text(.06,.218,mt("row layout: "+" · ".join(f"{lab} rows {a_}–{b_-1}"for lab,a_,b_,_ in seg)),fontsize=7.5,family=MONO,color=INK2,va="top")
    footer(fig,ctx,k,f"Figure {k}. Map of the consolidated memory after the fifth append. Each column is one token row, each row of the heat map one layer; colour shows the row's latent c_KV (top) or rotary key (bottom) norm, z-scored within each layer, for display only. "
           f"Vertical lines separate the instruction prefix and the five cartridges (T marks the target). Layers 0–{cfg['CUT']} come from the cartridges (rotary part re-phased to their memory positions); {upper_txt(cfg['CUT'],M['layers'])} were computed by the appends. The memory holds numbers only; no text is stored in it.")
    return finish(fig,path,ctx,[f"{T} token rows",f"{M['bytes']/2**20:.2f} MiB",f"{M['layers']} layers",f"L{M['layers']-1}"],"Consolidated memory")
# ------------------------------------------------------------------ FIG 4 · source-free question answering
def f4(ctx,k,path):
    L=L_(ctx);a1=L["answer"];T=L["memory"]["T"];qt=L["query_token_text"];fig=new_fig();ok=L["correct"]
    header(fig,k,"Answering from numerical memory without the source text",f"{MODEL_FAMILY} · the exact model input at query time, the installed memory and the generated answer.","LIVE MEASUREMENT",BLUE)
    frame(fig,.03,.645,.94,.19,ec=RULE,fc=PANEL,lw=.7)
    fig.text(.04,.815,mt(f"MODEL INPUT AT QUERY TIME · {L['query_tokens']} token ids (question template) · over {T} memory rows · source text: none"),fontsize=8.6,weight="bold",color=INK2,va="center")
    # DeepSeek special-token glyphs (fullwidth bar, lower block) are shown with ASCII look-alikes; display only
    toks=[t.replace("\n","⏎").replace("｜","|").replace("▁","_")for t in qt];x=.04;y=.775;fig.canvas.draw();rdr=fig.canvas.get_renderer();FW=fig.get_figwidth()*fig.dpi
    for t in toks:
        s_=mt(t if t.strip()else"·");tt=fig.text(x,y,s_,fontsize=8.2,family=MONO,color=INK,va="center",bbox=dict(boxstyle="square,pad=0.25",fc="white",ec=RULE,lw=.5))
        w=tt.get_window_extent(renderer=rdr).width/FW
        if x+w>.96 and x>.041:tt.set_position((.04,y-.045));x=.04;y-=.045
        x+=w+.006
    fig.text(.04,.665,mt(f"installed memory: {T} rows × {L['memory']['layers']} layers (prefix + C1…C5) · first query forward cache = {L['query_past']} · generation forwards {L['gen_forwards']} (one token each)"),fontsize=7.9,family=MONO,color=INK2,va="center")
    frame(fig,.03,.2,.46,.415,ec=BLUE if ok else ORANGE,fc=BLUE_L if ok else ORANGE_L,lw=1.3)
    fig.text(.045,.585,"REFERENCE QUESTION OF THIS CASE",fontsize=8,weight="bold",color=INK2,va="center")
    fit_text(fig,.045,.49,.43,.075,L["question"],fs_max=11,fs_min=7,color=INK)
    fig.text(.045,.44,"answer",fontsize=8,color=MUTED,va="center");fig.text(.045,.375,mt(short(a1["answer"],30)),fontsize=23,weight="bold",color=INK,va="center")
    fig.text(.045,.29,mt(f"expected {L['expected']} (audit metadata) · {'CORRECT'if ok else'INCORRECT'}"),fontsize=9,weight="bold",color=BLUE if ok else ORANGE,va="center")
    fig.text(.045,.228,mt(f"repeat: identical = {L['repeat_identical']} · memory unchanged by querying = {L['memory_unchanged_by_query']} · {1000*a1['elapsed_seconds']:.0f} ms"),fontsize=7.8,family=MONO,color=INK2,va="center")
    cu=L.get("custom")
    frame(fig,.51,.2,.46,.415,ec=RULE,fc="white",lw=.8)
    fig.text(.525,.585,"FREE-TEXT QUESTION (optional · unscored)",fontsize=8,weight="bold",color=INK2,va="center")
    if cu:
        fit_text(fig,.525,.49,.43,.075,cu["question"],fs_max=11,fs_min=7,color=INK)
        fig.text(.525,.44,"answer",fontsize=8,color=MUTED,va="center");fit_text(fig,.525,.255,.43,.165,cu["answer"],fs_max=15,fs_min=7.5,color=INK,weight="bold")
        fig.text(.525,.228,mt(f"{cu['query_tokens']} template tokens over the same {T}-row memory · not scored"),fontsize=7.8,family=MONO,color=INK2,va="center")
    else:fit_text(fig,.525,.25,.43,.3,"No free-text question in this run. Visitors may type their own question; it is answered from the same numerical memory and reported without a score.",fs_max=9,fs_min=7,color=MUTED)
    footer(fig,ctx,k,f"Figure {k}. Source-free question answering for case {L['case']:02d}. Top: the complete list of token ids given to the model at query time, decoded for display — the chat-formatted question only; none of the five source facts is supplied. "
           f"The model attends to the {T}-row numerical memory installed as its cache and generates the answer greedily. Repeating the question gives the same answer, and the memory's SHA-256 is unchanged by querying.")
    return finish(fig,path,ctx,[f"{L['query_tokens']} token ids",mt(short(a1["answer"],30)),f"expected {L['expected']}"],"Source-free answering")
# ------------------------------------------------------------------ FIG 5 · TEST596 reference, the three failures and the live result
def f5(ctx,k,path):
    L=L_(ctx);fig=new_fig();S_=TEST596["score"];ro=L["reference_outcome"]
    header(fig,k,"TEST596 reference, its three failures and the live result",f"Left: archived TEST596 values of {MODEL_FAMILY} on the shared locked panel (24 cases × 3 positions), reproduced verbatim. Right: live results of this session.","REFERENCE + LIVE",INK2)
    ax=fig.add_axes([.06,.40,.30,.40]);ax.set_zorder(3);clean(ax);cols={"FIRST":BLUE,"MIDDLE":TEAL,"LAST":"#7C3AED"}
    for j,p in enumerate(POSITIONS):
        ax.bar(j,S_[p],.62,color=cols[p]);ax.bar(j,24-S_[p],.62,bottom=S_[p],color=ORANGE_L,edgecolor=ORANGE,lw=.6)
        ax.text(j,24.6,f"{S_[p]}/24",ha="center",fontsize=8.2,color=INK,weight="bold")
    ax.set_xticks(range(3));ax.set_xticklabels(list(POSITIONS),fontsize=8.2);ax.set_ylim(0,28);ax.set_ylabel("correct of 24 cases")
    ax.set_title(f"TEST596 · INCR_DC3_MLA · total {S_['TOTAL']}/72",fontsize=8.6)
    bx=fig.add_axes([.42,.40,.15,.40]);bx.set_zorder(3);clean(bx);arms=[("JOINT",TEST597["joint"]),("INCR",TEST597["incr_on_failing"]),("INDEP",TEST597["native_indep"])]
    for j,(nm,(c_,n_))in enumerate(arms):
        bx.bar(j,c_,.6,color=[GRAYPT,BLUE,"#9CA3AF"][j]);bx.text(j,c_+.08,f"{c_}/{n_}",ha="center",fontsize=8,color=INK)
    bx.set_xticks(range(3));bx.set_xticklabels([a for a,_ in arms],fontsize=7.6);bx.set_ylim(0,3.6);bx.set_yticks([0,1,2,3])
    bx.set_title("TEST597 · the 3 failed slots",fontsize=8.2)
    fit_text(fig,.06,.155,.51,.17,f"INCR_DC3_MLA = {ARM_DESC['INCR_DC3_MLA']}. Orange segments are the three failed questions. TEST597 re-ran the three failed slots: the failures were reproduced; "
             f"the joint-text control (facts processed together as text, not a memory) answered {TEST597['joint'][0]}/{TEST597['joint'][1]} and native independent MLA cartridges {TEST597['native_indep'][0]}/{TEST597['native_indep'][1]}. "
             f"No verified readout fix was found, so no score above {S_['TOTAL']}/72 is claimed. Scoring: {CORRECT_RULE_TXT}.",fs_max=7.6,fs_min=6.2,color=INK2)
    tx=.6;frame(fig,tx,.47,.37,.36,ec=BLUE,fc=BLUE_L,lw=1.1)
    fig.text(tx+.01,.81,"LIVE · this run",fontsize=9.6,weight="bold",color=BLUE,va="center")
    fig.text(tx+.01,.778,mt(f"case {L['case']:02d} · target {L['position']} · answer {short(L['answer']['answer'],18)}"),fontsize=8.3,family=MONO,color=INK,va="center")
    fig.text(tx+.01,.750,mt(f"expected {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'}"),fontsize=8.3,family=MONO,color=BLUE if L["correct"]else ORANGE,weight="bold",va="center")
    fig.text(tx+.01,.722,mt(f"TEST596: {'correct' if ro['correct'] else 'failed ('+str(ro['answer'])+')'} · identical: {L['matches_reference']}"),fontsize=7.9,family=MONO,color=INK2,va="center")
    ses=L.get("session")or[];per={p:[r for r in ses if r["position"]==p]for p in POSITIONS}
    fig.text(tx+.01,.688,"LIVE · sealed runs in this session (including this one)",fontsize=8.2,weight="bold",color=INK2,va="center")
    for j,p in enumerate(POSITIONS):
        rr=per[p];fig.text(tx+.01,.660-j*.027,mt(f"{p:<7} {sum(int(r['correct'])for r in rr)}/{len(rr)} correct"),fontsize=8.2,family=MONO,color=INK,va="center")
    fit_text(fig,tx+.01,.478,.35,.09,"A single live run is a demonstration observation. The optional live replay of all 72 panel runs is reported in its own file and compared case by case with TEST596. Neither replaces the reference.",fs_max=7.6,fs_min=6.3,color=INK2)
    frame(fig,tx,.15,.37,.29,ec=ORANGE,fc=ORANGE_L,lw=.9)
    fig.text(tx+.01,.42,"THE THREE FAILED CASES · TEST596 (reproduced in TEST597)",fontsize=8.2,weight="bold",color=ORANGE,va="center")
    fig.text(tx+.01,.385,mt(f"{'case':<6}{'position':<9}{'expected':<10}{'answered':<10}"),fontsize=8,family=MONO,color=MUTED,va="center")
    for j,(c_,p_,g_,w_)in enumerate(TEST596["failures"]):
        hit=(L["case"],L["position"])==(c_,p_)
        fig.text(tx+.01,.352-j*.034,mt(f"{c_:02d}{'':<4}{p_:<9}{g_:<10}{w_:<10}"+(" ← this run" if hit else "")),fontsize=8.6,family=MONO,color=INK,weight="bold"if hit else"normal",va="center")
    fit_text(fig,tx+.01,.158,.35,.085,"In each failure the model answers with the name of a distractor cartridge from the same memory. The failures are kept in every record of this demonstration.",fs_max=7.6,fs_min=6.3,color=INK2)
    footer(fig,ctx,k,f"Figure {k}. Left: the TEST596 reference of {MODEL_FAMILY} on the shared 24-case panel at three target positions: append-only incremental MLA DC3 {S_['TOTAL']}/72 "
           f"(FIRST {S_['FIRST']}/24, MIDDLE {S_['MIDDLE']}/24, LAST {S_['LAST']}/24), with the TEST597 diagnostic of the three failed slots. Right: live results of this session, kept separate, "
           "and the three failed cases listed in full (expected name and the answer given).")
    return finish(fig,path,ctx,[f"total {S_['TOTAL']}/72","Kareth","Marvek","Luneth","LIVE · this run",f"expected {L['expected']}"],"TEST596 reference, failures and live result")
# ------------------------------------------------------------------ FIG 6 · provenance, cross-model reference and priority record
def f6(ctx,k,path):
    P=ctx["P"];L=P["live"];cfg=P["engine"]["config"];E_=P["environment"];fz=P["frozen"];fig=new_fig();AR=cfg["arch"];M_=P["model"]
    header(fig,k,"Provenance, verification and priority record",f"{MODEL_FAMILY} · configuration, measured integrity quantities, hashes, authorship, the cross-model references and the claim boundary of this run.","PROVENANCE · PRIORITY RECORD",INK2)
    cols=[(.03,"CONFIGURATION AND ENGINE",
           f"run           {P['run_id']}\nstart (UTC)   {P['run_start_utc']}\nmodel         {cfg['MODEL_ID'].split('/')[-1]}\nrevision      {cfg['REVISION'][:20]}…\nweights       frozen · {M_['dtype']} · {M_['attn']}\n"
           f"architecture  {arch_short(tuple(AR))}\nexperts       {M_.get('routed_experts')} routed · {M_.get('experts_per_token')} per token · {M_.get('shared_experts')} shared\n"
           f"cartridge     H{cfg['CUT']} residual + MLA L0–{cfg['CUT']}\nconsolidation append-only · {upper_txt(cfg['CUT'],AR[0])}\nprefix        {cfg['prefix_tokens']} tokens (chat template)\n"
           f"panel         locked · SHA-256\n  {cfg['panel_sha'][:32]}\n  {cfg['panel_sha'][32:]}\nGPU           {E_['gpu']}\ntorch         {E_['torch']}\ntransformers  {E_['transformers']}\ngradio        {E_['gradio']}\n"
           f"engine file   {P['engine']['file']} (unchanged)\ngit blob\n  {P['engine']['blob']}\nend (UTC)     {P['run_end_utc']}"),
          (.35,"VERIFICATION MEASURED IN THIS RUN",
           f"engine fingerprint  startup = before = after\n  {fz['sentinel_after'][:40]}…\nall-parameter guard  startup = before = after\n  {fz['guard_after'][:40]}…\n"
           f"trainable tensors   {fz['trainable_tensors']}\nLoRA / optimizer    {fz['lora']} / {fz['optimizer']}\nforeign hooks       {fz['foreign_hooks_before']} → {fz['foreign_hooks_after']}\nresidual hooks      {len(fz['residual_hooks_before'])} → {len(fz['residual_hooks_after'])}\n"
           f"source token-id forwards (load)  {L['source_reads']}\nzero-embedding consolidations    {L['zero_embed_forwards']}\nprior rows unchanged (4 appends) "
           f"{all(s['prior_rows_bitwise_unchanged']for s in L['steps'][1:])}\nquery input = template only      True\nmemory unchanged by querying     {L['memory_unchanged_by_query']}\n"
           f"repeat answer identical          {L['repeat_identical']}\n\nmemory  {L['memory']['T']} rows · {L['memory']['bytes']/2**20:.2f} MiB\n  SHA {L['memory_sha'][:36]}…\nanswer  {short(L['answer']['answer'],24)} · {'correct'if L['correct']else'incorrect'}\n"
           f"same outcome as TEST596         {L['matches_reference']}\npre-seal checks passed: {len(P['checks_pre_seal'])}\nverdict\n  {P['verdict'].split('_APPEND')[0]}\n  _APPEND{P['verdict'].split('_APPEND',1)[1]}"),
          (.67,"AUTHORSHIP · PRIORITY · CLAIM BOUNDARY",
           f"{DISCOVERY}\n{ORG} · {DATE_TXT}\n{AUTHOR_PLACE}\n\n{PRIORITY}\n\n{LICENSE_NAME}\n{COPYRIGHT}\n{LICENSE_URL}\n\nTEST596 ({MODEL_FAMILY})\n  INCR_DC3_MLA {TEST596['score']['TOTAL']}/72 · 3 failures kept\n\n"
           "not established:\n"+"\n".join("  • "+t for t in("error-free readout (69/72)","memories above five cartridges","cache-only cartridges (H3 is part of it)","scored free-text questions","models beyond the three tested families")))]
    for x,ttl,body in cols:
        frame(fig,x,.335,.30,.535,ec=RULE,fc="white",lw=.8);fig.text(x+.01,.85,ttl,fontsize=9,weight="bold",color=INK2,va="center")
        fit_text(fig,x+.01,.345,.282,.485,body,fs_max=9.0,fs_min=5.8,family=MONO if x<.6 else None,color=INK)
    frame(fig,.03,.145,.94,.175,ec=BLUE,fc=BLUE_L,lw=1.2)
    fig.text(.042,.302,mt(CROSS_TITLE),fontsize=10.5,weight="bold",color=BLUE,va="center")
    fit_text(fig,.042,.152,.916,.132,f"{CROSS_TEXT} {CROSS_RESULT}\nEarlier records (references only): {PRIOR_PUBS[0]['title']} · DOI {PRIOR_PUBS[0]['doi']} · {PRIOR_PUBS[0]['url']}\n"
             f"{PRIOR_PUBS[1]['title']} · DOI {PRIOR_PUBS[1]['doi']} · {PRIOR_PUBS[1]['url']}\n{CROSS_BOUNDARY}",fs_max=8.2,fs_min=6.0,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Provenance record of run {P['run_id']} (verdict {P['verdict']}). Payload SHA-256 {ctx['sha']}. "
           "Hashes are artifact-integrity seals, not scientific proof or third-party verification. All live values come from the raw records stored in the sealed payload; reference values are reproduced verbatim from the TEST596 reference and the TEST597 log. "
           "Earlier records are cited by DOI only; none of their values is shown as a result here.")
    return finish(fig,path,ctx,[P["run_id"],P["engine"]["blob"],fz["guard_after"][:40],DISCOVERY,ORG,CROSS_TITLE,PRIOR_PUBS[0]["doi"],PRIOR_PUBS[1]["doi"],PRIOR_PUBS[1]["url"]],"Provenance and priority record")
FIGURES=[("fig_01_write_append_ask","Fig. 1 · Write → append-only consolidation → source-free answer",f1),("fig_02_append_only_record","Fig. 2 · Append-only consolidation, measured",f2),
         ("fig_03_memory_map","Fig. 3 · The consolidated numerical memory",f3),("fig_04_source_free_answer","Fig. 4 · Answering without the source text",f4),
         ("fig_05_reference_and_live","Fig. 5 · TEST596 reference, three failures and live result",f5),("fig_06_provenance_priority","Fig. 6 · Provenance, cross-model reference and priority record",f6)]
N_FIGS=len(FIGURES)
def render_all(ctx,run_dir):
    imgs=[]
    try:
        for i,(slug,cap,fn)in enumerate(FIGURES,1):
            p=Path(run_dir)/f"{slug}_{ctx['run_id']}.jpg";fn(ctx,i,p);imgs.append((p,cap))
    finally:plt.close("all")
    return imgs
#<<FIGURES_END>>
say("PART 2 / 3 complete — measurement pipeline and figures defined. Now run PART 3 / 3 in the next cell.")
MAM_DSK_PART2_OK=True
# =====================================================================================================================================
# AKBASCORE MAM · PERSISTENT NUMERICAL MEMORY FOR FROZEN LANGUAGE MODELS — DEEPSEEK-V2-LITE-CHAT · PART 3 / 3 · SELF-TEST AND GRADIO DEMONSTRATION
# Same demo as PARTS 1 and 2. Run this cell after PART 2 / 3, in the same Colab runtime. It runs a GPU-free self-test of the full
# pipeline (including fail-closed cases) and then prints a public gradio.live link. Choose a case and a target position, press RUN.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 10 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_DSK_PART2_OK"):raise RuntimeError("PART 2 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<UI_BEGIN>>
_SYN_A=["Aq","Bryn","Cav","Dex","Eln","Frax","Gyr","Hex","Iln","Jor","Kex","Lyr"];_SYN_B=["adar","bren","cyr","dax","elor","fyn","grel","hyn"]
class SelfTestInstrument:
    """Synthetic stand-in used ONLY by the service self-test (no GPU, no model). Never used for a real run."""
    def __init__(s,mode="ok"):
        s.kind="SELF-TEST";s.mode=mode;s.calls=0;names=[a+b for a in _SYN_A for b in _SYN_B];s.names=names
        s.cars=[];s.P=PREFIX_EXPECTED
        for w in range(24):
            subj,cur,near,form,role=names[w*4],names[w*4+1],names[w*4+2],names[w*4+3],names[(w*4+5)%len(names)]
            for typ,fact,gold in(("CURRENT",f"The current capital of {subj} is {cur}.",cur),("NEAR",f"The largest city of {subj} is {near}.",near),
                                 ("FORMER",f"The former capital of {subj} was {form}.",form),("ROLE",f"The current capital of {role} is {subj}.",subj)):
                ci=len(s.cars);s.cars.append(dict(id=ci,world=w,type=typ,fact=fact,gold=gold,body_len=12+ci%3))
        s.questions=[]
        for w in range(24):
            t=s._target(w);c=s.cars[t];subj=s.names[w*4]
            q={"CURRENT":f"What is the current capital of {subj}?","FORMER":f"What was the former capital of {subj}?","NEAR":f"What is the largest city of {subj}?","ROLE":f"What is the current capital of {s.names[(w*4+5)%len(s.names)]}?"}[c["type"]]
            s.questions.append(q)
        s.mem=None;s.T=0
        s.info=dict(model_id=MODEL_ID,arch=list(ARCH),dtype="bfloat16",attn="sdpa",gpu="SELF-TEST INSTRUMENT (no GPU)",gpu_total_gib=0.0,torch="n/a",transformers="n/a",gradio="n/a",python="n/a",
            platform="n/a",params=1,rope="n/a",engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,engine_exec_sha256="e"*64,init_seconds=0.0,startup_utc="n/a",
            revision=MODEL_REVISION,heads=16,kv_lora_rank=ARCH[2],qk_rope_head_dim=ARCH[3],routed_experts=64,experts_per_token=6,shared_experts=2,
            weight_sha0=None,sentinel0="a"*64,guard0="b"*64,hooks0=0,
            sentinel_method="synthetic",guard_method="synthetic",full_weight_method="synthetic",forward_counter="synthetic",observer="synthetic")
    def _target(s,w):return w*4+{"CURRENT":0,"NEAR":1,"FORMER":2,"ROLE":3}[["CURRENT","FORMER","NEAR","ROLE"][w%4]]
    def config(s):
        return dict(MODEL_ID=MODEL_ID,REVISION=MODEL_REVISION,CUT=6 if s.mode=="config"else CUT_EXPECTED,source_blobs=dict(ENGINE_SOURCES_EXPECTED),
                    engine_gates={"MODEL_FROZEN":s.mode!="gates","PANEL_SHA_560_575":True},MAX_NEW=24,SEED=552552,PANEL_SEED=550550,EXPECTED_PANEL_SHA=EXPECTED_PANEL_SHA,panel_sha=EXPECTED_PANEL_SHA,prefix_tokens=s.P,
                    n_cartridges=96,n_cases=24,positions=list(POSITIONS),system="synthetic",arch=list(ARCH),engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,engine_exec_sha256="e"*64)
    def status(s):return{"title":"AKBASCORE MAM","reference_test":"TEST596","reference_score":"69/72","panel_sha":EXPECTED_PANEL_SHA,"cut":CUT_EXPECTED}
    def cases(s):return[{"case":w,"type":s.cars[s._target(w)]["type"],"question":s.questions[w],"positions":list(POSITIONS)}for w in range(24)]
    def keys(s,case,position):
        t=s._target(case);d=[((case+1+3*j)%24)*4+(j%4)for j in range(4)]
        return [t]+d if position=="FIRST"else d[:2]+[t]+d[2:]if position=="MIDDLE"else d+[t]
    def car(s,ci):return dict(s.cars[ci])
    def target(s,case):return s._target(case)
    def source_ids(s,ci):return list(range(1,s.P+1))+[1000+ci*20+j for j in range(s.cars[ci]["body_len"])]
    def _qtoks(s,q):return re.findall(r"\n|[^\s\n]+",f"<｜begin▁of▁sentence｜>User: {q} Answer with the name only.\n\nAssistant:")
    def query_ids(s,q):
        ids=[];s.__dict__.setdefault("_tokmap",{})
        for i,t in enumerate(s._qtoks(q)):tid=2000+(sum(map(ord,t))*31+i)%50000;s._tokmap[tid]=t;ids.append(tid)
        return ids
    def decode_each(s,ids):return[s._tokmap.get(t,"?")for t in ids]
    def panel_strings(s):return sorted(set(s.names))
    def sentinel(s):return"a"*64
    def guard(s):s.calls+=1;return"c"*64 if(s.mode=="guard_changes"and s.calls>1)else"b"*64
    def frozen(s):
        s.__dict__["fz_calls"]=s.__dict__.get("fz_calls",0)+1
        return dict(training=False,trainable_tensors=0,lora=False,optimizer=False,hooks=0,foreign_hooks=0,foreign_modules={},sentinel="a"*64,
                    residual_hooks=(["L3:layer"]if(s.mode=="residual_hook"and s.fz_calls>1)else[]))
    def full_weight_sha(s):return None
    def load(s,case,position):
        keys=s.keys(case,position);steps=[];allf=[];old=0;ev_=[]
        for n_,ci in enumerate(keys):
            src=s.source_ids(ci)
            if n_==0:
                f=[dict(kind="ids",n=len(src),ids=src,embeds_zero=None,has_past=False,past_len=None),dict(kind="embeds",n=len(src),ids=None,embeds_zero=True,has_past=False,past_len=None)];new=len(src)
                st=dict(kind="init",cartridge_id=ci,old_len=0,new_len=new,prior_rows_bitwise_unchanged=None,prior_sha_before=None,prior_sha_after=None)
            else:
                q=len(src)-s.P;new=old+q
                f=[dict(kind="ids",n=len(src),ids=src,embeds_zero=None,has_past=False,past_len=None),dict(kind="embeds",n=q,ids=None,embeds_zero=True,has_past=True,past_len=old)]
                if s.mode=="reread"and n_==2:f.insert(1,dict(kind="ids",n=len(s.source_ids(keys[0])),ids=s.source_ids(keys[0]),embeds_zero=None,has_past=False,past_len=None))
                same=not(s.mode=="prior_changed"and n_==3)
                st=dict(kind="append",cartridge_id=ci,old_len=old,new_len=new,prior_rows_bitwise_unchanged=same,prior_sha_before=hashlib.sha256(f"m{old}".encode()).hexdigest(),
                        prior_sha_after=hashlib.sha256(f"m{old}{'' if same else 'x'}".encode()).hexdigest())
            st.update(memory_sha=hashlib.sha256(f"m{new}".encode()).hexdigest(),seconds=0.3,forwards=f,engine_hook_calls=1,engine_init_delta=0.0,engine_unchanged_flag=True);steps.append(st);allf+=f;old=new
            ev_.append(dict(step=n_+1,cartridge_id=ci,position=n_+1,cache_tokens=new,source_replayed_in_consolidation=False,prior_tokens_recomputed=False))
        s.T=old;s.mem=(case,position)
        setup=dict(case=case,position=position,cartridges=[dict(id=ci,type=s.cars[ci]["type"],fact=s.cars[ci]["fact"],is_target=ci==s._target(case))for ci in keys],question=s.questions[case],
                   expected=s.cars[s._target(case)]["gold"],events=ev_,elapsed_seconds=1.5,weights_updated=False,source_replay_during_consolidation=False,prior_memory_recomputed=False)
        return setup,dict(steps=steps,forwards=allf),1.5
    def ask(s,q=None):
        case=s.mem[0];ref=s.questions[case];qq=ref if q is None else q;ids=s.query_ids(qq)
        if s.mode=="source_in_query":ids=s.source_ids(s._target(case))+ids
        exp=s.cars[s._target(case)]["gold"];ans=exp if case%7!=3 else s.names[(case*4+2)%len(s.names)]
        if q is not None and q!=ref:ans="Not provided in the information."
        rec=[dict(kind="ids",n=len(ids),ids=ids,embeds_zero=None,has_past=True,past_len=s.T)]+[dict(kind="ids",n=1,ids=[7],embeds_zero=None,has_past=True,past_len=s.T+len(ids)+i)for i in range(2)]
        isref=qq==ref
        return dict(question=qq,answer=ans,reference_question=isref,expected=exp if isref else None,correct=(correct_rule(ans,exp)if isref else None),elapsed_seconds=0.25,
                    source_text_supplied_to_query=False,model_weights_changed=False),rec,0.25
    def memory_len(s):return s.T
    def memory_sha_now(s):return hashlib.sha256(f"m{s.T}".encode()).hexdigest()
    def memory_profile(s):
        NL,_,LAT,ROT=ARCH;rng=np.random.default_rng(s.T);k=(rng.normal(20,3,(NL,s.T))+np.linspace(0,10,NL)[:,None]).tolist();v=(rng.normal(2,.4,(NL,s.T))).tolist()
        return dict(layers=NL,T=s.T,k_norm=k,v_norm=v,bytes=s.T*NL*(LAT+ROT)*2,kv_heads=1,head_dim=LAT,rope_dim=ROT,dtype="bfloat16")
    def replay(s,progress=None):
        rows=[]
        for c in range(24):
            for p in POSITIONS:
                ro=reference_outcome(c,p);g=s.cars[s._target(c)]["gold"]
                rows.append(dict(case=c,position=p,answer=g if ro["correct"]else ro["answer"],expected=g,correct=ro["correct"]))
                if progress:progress(len(rows),72,rows[-1])
        return dict(score=sum(int(r["correct"])for r in rows),total=72,matches_TEST596=True,panel_sha=EXPECTED_PANEL_SHA,rows=rows,elapsed_seconds=0.1)
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
FILE_KEYS=["zip","json","txt","man","ledger"]
DL_LABELS=["⬇ EVIDENCE PACKAGE (.zip)","⬇ FULL RUN LOG (.json)","⬇ READABLE RUN LOG (.txt)","⬇ RUN MANIFEST (.json)","⬇ SESSION LEDGER (.jsonl)"]
DL_ALL_LABEL=f"⬇ DOWNLOAD ALL {N_FIGS} FIGURES (JPEG, 300 dpi)"
RAW_KEYS=["txt","json","man","ledger"]
N_OUTPUTS=2+2+len(FILE_KEYS)*2+len(RAW_KEYS)
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
def pack(status=_K,gal=_K,files=_K,jpgs=_K,raw=_K):
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
    return tuple(out)
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{html.escape(title)}</div><div class="small">{body}</div></div>'
def pbar_html(done,total):
    pc_=100.0*done/max(1,total);return f'<div style="margin-top:6px;background:#e5e7eb;height:10px"><div style="width:{pc_:.1f}%;height:10px;background:#1d4ed8"></div></div><div class="small">{done}/{total}</div>'
def stage_card(e):return card_html(f"{e['stage']}/{N_STAGES} · {e['title']}",html.escape(e["body"])+pbar_html(e["stage"],N_STAGES),"info")
def case_choices(I):return[f"{c['case']:02d} · {c['type']} · {c['question']}"for c in I.cases()]
def parse_case(v):
    m=re.match(r"\s*(\d{1,2})",str(v or""));return int(m.group(1))if m else None
def ledger_text():return"\n".join(json.dumps(x,ensure_ascii=False)for x in SESSION_LEDGER)
def case_info_html(case_v,pos_v,I=None):
    I=I or INSTR;c=parse_case(case_v);pos=str(pos_v or"MIDDLE").upper()
    if c is None or pos not in POSITIONS:return card_html("Case","Select a case and a target position.","info")
    keys=I.keys(c,pos);t=I.target(c);q=I.cases()[c]["question"]
    rows="".join(f'<div class="small" style="{"font-weight:700" if ci==t else ""}"><span class="mono">C{n_+1} · #{ci:02d}</span> {html.escape(I.car(ci)["fact"])}'
                 +(' <span style="color:#1d4ed8">← target</span>'if ci==t else"")+"</div>"for n_,ci in enumerate(keys))
    body=(f'{rows}<div class="small" style="margin-top:6px"><span class="mono">question</span> {html.escape(q)}<br><span class="mono">expected</span> {html.escape(I.car(t)["gold"])} '
          f'<span style="opacity:.75">(audit metadata; not an input to writing, consolidation or answering)</span><br>'
          f'<span class="mono">order</span> C1 → C5 are written and appended in this order; the source text is not given to the model when the question is asked.<br>'
          +(f'<span class="mono">TEST596</span> this case and position was answered correctly in the reference run.'if reference_outcome(c,pos)["correct"]else
            f'<span class="mono">TEST596</span> <b>one of the three recorded failures</b>: the reference run answered {html.escape(str(reference_outcome(c,pos)["answer"]))} instead of {html.escape(I.car(t)["gold"])}.')+'</div>')
    return card_html(f"Case {c:02d} · target at {pos}",body,"info")
def engine_card_html(I):
    inf=I.info;cfg=I.config();S_=TEST596["score"]
    rows=[("model",f"{inf['model_id']} · revision {inf.get('revision','')[:12]}… · frozen · {inf['dtype']}"),("architecture",arch_txt(tuple(cfg['arch']))),
          ("experts",f"{inf.get('routed_experts')} routed · {inf.get('experts_per_token')} per token · {inf.get('shared_experts')} shared (from the model config)"),("GPU",inf["gpu"]),
          ("cartridge",f"H{cfg['CUT']} residual + {CACHE_SHORT} of layers 0–{cfg['CUT']} · written once per fact"),
          ("consolidation",f"append-only · {upper_txt(cfg['CUT'],cfg['arch'][0])} computed only for new rows · prior rows never recomputed"),("panel",f"locked 24-case panel (shared with the earlier records) · FIRST/MIDDLE/LAST · SHA {cfg['panel_sha'][:16]}…"),
          ("engine",f"{inf['engine_file']} (unchanged) · git blob {inf['engine_blob'][:16]}… · sources "+", ".join(f"{k} {v[:10]}…"for k,v in cfg["source_blobs"].items())),
          ("weights",f"engine fingerprint {inf['sentinel0'][:16]}… · all-parameter guard {inf['guard0'][:16]}…"),
          ("init",f"{inf['init_seconds']:.1f} s" if isinstance(inf['init_seconds'],float)and math.isfinite(inf['init_seconds'])else"reused")]
    body="".join(f'<span class="mono">{html.escape(a)}</span> {html.escape(str(b))}<br>'for a,b in rows)
    body+=(f'<div class="small" style="margin-top:8px"><b>TEST596 REFERENCE · {html.escape(MODEL_FAMILY)}</b> (72 = 24 cases × 3 target positions · not recomputed by single runs)<br>'
           f'<span class="mono">INCR_DC3_MLA</span> <b>{S_["TOTAL"]}/72</b> · FIRST {S_["FIRST"]} · MIDDLE {S_["MIDDLE"]} · LAST {S_["LAST"]} — {html.escape(ARM_DESC["INCR_DC3_MLA"])}<br>'
           +"<br>".join(f'<span class="mono">failure</span> case {c_:02d} {p_} · expected {html.escape(g_)} · answered {html.escape(w_)}'for c_,p_,g_,w_ in TEST596["failures"])
           +f'<br><span class="mono">TEST597</span> failures reproduced · on the three failed slots: joint text {TEST597["joint"][0]}/{TEST597["joint"][1]}, native independent MLA {TEST597["native_indep"][0]}/{TEST597["native_indep"][1]} · no verified readout fix, so no score above 69/72 is claimed'
           +f'<br><span class="mono">scoring</span> {html.escape(CORRECT_RULE_TXT)}<br><span class="mono">TEST597 log SHA-256</span> {TEST597["log_sha256"]}</div>')
    return card_html("Engine status · reference record",body,"on")
READY_HTML=card_html("Ready",f"Choose a case and the target position, optionally type your own question, and press <b>RUN</b>. One run writes five independent cartridges, appends them one by one, "
    f"asks the question without the source text, verifies the frozen weights, seals the payload and renders {N_FIGS} figures (JPEG 300 dpi + vector PDF). "
    "Each run is a LIVE demonstration observation; the TEST596 values above are the reference. Choosing case 11 LAST, 14 LAST or 23 MIDDLE shows one of the three recorded failures.","info")
def is_cuda_error(ex):
    s=f"{type(ex).__name__} {ex}".lower();return"cuda"in s or"device-side assert"in s or"cublas"in s
def run_handler(case_v,pos_v,custom_v):
    global RUN_COUNTER
    c=parse_case(case_v);pos=str(pos_v or"").upper()
    if c is None or pos not in POSITIONS:
        yield pack(card_html("Invalid selection","Choose a case and a target position (FIRST / MIDDLE / LAST).","warn"));return
    if not GPU_LOCK.acquire(blocking=False):
        yield pack(card_html("Busy","Another run is in progress. Please try again in a moment.","warn"));return
    stage_name="initialisation"
    try:
        RUN_COUNTER+=1;run_id=f"{RUN_PREFIX}{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER:04d}-C{c:02d}{pos[0]}"
        prune_runs(4);run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
        preview=[dict(case=x["case"],position=x["position"],correct=x["correct"])for x in SESSION_LEDGER]
        gen=execute_run(INSTR,dict(run_id=run_id,run_dir=run_dir,case=c,position=pos,custom=(custom_v or"")[:300],session=preview+[dict(case=c,position=pos,correct=None)]));first=True
        while True:
            try:e=next(gen)
            except StopIteration as s:B=s.value;break
            stage_name=f"{e['stage']}/{N_STAGES} {e['title']}"
            yield pack(stage_card(e),[],None,None,None)if first else pack(stage_card(e));first=False
        SESSION_LEDGER.append(B["ledger"]);LEDGER_PATH.write_text(ledger_text()+"\n",encoding="utf-8")
        P=B["P"];L=P["live"];ok=L["correct"]
        raw={"txt":B["txt"].read_text(encoding="utf-8"),"json":B["payload"].read_bytes().decode("utf-8"),"man":B["manifest"].read_text(encoding="utf-8"),"ledger":ledger_text()}
        body=(f"<b>{html.escape(B['run_id'])}</b> · LIVE DEMONSTRATION RUN<br>"
              f'<span class="mono">question</span> {html.escape(L["question"])}<br>'
              f'<span class="mono">answer</span> <b>{html.escape(L["answer"]["answer"])}</b> · expected {html.escape(L["expected"])} · <b>{"CORRECT" if ok else "INCORRECT"}</b><br>'
              f'<span class="mono">memory</span> 5 cartridges appended · {L["memory"]["T"]} rows · source token-id forwards during load {L["source_reads"]} (each fact once) · '
              f'previously consolidated rows bitwise unchanged at all 4 appends: {all(s_["prior_rows_bitwise_unchanged"] for s_ in L["steps"][1:])}<br>'
              f'<span class="mono">query</span> {L["query_tokens"]} template tokens over the memory · source text given to the model: none · repeat identical: {L["repeat_identical"]}<br>'
              f'<span class="mono">reference</span> TEST596 for this case and position: {"correct" if L["reference_outcome"]["correct"] else "failed ("+html.escape(str(L["reference_outcome"]["answer"]))+")"} · live outcome identical: {L["matches_reference"]}<br>'
              +(f'<span class="mono">free text</span> {html.escape(L["custom"]["question"])} → {html.escape(L["custom"]["answer"])} (unscored)<br>'if L.get("custom")else"")+
              f"weights unchanged (sentinel + all-parameter guard) · verdict <b>{html.escape(B['verdict'])}</b> · {N_FIGS} figures + PDFs + ZIP · {B['checks']}/{B['checks']} checks PASS<br>"
              f'<span class="mono">payload SHA-256 {B["sha"]}</span>')
        yield pack(card_html(f"{N_STAGES}/{N_STAGES} · Sealed",body,"on"),[(str(p),c_)for p,c_ in B["imgs"]],
                   {"zip":B["zip"],"json":B["payload"],"txt":B["txt"],"man":B["manifest"],"ledger":LEDGER_PATH},[p for p,_ in B["imgs"]],raw)
    except Exception as ex:
        print("="*140);print(f"RUN FAILED — stage: {stage_name}");print(f"{type(ex).__name__}: {ex}");traceback.print_exc();print("="*140)
        kind="AUDIT FAIL — nothing was sealed. "if isinstance(ex,AuditFail)else"";partial=getattr(ex,"partial",None)
        if partial:kind+="The raw outputs gathered so far were preserved (FULL RUN LOG). "
        tip="<br><b>CUDA error: Runtime → Restart runtime, then run PARTS 1–3 again.</b>"if is_cuda_error(ex)else"<br>The full traceback is printed in the Colab console."
        card=card_html("Run failed",f"{html.escape(kind)}Stage: {html.escape(stage_name)}<br>{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:600])}{tip}","err")
        if partial:yield pack(card,[],{"json":partial},None,{"txt":traceback.format_exc(),"json":Path(partial).read_text(encoding="utf-8"),"man":"","ledger":ledger_text()})
        else:yield pack(card,[],None,None,None)
    finally:
        plt.close("all");GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
REPLAY_LABELS=("⬇ LIVE REPLAY (.json)","⬇ LIVE REPLAY (.txt)")
def replay_handler():
    if not GPU_LOCK.acquire(blocking=False):
        yield card_html("Busy","Another run is in progress. Please try again in a moment.","warn"),dl_update(None,REPLAY_LABELS[0]),dl_update(None,REPLAY_LABELS[1]);return
    started=False
    try:
        rid=f"{REPLAY_PREFIX}{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}";prune_runs(2,REPLAY_PREFIX);d=ROOT/rid;d.mkdir(parents=True,exist_ok=True)
        st={"done":0,"last":None,"res":None,"err":None}
        def prog(n,tot,row):st["done"]=n;st["last"]=row
        def work():
            try:st["res"]=replay_report(INSTR,rid,d,prog)
            except Exception as ex:st["err"]=ex;traceback.print_exc()
            finally:GPU_LOCK.release()  # the GPU lock is held until the replay thread itself ends, even if the browser disconnects
        th=threading.Thread(target=work,daemon=True);started=True;th.start()
        while th.is_alive():
            lr=st["last"];tail=(f"<br>last: case {lr['case']:02d} {lr['position']} → {html.escape(str(lr['answer']))} ({'PASS' if lr['correct'] else 'FAIL'})"if lr else"")
            yield card_html("Live replay running",f"Replaying the 72 locked panel runs (INCR_DC3_MLA) with the unchanged engine's demo_verify_all (LIVE; separate from the TEST596 reference).{tail}"+pbar_html(st["done"],72),"info"),SKIP(),SKIP()
            time.sleep(1.5)
        if st["err"]is not None:raise st["err"]
        R=st["res"];o=R["out"];pp=o["per_position"]
        body=(f"<b>LIVE REPLAY {o['score']}/72</b> · FIRST {pp['FIRST']}/24 · MIDDLE {pp['MIDDLE']}/24 · LAST {pp['LAST']}/24 · {o['seconds']} s · weights unchanged: {o['weights_unchanged']}<br>"
              f"Failed cases now: {html.escape(str(o['failed_cases']))} · same three failures as TEST596: {o['failed_cases_match_TEST596']} · outcomes identical to TEST596: {o['outcomes_identical_to_TEST596']}/72. TEST596 reference: {TEST596['score']['TOTAL']}/72. The replay is a separate live measurement and does not replace the reference.<br>"
              f'<span class="mono">replay SHA-256 {R["sha"]}</span>')
        yield card_html("Live replay complete",body,"on"),dl_update(R["json"],REPLAY_LABELS[0]),dl_update(R["txt"],REPLAY_LABELS[1])
    except Exception as ex:
        yield card_html("Live replay failed",f"{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:500])}","err"),dl_update(None,REPLAY_LABELS[0]),dl_update(None,REPLAY_LABELS[1])
    finally:
        if not started:GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
def verify_handler():
    if not GPU_LOCK.acquire(blocking=False):return"Busy: a run is in progress."
    try:
        t=time.perf_counter();g=INSTR.guard();fz=INSTR.frozen();cfg=INSTR.config()
        out=dict(model=MODEL_ID,revision=cfg.get("REVISION"),engine_status=INSTR.status(),engine_gates=cfg.get("engine_gates"),engine_fingerprint_now=fz["sentinel"],
                 all_parameter_guard_now=g,all_parameter_guard_startup=INSTR.info["guard0"],guard_unchanged=g==INSTR.info["guard0"],
                 sentinel_unchanged=fz["sentinel"]==INSTR.info["sentinel0"],trainable_tensors=fz["trainable_tensors"],foreign_hooks=fz["foreign_hooks"],residual_hooks=fz["residual_hooks"],
                 engine_file=ENGINE_FILE,engine_blob=cfg["engine_blob"],engine_blob_expected=ENGINE_BLOB_EXPECTED,source_blobs=cfg.get("source_blobs"),engine_sha256=cfg["engine_sha256"],engine_sha256_expected=ENGINE_SHA256_EXPECTED,
                 engine_exec_sha256=cfg["engine_exec_sha256"],panel_sha=cfg["panel_sha"],consolidation_cut=cfg["CUT"],architecture=cfg["arch"],checked_utc=utc_now(),seconds=round(time.perf_counter()-t,3))
        return json.dumps(jsafe(out),indent=2,ensure_ascii=False)
    finally:GPU_LOCK.release()
# ---------------- service self-test (runs before the public link is printed) ----------------
def service_selftest():
    global QUIET;out=[];d=ROOT/"_selftest";shutil.rmtree(d,ignore_errors=True);d.mkdir(parents=True);(d/"w.txt").write_text("ok");out.append("ROOT writable");QUIET=True
    try:
        for c_,p_,cu in((0,"MIDDLE",""),(3,"FIRST","Who governs this place?")):
            dd=d/f"ok{c_}";dd.mkdir()
            B=drain(execute_run(SelfTestInstrument("ok"),dict(run_id=f"SELFTEST-C{c_:02d}",run_dir=dd,case=c_,position=p_,custom=cu,session=[dict(case=c_,position=p_,correct=True)])))
            assert len(B["imgs"])==N_FIGS and len(B["pdfs"])==N_FIGS and file_ready(B["zip"])and B["verdict"].endswith("SOURCE_FREE_QUERY")
        out.append(f"full pipeline on a synthetic instrument (correct and incorrect answer, with and without free text): {N_FIGS}/{N_FIGS} figures (JPEG 300 dpi + PDF), ZIP, {B['checks']} checks")
        for mode,what in(("guard_changes","a changed weight guard"),("prior_changed","a changed previously consolidated row"),("reread","an earlier source read again during consolidation"),
                         ("source_in_query","source tokens in the query input"),("config","a changed consolidation cut"),("gates","an engine gate reporting FAIL"),
                         ("residual_hook","a hook left on a decoder layer after the run")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestInstrument(mode),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2,case=5,position="LAST",custom="",session=[])))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (nothing sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
        d3=d/"replay";d3.mkdir();R=replay_report(SelfTestInstrument("ok"),"SELFTEST-REPLAY",d3);assert R["out"]["score"]==69 and R["out"]["failed_cases_match_TEST596"] and R["out"]["outcomes_identical_to_TEST596"]==72 and file_ready(R["json"])and file_ready(R["txt"])
        out.append("live replay report (synthetic): JSON + TXT, score re-derived from rows, failed cases compared with TEST596")
    finally:QUIET=False
    if len(pack(READY_HTML))!=N_OUTPUTS or len(pack(READY_HTML,[],None,None,None))!=N_OUTPUTS:raise RuntimeError("callback output count mismatch")
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
      f'<div class="brand">AKBASCORE MAM</div><div class="title">Persistent numerical memory for frozen language models · {html.escape(MODEL_FAMILY)}</div>'
      f'<div class="para">{html.escape(PARADIGM)}.</div>'
      f'<div class="by"><b>{html.escape(DISCOVERY)}.</b><br>{html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(AUTHOR_PLACE)}</div>'
      '<ul class="msg">'+"".join(f"<li>{html.escape(m)}</li>"for m in CORE_MESSAGE)+"</ul>"
      f'<div class="lic">{html.escape(PRIORITY)}</div>'
      f'<div class="lic">{html.escape(LICENSE_NAME)} · {html.escape(COPYRIGHT)} · <a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a> · '
      f'<a href="{SOURCE_URL}" target="_blank" rel="noopener">596.py engine</a> · <a href="{LOCK_URL}" target="_blank" rel="noopener">TEST597 program</a> · <a href="{LOG_URL}" target="_blank" rel="noopener">TEST597 log</a> · '
      f'<a href="{SCALE_LOG_URL}" target="_blank" rel="noopener">590.py (CUT3 MLA functions)</a></div></div>')
CROSS_HTML=card_html(CROSS_TITLE,f'{html.escape(CROSS_TEXT)}<br><b>{html.escape(CROSS_RESULT)}</b><br><br>'
    +"".join(f'<span class="mono">earlier record</span> {html.escape(p["title"])}<br><span class="mono">DOI</span> <a href="{p["url"]}" target="_blank" rel="noopener">{html.escape(p["doi"])}</a> · {html.escape(p["date"])}<br>'for p in PRIOR_PUBS)
    +f'<div class="small" style="opacity:.8;margin-top:6px">{html.escape(CROSS_BOUNDARY)}</div>',"on")
SCOPE_HTML=card_html("Scope of this demonstration",
    "<b>Demonstrated live:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["demonstrated"])+"<br><br><b>Archived (sealed, not run by this demo):</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["archived"])
    +"<br><br><b>Not established:</b><br>"+"<br>".join("• "+html.escape(t)for t in SCOPE["not_established"]),"info")
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
_TITLE="AKBASCORE MAM · DeepSeek-V2-Lite-Chat · persistent numerical memory · Mustafa Akbaş"
if _GR_MAJOR>=6:_blocks=gr.Blocks(title=_TITLE)
else:
    try:_blocks=gr.Blocks(css=CSS,title=_TITLE)
    except TypeError:_blocks=gr.Blocks(title=_TITLE)
_CCH=case_choices(INSTR)
with _blocks as demo:
    gr.HTML(HERO);gr.HTML(CROSS_HTML);gr.HTML(engine_card_html(INSTR))
    with gr.Row():
        case_dd=gr.Dropdown(choices=_CCH,value=_CCH[0],label="Case (locked 24-case panel · reference question)",interactive=True)
        pos_rd=gr.Radio(choices=list(POSITIONS),value="MIDDLE",label="Target cartridge position",interactive=True)
    case_info=gr.HTML(case_info_html(_CCH[0],"MIDDLE"))
    custom_tb=gr.Textbox(value="",label="Optional free-text question (answered from the same numerical memory; not scored)",placeholder="e.g. Which city is the current capital of Zorvan?",lines=1,max_lines=2)
    run_btn=gr.Button("RUN · write 5 cartridges · append-only consolidation · ask without the source · seal",variant="primary",elem_id="kz_run")
    status=gr.HTML(READY_HTML)
    gallery=_gallery(label=f"{N_FIGS} figures (tap to open · JPEG 300 dpi; vector PDFs in the ZIP)",show_label=True)
    dl_all=gr.Button(DL_ALL_LABEL,interactive=False)
    jpg_urls=gr.Textbox(value="",visible=False)
    with gr.Row():
        dlb=[gr.DownloadButton(label=l,value=None,interactive=False)for l in DL_LABELS]
    with gr.Accordion("Direct file links (fallback)",open=False):
        fls=[gr.File(label=n,interactive=False)for n in("ZIP · figures, PDFs, payload, logs","FULL RUN LOG (.json)","READABLE RUN LOG (.txt)","RUN MANIFEST (.json)","SESSION LEDGER (.jsonl)")]
    with gr.Tabs():
        with gr.Tab("READABLE RUN LOG (.txt)"):raw_txt=_tb(lines=18,max_lines=40,label="readable run log")
        with gr.Tab("FULL RUN LOG (.json)"):raw_json=_tb(lines=18,max_lines=40,label="sealed canonical payload")
        with gr.Tab("MANIFEST (.json)"):raw_man=_tb(lines=12,max_lines=30,label="run manifest")
        with gr.Tab("SESSION LEDGER"):raw_led=_tb(lines=10,max_lines=30,label="one line per sealed run in this session")
    with gr.Accordion("Live replay of the locked panel · 72 INCR_DC3_MLA runs · a few minutes on an A100 (reported separately from the TEST596 reference)",open=False):
        rp_btn=gr.Button("Run the live replay (72 runs)")
        rp_status=gr.HTML(card_html("Live replay","Replays all 24 cases × FIRST/MIDDLE/LAST with the engine's own demo_verify_all (unchanged), compares every outcome with TEST596 (69/72, three recorded failures) and writes its own JSON/TXT report. It does not replace the reference.","info"))
        with gr.Row():rp_dl=[gr.DownloadButton(label=l,value=None,interactive=False)for l in REPLAY_LABELS]
    with gr.Accordion("Verification · engine gates · weight fingerprint and guard · engine and panel hashes",open=False):
        ver_btn=gr.Button("Run verification now")
        ver_out=_tb(lines=14,max_lines=40,label="verification record")
    gr.HTML(SCOPE_HTML)
    gr.HTML(f'<div class="kz small" style="opacity:.8;margin:8px 0 18px">{html.escape(DISCOVERY)} · {html.escape(ORG)} · {html.escape(DATE_TXT)} · {html.escape(LICENSE_NAME)} · '
            f'<a href="{LICENSE_URL}" target="_blank" rel="noopener">{html.escape(LICENSE_URL)}</a></div>')
    OUTS=[status,gallery,dl_all,jpg_urls]+dlb+fls+[raw_txt,raw_json,raw_man,raw_led]
    if len(OUTS)!=N_OUTPUTS:raise RuntimeError(f"UI outputs {len(OUTS)} != {N_OUTPUTS}")
    case_dd.change(lambda a,b:case_info_html(a,b),inputs=[case_dd,pos_rd],outputs=case_info)
    pos_rd.change(lambda a,b:case_info_html(a,b),inputs=[case_dd,pos_rd],outputs=case_info)
    run_btn.click(run_handler,inputs=[case_dd,pos_rd,custom_tb],outputs=OUTS)
    rp_btn.click(replay_handler,inputs=None,outputs=[rp_status]+rp_dl)
    ver_btn.click(verify_handler,inputs=None,outputs=ver_out)
    try:dl_all.click(None,inputs=[jpg_urls],outputs=None,js=DL_ALL_JS)
    except TypeError:dl_all.click(None,inputs=[jpg_urls],outputs=None,_js=DL_ALL_JS)
try:demo.queue(default_concurrency_limit=1)
except TypeError:demo.queue(concurrency_count=1)
say("[3/3] LAUNCH (public share link) — open the printed gradio.live link, choose a case and a target position, press RUN.")
_LAUNCH=dict(share=True,allowed_paths=ALLOWED_PATHS,show_error=True,inline=False)
if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
try:demo.launch(**_LAUNCH)
except TypeError as _ex:
    if"css"not in str(_ex):raise
    _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)

