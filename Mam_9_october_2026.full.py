# =====================================================================================================================================
# AKBASCORE MAM · PERSISTENT NUMERICAL MEMORY FOR FROZEN LANGUAGE MODELS — QWEN2.5-7B · PART 1 / 3 · ENGINE
# Cross-model replication record · Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 9 October 2026
# Run PART 1 → PART 2 → PART 3 as three consecutive Google Colab cells in the SAME runtime (GPU: A100 40 GB recommended).
# PART 1 embeds 566.py (the verified Qwen engine) VERBATIM, verifies its git blob SHA-1 against TEST575, executes its verified engine
# section exactly as TEST575 does, loads frozen Qwen2.5-7B-Instruct and attaches read-only measurement. PART 2 adds the fail-closed
# measurement pipeline and figures. PART 3 starts the Gradio demonstration.
# AKBASCORE RESEARCH SOFTWARE LICENSE · Copyright © 2026 Mustafa Akbaş · https://github.com/ceceli33/titan-cognitive-core-v2
#
# ENGINE         : 566.py, embedded byte-for-byte as ENGINE_SRC. Its git blob SHA-1 must equal the TEST575 base blob. As in TEST575, only
#                  the section before the TEST566 additions marker is executed (checkpoint / init / append / answer and the locked panel),
#                  in its own module namespace. The memory algorithm is not modified, recalibrated or re-implemented.
# ADAPTER        : QwenDC3 calls only the engine functions init(·,3), append(·,·,3) and answer(·,·), in the TEST575 incremental order.
# MECHANISM      : each source fact is read once by the frozen model and stored as an independent numerical cartridge
#                  (layer-3 residual H3 + layers 0–3 K/V). Cartridges are appended one by one into a consolidated numerical memory:
#                  layers 4–27 are computed only for the new tokens, attending to the existing memory. Questions are answered from
#                  the numerical memory; the source text is not given to the model at query time. Model weights are never changed.
# INSTRUMENTATION: (1) a recording-only forward pre-hook, attached only while an engine API call runs; it records the kind and length of
#                  every model input (token ids or embeddings), whether input embeddings are all zero, and the installed cache length;
#                  (2) an append observer, attached only during load_case(): it wraps the adapter's init/append calls, calls them unchanged
#                  and checks that every previously consolidated K/V row is bitwise identical after the append; (3) an all-parameter
#                  weight guard plus the TEST575 full-weight SHA-256; (4) re-derivation of every reported quantity from raw records.
# =====================================================================================================================================
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,gc,types
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
DEMO_TITLE="AKBASCORE MAM · Persistent numerical memory for frozen language models"
DEMO_SHORT="AKBASCORE MAM · Qwen2.5-7B incremental DC3 demonstration"
MODEL_FAMILY="Qwen2.5-7B-Instruct"
AUTHOR="Mustafa Akbaş";ORG="AkbasCore AI Teknoloji";DATE_TXT="9 October 2026";DATE_ISO="2026-10-09";AUTHOR_PLACE="Mersin, Türkiye"
COPYRIGHT="Copyright © 2026 Mustafa Akbaş";LICENSE_NAME="AKBASCORE RESEARCH SOFTWARE LICENSE"
LICENSE_URL="https://github.com/ceceli33/titan-cognitive-core-v2"
REPO_RAW="https://github.com/ceceli33/titan-cognitive-core-v2/blob/main/"
SOURCE_URL=REPO_RAW+"566.py";LOCK_URL=REPO_RAW+"575.py";LOG_URL=REPO_RAW+"575.log";SCALE_LOG_URL=REPO_RAW+"572.log"
PARADIGM="A new numerical-memory paradigm for frozen language models"
DISCOVERY="Discovered and developed by Mustafa Akbaş"
# ---- cross-model reference (the earlier, separately published record; its values are NOT Qwen results) ----
PRIOR_PUB=dict(title="AKBASCORE MAM v1.0 — Source-Free Persistent Memory for Frozen LLMs via DC6 Consolidation (Mistral-7B)",doi="10.5281/zenodo.23245358",
               url="https://doi.org/10.5281/zenodo.23245358",date="8 October 2026")
CROSS_TITLE="Cross-Model Replication — From Mistral-7B (DC6) to Qwen2.5-7B (DC3)"
CROSS_TEXT=("The AKBASCORE MAM architecture has been independently implemented and validated across two transformer model families, "
            "using model-specific consolidation boundaries.")
CROSS_BOUNDARY=("Two model families are experimental evidence that the architecture transfers across models; they are not evidence that it works in every language model. "
                "The earlier record is cited as a reference only; every number in this demonstration comes from Qwen2.5-7B-Instruct.")
CROSS_REF=f"Earlier record: {PRIOR_PUB['title']} · DOI {PRIOR_PUB['doi']} · {PRIOR_PUB['url']}"
CROSS_ALLOWED=(CROSS_TITLE,PRIOR_PUB["title"],"Mistral-7B (DC6)","Mistral-7B, DC6")
PRIORITY=("Public technical demonstration of the AKBASCORE MAM architecture on a second transformer model family. "
          "Mustafa Akbaş (AkbasCore AI Teknoloji) discovered and developed this numerical-memory mechanism (first public record: DOI "
          f"{PRIOR_PUB['doi']}, {PRIOR_PUB['date']}); this Qwen2.5-7B replication record is dated 9 October 2026.")
CORE_MESSAGE=["Independent numerical cartridges are written by the frozen model — each source fact is read once.",
              "Cartridges are consolidated append-only into one shared numerical memory.",
              "No earlier source text is replayed during consolidation.",
              "Previously consolidated memory tokens are not recomputed.",
              "Model weights are not changed.",
              "The question is answered without giving the source text to the model again."]
ENGINE_FILE="566.py"
ENGINE_SHA256_EXPECTED="0155cda553e5f08ed2bec53611e49898a5ca24e0d270c3a1274e661bc205bdcd"
ENGINE_BLOB_EXPECTED="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
ENGINE_MARKER="# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged."  # identical to the TEST575 marker
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
WEIGHT_SHA_EXPECTED="b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";ARCH=(28,3584,28,4,128);CUT_EXPECTED=3;PREFIX_EXPECTED=29;N_CASES=24;N_CART=5;POSITIONS=("FIRST","MIDDLE","LAST")
TARGET_SLOT={"FIRST":0,"MIDDLE":2,"LAST":4}
CANON_GPU="NVIDIA A100-SXM4-40GB"
def arch_txt(a=ARCH):return f"{a[0]} layers · hidden {a[1]} · {a[2]} Q heads · {a[3]} KV heads · head {a[4]}"
def arch_short(a=ARCH):return f"{a[0]} L · {a[1]} · {a[2]} Q / {a[3]} KV · {a[4]}"
def upper_txt(cut=CUT_EXPECTED,nl=ARCH[0]):return f"layers {cut+1}–{nl-1}"
# ---- sealed records (read from the archived Qwen experiment logs; never produced or recomputed by this program) ----
TEST575=dict(name="TEST575",lock="76a28d6f10b2c1e431311629b8b3e74f6bceb7048a6e84d5295a35c86cfce4d1",result_sha="579b3fb45f6fc7987694d21a313bb2980b279beb7a7e2de4d23aa69999b8581e",
    weight_sha=WEIGHT_SHA_EXPECTED,base_file=ENGINE_FILE,base_blob=ENGINE_BLOB_EXPECTED,panel_sha=EXPECTED_PANEL_SHA,model=MODEL_ID,gpu=CANON_GPU,torch="2.11.0+cu130",transformers="5.18.0",
    cut=3,prefix=29,cases=24,positions=3,total=72,seconds=258.62,gates=(21,21),trainable=0,
    arms={"INCR_DC3":{"FIRST":24,"MIDDLE":24,"LAST":24,"TOTAL":72},"BATCH_DC3":{"FIRST":24,"MIDDLE":24,"LAST":24,"TOTAL":72},
          "JOINT":{"FIRST":24,"MIDDLE":24,"LAST":24,"TOTAL":72},"NATIVE_INDEP":{"FIRST":10,"MIDDLE":10,"LAST":6,"TOTAL":26},
          "BATCH_BLOCK3":{"FIRST":10,"MIDDLE":11,"LAST":7,"TOTAL":28}},
    answer_identity_joint={"INCR_DC3":72,"BATCH_DC3":72,"NATIVE_INDEP":26,"BATCH_BLOCK3":22},
    counterfactual=dict(upper_kv_changed=72,early_kv_identical=72,answer_changed=50,total=72),
    incr_gain_vs_indep=46,incr_loss_vs_indep=0,
    decision="DEMO GATES PASS — FREEZE VERIFIED CUT3 IMPLEMENTATION")
TEST575_INCR_ANSWERS=[(0,'FIRST','Melket'),(0,'MIDDLE','Melket'),(0,'LAST','Melket'),(1,'FIRST','Branik'),(1,'MIDDLE','Branik'),(1,'LAST','Branik'),(2,'FIRST','Varos'),(2,'MIDDLE','Varos'),(2,'LAST','Varos'),(3,'FIRST','Belnor'),(3,'MIDDLE','Belnor'),(3,'LAST','Belnor'),(4,'FIRST','Selora'),(4,'MIDDLE','Selora'),(4,'LAST','Selora'),(5,'FIRST','Pareth'),(5,'MIDDLE','Pareth'),(5,'LAST','Pareth'),(6,'FIRST','Keldin'),(6,'MIDDLE','Keldin'),(6,'LAST','Keldin'),(7,'FIRST','Feron'),(7,'MIDDLE','Feron'),(7,'LAST','Feron'),(8,'FIRST','Nerith'),(8,'MIDDLE','Nerith'),(8,'LAST','Nerith'),(9,'FIRST','Dorin'),(9,'MIDDLE','Dorin'),(9,'LAST','Dorin'),(10,'FIRST','Valen'),(10,'MIDDLE','Valen'),(10,'LAST','Valen'),(11,'FIRST','Mirev'),(11,'MIDDLE','Mirev'),(11,'LAST','Mirev'),(12,'FIRST','Solith'),(12,'MIDDLE','Solith'),(12,'LAST','Solith'),(13,'FIRST','Toren'),(13,'MIDDLE','Toren'),(13,'LAST','Toren'),(14,'FIRST','Corven'),(14,'MIDDLE','Corven'),(14,'LAST','Corven'),(15,'FIRST','Dervan'),(15,'MIDDLE','Dervan'),(15,'LAST','Dervan'),(16,'FIRST','Verith'),(16,'MIDDLE','Verith'),(16,'LAST','Verith'),(17,'FIRST','Maros'),(17,'MIDDLE','Maros'),(17,'LAST','Maros'),(18,'FIRST','Vindor'),(18,'MIDDLE','Vindor'),(18,'LAST','Vindor'),(19,'FIRST','Barel'),(19,'MIDDLE','Barel'),(19,'LAST','Barel'),(20,'FIRST','Mervor'),(20,'MIDDLE','Mervor'),(20,'LAST','Mervor'),(21,'FIRST','Calvenor'),(21,'MIDDLE','Calvenor'),(21,'LAST','Calvenor'),(22,'FIRST','Mardin'),(22,'MIDDLE','Mardin'),(22,'LAST','Mardin'),(23,'FIRST','Karev'),(23,'MIDDLE','Karev'),(23,'LAST','Karev')]
ARM_ORDER=("INCR_DC3","BATCH_DC3","JOINT","NATIVE_INDEP","BATCH_BLOCK3")
ARM_DESC={"INCR_DC3":"append-only incremental consolidation at layer 3 (this engine; the arm run live)",
          "BATCH_DC3":"all five cartridges consolidated in one pass (shared upper-layer attention)",
          "JOINT":"facts processed together as text (upper bound, not a memory)",
          "NATIVE_INDEP":"independently written native K/V, RoPE re-phased, no consolidation (control)",
          "BATCH_BLOCK3":"counterfactual: same H3 and layers 0–3 K/V, cross-cartridge attention blocked in layers 4–27"}
SCALE=dict(name="TEST572",lock="0989eaea722086f8ac6eae635a1bed534288feac7535833828ecdd05f7fde056",result_sha="8efc22b0d4f3be43f2d39097e0f93a4b0c3ab43d51786380ab584a46789d0a90",
    rows=[(5,8,8),(10,7,8),(20,5,8),(40,6,8)],hop2=[(5,7,8),(10,5,8),(20,3,8),(40,0,8)],unknown=[(5,8,8),(10,8,8),(20,8,8),(40,8,8)],
    note="same CUT3 mechanism · direct recall · 8 tasks per memory size · separate fact panel with an UNKNOWN option (sealed)")
SCOPE={
 "demonstrated":[
  "Five source facts are written independently by frozen Qwen2.5-7B-Instruct: each fact is read once and stored as a numerical cartridge (layer-3 residual H3 + layers 0–3 K/V).",
  "The cartridges are appended one by one into one consolidated numerical memory. For each append, layers 4–27 are computed only for the new tokens, attending to the existing memory; layers 0–3 keys are re-phased to the new positions with the model's own rotary embedding.",
  "Measured in every run: the consolidation forward receives all-zero input embeddings plus the stored H3 (no token ids); no earlier cartridge's source is read again; every previously consolidated K/V row is bitwise identical after each append.",
  "The question is answered from the numerical memory: the query forward receives only the question-template tokens over the installed memory.",
  "Model weights are unchanged (all-parameter guard before and after the run; full-weight SHA-256 at start-up equal to the TEST575 reference)."],
 "archived":[
  "TEST575 sealed results on the locked panel (24 five-cartridge cases × FIRST/MIDDLE/LAST = 72): append-only incremental DC3 72/72, batch DC3 72/72, joint text 72/72, native independent K/V 26/72, counterfactual BATCH_BLOCK3 28/72; incremental and batch answers identical to the joint-text answers in 72/72; 21/21 gates. These are reference values; single live runs do not recompute them. The optional live replay reports its own score separately.",
  "TEST575 counterfactual: with the same H3 and the same layers 0–3 K/V but cross-cartridge attention blocked in layers 4–27, the upper-layer K/V changed in 72/72 cases and the answer changed in 50/72.",
  "TEST572 scale probe of the same CUT3 mechanism (separate fact panel with an UNKNOWN option): direct recall 8/8, 7/8, 5/8, 6/8 with 5, 10, 20, 40 cartridges in memory.",
  "Cross-model reference: the earlier Mistral-7B (DC6) record, DOI "+PRIOR_PUB["doi"]+". Cited as a reference only; it is not a Qwen result."],
 "not_established":[
  "Large memory populations: direct recall is 6/8 at 40 cartridges (TEST572); larger populations are not measured for Qwen.",
  "Multi-step inference that combines several cartridges: two-hop recall falls from 7/8 at 5 cartridges to 0/8 at 40 (TEST572).",
  "Strict KV-only sufficiency: the cartridge is augmented (H3 residual + layers 0–3 K/V).",
  "Questions outside the locked panel template: free-text questions are shown but not scored.",
  "Order-free consolidation: the append uses causal order; equal accuracy at FIRST/MIDDLE/LAST was measured in TEST575 on this panel.",
  "Generality beyond the two tested model families: each family needs its own consolidation boundary; other models are not tested."],
 "notes":[
  "Writing cartridge k reads only fact k (inside the fixed instruction template) once; this is the write event of that cartridge.",
  "The 29-token instruction prefix is written with the first cartridge; later cartridges contribute their body rows only.",
  "Expected answers come from panel construction and are audit metadata; they are not an input to writing, consolidation or answering.",
  "The engine is the verified section of 566.py (git blob "+ENGINE_BLOB_EXPECTED[:12]+"…), executed unchanged as in TEST575; the demo adapter only calls init(·,3), append(·,·,3) and answer(·,·) in the TEST575 incremental order.",
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
ALLOWED_TESTS={"566","572","575"}
CLAIM_RULES=[("capacity-overclaim",r"(?i)\bunlimited\b|\binfinite\b"),("universal-claim",r"(?i)\buniversal\b"),("human-like-claim",r"(?i)human-like"),
    ("model-independence-claim",r"(?i)model-independent"),("all-transformers-claim",r"(?i)all transformers|every transformer|all (?:large )?language models"),("solved-claim",r"(?i)\bsolved\b"),
    ("perfect-claim",r"(?i)\bperfect\b"),("world-first-claim",r"(?i)world'?s first|first in the world|independently proven"),
    ("legal-status-claim",r"(?i)\bpatent(ed|-pending)\b|non-obvious|\bpatentab"),("metaphor",r"(?i)\bbrain\b|\bquantum\b|\bneuron"),
    ("complexity-claim",r"O\(1\)"),("compression-wording",r"(?i)\bcompress")]
# ---- foreign-model leak scanner: no value of the earlier Mistral record may appear as a Qwen value. Only the citation strings are exempt. ----
LEAK_RULES=[("mistral-name",r"(?i)mistral"),("mistral-cut",r"\bH6\b|\bDC6\b|\b(?:layers?|L)\s?0[–-]6\b|\bL7[–-]31\b|\blayers 7[–-]31\b"),
    ("mistral-tests",r"TEST\s?56[03]\b"),("mistral-arch-words",r"\b32 (?:layers|L\b|Q)|\b8 KV\b|\b27-token")]
LEAK_RULES_NUM=[("mistral-hidden",r"(?<![\d.])4096(?![\d.])")]
def _scan_norm(text):
    for a in CROSS_ALLOWED:text=text.replace(a,"<cited>")
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
    assert hit_ans("Melket","Melket") and hit_ans("The capital is Melket.","melket") and not hit_ans("Melketon","Melket");n+=1
    ok_txt=("Qwen2.5-7B-Instruct TEST566 TEST575 TEST572 72/72 append-only DC3 H3 layers 0–3 layers 4–27 28 layers hidden 3584 "+PARADIGM+" "+DISCOVERY+" "+PRIORITY+" "
            +CROSS_TITLE+" "+CROSS_TEXT+" "+CROSS_BOUNDARY+" "+CROSS_REF+" "+" ".join(SCOPE["archived"]+SCOPE["not_established"]+SCOPE["demonstrated"]+SCOPE["notes"]))
    assert claim_scan("t",ok_txt,numeric=True)==[],claim_scan("t",ok_txt,numeric=True);n+=1
    for b in("unlimited memory","universal memory","human-like memory","model-independent","works on all transformers","works in all language models","memory solved","perfect recall",
             "world's first","patent-pending","brain","quantum","O(1)","compressed","TEST528","TEST560","TEST563","Mistral","H6 residual","DC6","layers 0–6","32 layers","8 KV heads","27-token prefix"):
        assert claim_scan("t",b),b;n+=1
    assert claim_scan("t","hidden 4096",numeric=True) and not claim_scan("t","hidden 4096") and not claim_scan("t","t=0.40963 sha 4096ab12cd34ef56",numeric=True);n+=1
    assert strip_words("Zorvan BRAIN carried",["BRAIN"])=="Zorvan  carried";n+=1
    for a in ARM_ORDER:assert sum(TEST575["arms"][a][p]for p in POSITIONS)==TEST575["arms"][a]["TOTAL"],a
    n+=1
    assert len(TEST575_INCR_ANSWERS)==72 and [(c,p)for c,p,_ in TEST575_INCR_ANSWERS]==[(c,p)for c in range(24)for p in POSITIONS];n+=1
    assert TEST575["incr_gain_vs_indep"]==TEST575["arms"]["INCR_DC3"]["TOTAL"]-TEST575["arms"]["NATIVE_INDEP"]["TOTAL"]==46;n+=1
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
say(f"[1/6] CPU self-test PASS ({cpu_selftest()} checks)")
# ---------------- 566.py — embedded verbatim (do not edit; its git blob SHA-1 and SHA-256 are verified below) ----------------
ENGINE_SRC=r'''# TEST566 — AKBASCORE MAM · QWEN CAUSAL CONSOLIDATION CONTROLS
# TEST564 FROZEN BACKBONE · JOINT / INDEP / DC / NOCROSS · CUTS 1/2/3/6
# SAME 24-CASE PANEL SHA · SAME TOKENS / LOGICAL POSITIONS
# NO TRAINING · NO RETRIEVAL · NO ROUTER · NO SOURCE REPLAY AT ASK
import os,sys,subprocess,importlib.util,time,random,re,hashlib,json,gc
for m in ("torch","transformers","accelerate"):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",m])
import torch,numpy as np,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="566";MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SEED=552552;PANEL_SEED=550550;CUTS=[1,2,3,6];MAX_NEW=16;DEVICE=torch.device("cuda");DTYPE=torch.bfloat16
ARMS=["JOINT","INDEP","DC","NOCROSS"];CASES=list(range(8));SLOTS=["FIRST","MIDDLE","LAST"]
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
torch.set_grad_enabled(False);T0=time.perf_counter()
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold()).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
print("="*132);print("TEST566 — QWEN MAM · CAUSAL CONSOLIDATION CONTROLS");print("="*132)
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
assert (NL,H,QH,KVH,HD)==(28,3584,28,4,128),(NL,H,QH,KVH,HD)
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
print("ARCH:",NL,H,QH,KVH,HD,"CUTS:",CUTS,"ARMS:",ARMS)
FP=[model.model.layers[i].self_attn.o_proj.weight for i in (0,7,14,21,27)]+[model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()
BASE=[
("Zorvan","Melket","Dravel","Oakhaven","Pelnor"),("Kelvar","Nareth","Solven","Branik","Tarsen"),("Tarev","Luneth","Varos","Cedran","Mireth"),("Belnor","Arven","Dorel","Kesmar","Falven"),
("Ravik","Selora","Terven","Maldor","Nerik"),("Nemor","Calven","Istral","Pareth","Dovren"),("Darsen","Velora","Keldin","Orvek","Sarnel"),("Feron","Talven","Merith","Sovran","Belvik"),
("Larev","Nerith","Calder","Veyron","Tormek"),("Torven","Elsar","Marvek","Dorin","Kaleth"),("Selnor","Kareth","Valen","Ordan","Mervek"),("Mirev","Taldor","Neris","Kelmar","Sorvik"),
("Varen","Solith","Deran","Malvek","Cordan"),("Kelor","Ardin","Velmar","Toren","Narell"),("Narev","Belith","Corven","Sareth","Dorvik"),("Dervan","Mirel","Talvek","Orsen","Kelron"),
("Calnor","Verith","Naldor","Seren","Parvek"),("Parel","Dorven","Kelith","Maros","Tervik"),("Sorven","Tarell","Vindor","Nelmar","Calrek"),("Barel","Corith","Laven","Derik","Solmar"),
("Ralen","Mervor","Talith","Kesven","Noreth"),("Norel","Valdor","Serith","Calvenor","Darvek"),("Tervan","Orel","Mardin","Velos","Karven"),("Karev","Solen","Dareth","Mirven","Talrek")]
W=[];CAR=[];KEY={}
for wi,(s,current,near,former,role) in enumerate(BASE):
    facts=[("CURRENT",f"The current capital of {s} is {current}.",current),("NEAR",f"The largest city of {s} is {near}.",near),("FORMER",f"The former capital of {s} was {former}.",former),("ROLE",f"The current capital of {role} is {s}.",s)]
    queries={"CURRENT":f"What is the current capital of {s}?","FORMER":f"What was the former capital of {s}?","NEAR":f"What is the largest city of {s}?","ROLE":f"What is the current capital of {role}?"}
    W.append(queries)
    for typ,f,gold in facts:
        KEY[(wi,typ)]=len(CAR);CAR.append(dict(world=wi,type=typ,fact=f,gold=gold))
SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
PREFIX=f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nINFORMATION:\n"
def source(f):return PREFIX+f+"\n\n"
def suffix(q):return f"\nQUESTION:\n{q}\n\nANSWER:\n<|im_end|>\n<|im_start|>assistant\n"
PIDS=tok(PREFIX,add_special_tokens=False).input_ids;P=len(PIDS)
for c in CAR:
    ids=tok(source(c["fact"]),add_special_tokens=False).input_ids
    if ids[:P]!=PIDS:raise RuntimeError("Prefix mismatch")
    c["body"]=ids[P:]
rng=random.Random(PANEL_SEED);pool=list(range(96));rng.shuffle(pool);PANEL=[];cursor=0;cycle=["CURRENT","FORMER","NEAR","ROLE"]
for wi in range(24):
    typ=cycle[wi%4];target=KEY[(wi,typ)];others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]["world"]==wi or any(CAR[x]["world"]==CAR[ci]["world"] for x in others):continue
        others.append(ci)
    PANEL.append(dict(case=wi,target=target,others=others))
PANEL_SHA=sha(PANEL);assert PANEL_SHA==EXPECTED_PANEL_SHA,(PANEL_SHA,EXPECTED_PANEL_SHA)
print("PANEL SHA:",PANEL_SHA,"PASS","PREFIX:",P)
def ordered(z,slot):
    d=z["others"];t=z["target"]
    return [t]+d if slot=="FIRST" else d[:2]+[t]+d[2:] if slot=="MIDDLE" else d+[t]
def pairs(c):
    if hasattr(c,"layers"):return tuple((l.keys,l.values) for l in c.layers)
    return tuple(zip(c.key_cache,c.value_cache))
def clone(c):return tuple((k.detach().clone(),v.detach().clone()) for k,v in pairs(c))
def cache_build(x):
    c=DynamicCache()
    for i,(k,v) in enumerate(x):c.update(k,v,i)
    return c
def replace(out,h):
    if isinstance(out,tuple):return (h,)+out[1:]
    if isinstance(out,list):return [h]+out[1:]
    return h
def half(x):
    a,b=x.chunk(2,-1);return torch.cat((-b,a),-1)
def rope(pos):
    dummy=torch.zeros((1,len(pos),H),device=DEVICE,dtype=DTYPE)
    c,s=model.model.rotary_emb(dummy,pos.unsqueeze(0))
    return c.unsqueeze(1).float(),s.unsqueeze(1).float()
def rephase(k,old,new):
    if torch.equal(old,new):return k
    co,so=rope(old);cn,sn=rope(new);x=k.float()
    x=x*co-half(x)*so
    return (x*cn+half(x)*sn).to(k.dtype)
@torch.no_grad()
def checkpoint(ci,cut):
    ids=PIDS+CAR[ci]["body"];n=len(ids);x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,device=DEVICE);state={}
    def capture(module,args,out):state["h"]=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
    h=model.model.layers[cut].register_forward_hook(capture)
    try:
        out=model(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        kv=clone(out.past_key_values)
        return dict(h=state["h"],kv=kv,pos=pos,body=n-P)
    finally:h.remove()
@torch.no_grad()
def init(ci,cut):
    cp=checkpoint(ci,cut);n=cp["h"].shape[1]
    def inject(module,args,out):return replace(out,cp["h"])
    h=model.model.layers[cut].register_forward_hook(inject)
    try:
        pos=torch.arange(n,device=DEVICE)
        out=model(inputs_embeds=torch.zeros((1,n,H),device=DEVICE,dtype=DTYPE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        upper=clone(out.past_key_values)
        return cp["kv"][:cut+1]+upper[cut+1:]
    finally:h.remove()
@torch.no_grad()
def append(memory,ci,cut):
    cp=checkpoint(ci,cut);oldn=memory[0][0].shape[-2];q=cp["body"];pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp["h"][:,P:,:]
    if body.shape!=(1,q,H):raise RuntimeError("Body shape mismatch")
    def inject(module,args,out):
        current=out[0] if isinstance(out,(tuple,list)) else out
        if current.shape!=body.shape:raise RuntimeError("Injection shape mismatch")
        return replace(out,body)
    h=model.model.layers[cut].register_forward_hook(inject)
    try:
        out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        grown=list(clone(out.past_key_values))
        for li in range(cut+1):
            ok,ov=memory[li];k,v=cp["kv"][li];k=rephase(k[:,:,P:,:],cp["pos"][P:],pos);v=v[:,:,P:,:]
            grown[li]=(torch.cat((ok,k),-2),torch.cat((ov,v),-2))
        result=tuple(grown)
        for (ok,ov),(nk,nv) in zip(memory,result):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError("APPEND-ONLY BITWISE FAIL")
        if any(k.shape[-2]!=oldn+q for k,v in result):raise RuntimeError("Cache length mismatch")
        return result
    finally:h.remove()
@torch.no_grad()
def answer(memory,q):
    cache=cache_build(memory);n=memory[0][0].shape[-2];ids=tok(suffix(q),add_special_tokens=False).input_ids
    x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,n+len(ids),device=DEVICE)
    out=model(input_ids=x,past_key_values=cache,attention_mask=torch.ones((1,n+len(ids)),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True);gen=[]
    for _ in range(MAX_NEW):
        t=int(next_id.item())
        if t in (tok.eos_token_id,tok.convert_tokens_to_ids("<|im_end|>")):break
        gen.append(t);n=cache.get_seq_length();pos=torch.tensor([n],device=DEVICE)
        out=model(input_ids=next_id,past_key_values=cache,attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(gen,skip_special_tokens=True).strip()
# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.
@torch.no_grad()
def joint(keys):
    ids=PIDS+sum((CAR[ci]["body"] for ci in keys),[])
    n=len(ids);x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,device=DEVICE)
    out=model(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    return clone(out.past_key_values)
@torch.no_grad()
def independent(keys,cut,mode):
    if mode not in ("INDEP","NOCROSS"):raise ValueError(mode)
    memory=None;oldn=0
    for j,ci in enumerate(keys):
        cp=checkpoint(ci,cut) if mode=="INDEP" else None
        kv=cp["kv"] if mode=="INDEP" else init(ci,cut)
        if j==0:
            memory=kv;oldn=kv[0][0].shape[-2]
            continue
        q=len(CAR[ci]["body"]);oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE);grown=[]
        for li,((ok,ov),(k,v)) in enumerate(zip(memory,kv)):
            kk=rephase(k[:,:,P:,:],oldpos,newpos);vv=v[:,:,P:,:]
            grown.append((torch.cat((ok,kk),-2),torch.cat((ov,vv),-2)))
        result=tuple(grown)
        for (ok,ov),(nk,nv) in zip(memory,result):
            if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError("INDEPENDENT APPEND BITWISE FAIL")
        memory=result;oldn+=q
    return memory
@torch.no_grad()
def consolidated(keys,cut):
    memory=init(keys[0],cut)
    for ci in keys[1:]:memory=append(memory,ci,cut)
    return memory
def expected_length(keys):return P+sum(len(CAR[ci]["body"]) for ci in keys)
def validate(memory,keys):
    n=expected_length(keys)
    if len(memory)!=NL:raise RuntimeError("Layer count mismatch")
    for li,(k,v) in enumerate(memory):
        if k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD):raise RuntimeError(f"Cache shape mismatch L{li}: {k.shape}/{v.shape}")
        if not torch.isfinite(k).all() or not torch.isfinite(v).all():raise RuntimeError(f"Nonfinite cache L{li}")
    return n
def score_rows(rows):
    return {s:sum(int(r["ok"]) for r in rows if r["slot"]==s) for s in SLOTS}
RESULTS=[];RAW=[];ERRORS=[]
print("\n[1/5] JOINT natural reference — same 24 questions...")
for wi in CASES:
    z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
    for slot in SLOTS:
        keys=ordered(z,slot)
        try:
            mem=joint(keys);validate(mem,keys);a=answer(mem,q);ok=hit(a,gold)
            RAW.append(dict(arm="JOINT",cut=None,case=wi,slot=slot,gold=gold,answer=a,ok=ok))
            print(f"JOINT CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
            del mem
        except Exception as e:
            ERRORS.append(dict(arm="JOINT",case=wi,slot=slot,error=repr(e)));print("JOINT ERROR:",repr(e),flush=True);raise
JROWS=[r for r in RAW if r["arm"]=="JOINT"];JS=score_rows(JROWS)
print("JOINT:",JS,"TOTAL:",sum(JS.values()),"/24")
print("\n[2/5] INDEP native independent KV reference...")
for wi in CASES:
    z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
    for slot in SLOTS:
        keys=ordered(z,slot)
        try:
            mem=independent(keys,0,"INDEP");validate(mem,keys);a=answer(mem,q);ok=hit(a,gold)
            RAW.append(dict(arm="INDEP",cut=None,case=wi,slot=slot,gold=gold,answer=a,ok=ok))
            print(f"INDEP CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
            del mem
        except Exception as e:
            ERRORS.append(dict(arm="INDEP",case=wi,slot=slot,error=repr(e)));print("INDEP ERROR:",repr(e),flush=True);raise
IROWS=[r for r in RAW if r["arm"]=="INDEP"];IS=score_rows(IROWS)
print("INDEP:",IS,"TOTAL:",sum(IS.values()),"/24")
print("\n[3/5] CUT-wise DC and NOCROSS...")
for cut in CUTS:
    tcut=time.perf_counter()
    for arm in ("DC","NOCROSS"):
        rows=[];failed=None
        print("\n"+"="*110);print("CUT",cut,"ARM",arm,flush=True)
        try:
            for wi in CASES:
                z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
                for slot in SLOTS:
                    keys=ordered(z,slot)
                    mem=consolidated(keys,cut) if arm=="DC" else independent(keys,cut,"NOCROSS")
                    validate(mem,keys);a=answer(mem,q);ok=hit(a,gold)
                    r=dict(arm=arm,cut=cut,case=wi,slot=slot,gold=gold,answer=a,ok=ok)
                    rows.append(r);RAW.append(r)
                    print(f"{arm:7s} CUT={cut:02d} CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | gold={gold} | answer={a!r}",flush=True)
                    del mem
        except Exception as e:
            failed=repr(e);ERRORS.append(dict(arm=arm,cut=cut,error=failed));print("ARM ERROR:",failed,flush=True)
        ss=score_rows(rows);complete=len(rows)==24 and failed is None
        RESULTS.append(dict(arm=arm,cut=cut,scores=ss,total=sum(ss.values()),complete=complete,error=failed))
        print(f"ARM={arm} CUT={cut} SCORE={sum(ss.values())}/24 COMPLETE={complete} FIRST={ss['FIRST']}/8 MIDDLE={ss['MIDDLE']}/8 LAST={ss['LAST']}/8",flush=True)
        if not complete:raise RuntimeError(f"INCOMPLETE ARM {arm} CUT {cut}: {failed}")
    print("CUT",cut,"SECONDS",round(time.perf_counter()-tcut,2))
    torch.cuda.empty_cache();gc.collect()
print("\n[4/5] Causal comparison and paired transitions...")
MAP={(r["arm"],r["cut"],r["case"],r["slot"]):r for r in RAW}
DIAG=[]
for cut in CUTS:
    for arm in ("DC","NOCROSS"):
        gain=loss=matchj=matchi=ansj=0
        for wi in CASES:
            for slot in SLOTS:
                a=MAP[(arm,cut,wi,slot)];j=MAP[("JOINT",None,wi,slot)];i=MAP[("INDEP",None,wi,slot)]
                gain+=int(not i["ok"] and a["ok"]);loss+=int(i["ok"] and not a["ok"])
                matchj+=int(a["ok"]==j["ok"]);matchi+=int(a["ok"]==i["ok"]);ansj+=int(norm(a["answer"])==norm(j["answer"]))
        d=dict(arm=arm,cut=cut,gain=gain,loss=loss,net=gain-loss,ok_identity_joint=matchj,ok_identity_indep=matchi,answer_identity_joint=ansj)
        DIAG.append(d)
        print(f"{arm:7s} CUT={cut:02d} GAIN={gain:02d} LOSS={loss:02d} NET={gain-loss:+03d} OK-ID-JOINT={matchj:02d}/24 ANS-ID-JOINT={ansj:02d}/24")
print("\n[5/5] Final research record...")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
INTEGRITY=len(JROWS)==24 and len(IROWS)==24 and len(RAW)==24*(2+2*len(CUTS)) and len({(r["arm"],r["cut"],r["case"],r["slot"]) for r in RAW})==len(RAW) and not ERRORS
DCROWS=[r for r in RESULTS if r["arm"]=="DC"];NCROWS=[r for r in RESULTS if r["arm"]=="NOCROSS"]
print("="*132);print("TEST566 — FINAL RESEARCH RECORD");print("="*132)
print("MODEL:",MODEL_ID,"GPU:",torch.cuda.get_device_name(0))
print("PANEL SHA:",PANEL_SHA,"INTEGRITY:","PASS" if INTEGRITY else "FAIL")
print(f"JOINT   FIRST={JS['FIRST']:02d}/8 MIDDLE={JS['MIDDLE']:02d}/8 LAST={JS['LAST']:02d}/8 TOTAL={sum(JS.values()):02d}/24")
print(f"INDEP   FIRST={IS['FIRST']:02d}/8 MIDDLE={IS['MIDDLE']:02d}/8 LAST={IS['LAST']:02d}/8 TOTAL={sum(IS.values()):02d}/24")
for cut in CUTS:
    for arm in ("DC","NOCROSS"):
        r=next(x for x in RESULTS if x["arm"]==arm and x["cut"]==cut);s=r["scores"]
        print(f"{arm:7s} CUT={cut:02d} FIRST={s['FIRST']:02d}/8 MIDDLE={s['MIDDLE']:02d}/8 LAST={s['LAST']:02d}/8 TOTAL={r['total']:02d}/24")
print("WEIGHT SENTINEL:","PASS" if WEIGHT_OK else "FAIL","TRAINABLE:",TRAINABLE)
print("TOTAL SECONDS:",round(time.perf_counter()-T0,2))
LOCK={"test":TEST,"model":MODEL_ID,"panel_sha":PANEL_SHA,"seed":SEED,"panel_seed":PANEL_SEED,"cuts":CUTS,"arms":ARMS,"joint":"native concatenated source prefill","indep":"native independently forged complete KV with RoPE rephase","dc":"TEST564 Hcut + early KV + append-only upper consolidation","nocross":"independent Hcut checkpoint upper reconstruction, no cross-cartridge prefill","training":False,"retrieval":False,"router":False}
FINAL={"lock":LOCK,"joint":JS,"indep":IS,"results":RESULTS,"diag":DIAG,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE,"integrity":INTEGRITY}
print("LOCK SHA:",sha(LOCK));print("RESULT SHA:",sha(FINAL))
print("INTERPRETATION: INDEP is native isolated KV concatenation. NOCROSS is independently reconstructed checkpoint KV concatenation; both prohibit cross-cartridge attention during memory construction.")
print("CAUTION: NOCROSS is not an exact DC ablation: independent upper-layer reconstruction may also change local numerical states. Interpret DC-NOCROSS differences as a mechanism-screening result, not isolated proof of cross-attention necessity.")
print("CAUTION: Same panel as TEST564; this is calibration, not independent validation. Correctness alone does not establish numerical identity to JOINT.")
if not INTEGRITY or not WEIGHT_OK or TRAINABLE!=0:raise RuntimeError("FINAL INTEGRITY FAILURE")
print("DECISION: CAUSAL SCREEN COMPLETE — REVIEW DC vs NOCROSS vs INDEP BEFORE TEST567")
print("="*132)
'''
_ENGINE_BYTES=ENGINE_SRC.encode("utf-8")
ENGINE_SHA256=hashlib.sha256(_ENGINE_BYTES).hexdigest();ENGINE_BLOB=git_blob_sha1(_ENGINE_BYTES)
if ENGINE_BLOB!=ENGINE_BLOB_EXPECTED or ENGINE_SHA256!=ENGINE_SHA256_EXPECTED:
    raise RuntimeError(f"Embedded 566.py differs from the TEST575-verified file (blob {ENGINE_BLOB} / SHA-256 {ENGINE_SHA256}). Re-paste PART 1 unchanged.")
if ENGINE_SRC.count(ENGINE_MARKER)!=1:raise RuntimeError("566.py engine-section boundary not found exactly once")
ENGINE_EXEC_SRC=ENGINE_SRC.split(ENGINE_MARKER,1)[0];ENGINE_EXEC_SHA256=hashlib.sha256(ENGINE_EXEC_SRC.encode("utf-8")).hexdigest()
say(f"[2/6] Engine source verified · {ENGINE_FILE} · git blob {ENGINE_BLOB} (= TEST575 base blob) · SHA-256 {ENGINE_SHA256}")
STARTUP_UTC=utc_now()
if"ENG"in globals()and getattr(globals()["ENG"],"__engine_blob__",None)==ENGINE_BLOB and globals().get("DEMO_ENGINE")is not None:
    say("[3/6] Engine already initialised in this runtime — reusing it (no second model load).");ENGINE_INIT_SECONDS=globals().get("ENGINE_INIT_SECONDS",float("nan"))
else:
    say("[3/6] Executing the verified 566.py engine section in its own module namespace (loads frozen Qwen2.5-7B-Instruct, about 1–3 minutes) …")
    say("      (the next lines up to the panel SHA are printed by 566.py itself while its verified section runs)");_t=time.perf_counter()
    ENG=types.ModuleType("mam_test566_qwen_engine");ENG.__file__=ENGINE_FILE+" (embedded verbatim; verified section)";sys.modules[ENG.__name__]=ENG
    exec(compile(ENGINE_EXEC_SRC,ENGINE_FILE,"exec"),ENG.__dict__)
    ENG.__engine_blob__=ENGINE_BLOB;ENG.__engine_sha256__=ENGINE_SHA256
    # ---------------- demo adapter: calls ONLY the verified engine functions, in the TEST575 incremental order ----------------
    class QwenDC3:
        """init(first,3) → append(memory,next,3) × 4 → answer(memory,question). No other computation touches the memory."""
        CUT=3
        def __init__(s):
            s.tok=ENG.tok;s.model=ENG.model;s.nl=ENG.NL;s.h=ENG.H;s.prefix_ids=list(ENG.PIDS);s.p=ENG.P;s.car=ENG.CAR;s.panel=ENG.PANEL;s.panel_sha=ENG.PANEL_SHA
            s.lock=threading.RLock();s._memory=None;s._selection=None;s._events=[]
        def _question(s,case):return ENG.W[case][s.car[s.panel[case]["target"]]["type"]]
        def _gold(s,case):return s.car[s.panel[case]["target"]]["gold"]
        def _keys(s,case,position):return list(ENG.ordered(s.panel[case],position))
        def _init_incremental(s,ci):return ENG.init(ci,s.CUT)
        def _append_incremental(s,memory,ci):return ENG.append(memory,ci,s.CUT)
        def _answer_layers(s,memory,q):return ENG.answer(memory,q)
        def _validate(s,memory,keys):
            n=s.p+sum(len(s.car[ci]["body"])for ci in keys)
            if len(memory)!=ENG.NL:raise RuntimeError("LAYER COUNT MISMATCH")
            for li,(k,v)in enumerate(memory):
                shape=(1,ENG.KVH,n,ENG.HD)
                if tuple(k.shape)!=shape or tuple(v.shape)!=shape:raise RuntimeError(f"CACHE SHAPE L{li}: {tuple(k.shape)}/{tuple(v.shape)}")
                if not torch.isfinite(k).all()or not torch.isfinite(v).all():raise RuntimeError(f"NONFINITE CACHE L{li}")
            return n
        def cases(s):return[{"case":i,"type":s.car[z["target"]]["type"],"question":s._question(i),"positions":list(POSITIONS)}for i,z in enumerate(s.panel)]
        def status(s):
            return{"title":"AKBASCORE MAM","developer":ORG,"author":AUTHOR,"date":DATE_ISO,"model":ENG.MODEL_ID,"cut":s.CUT,"panel_sha":s.panel_sha,"reference_test":"TEST575",
                   "reference_score":"72/72","loaded_case":s._selection,"cartridges":0 if s._selection is None else 5,"events":list(s._events)}
        def load_case(s,case=0,position="MIDDLE"):
            case=int(case);position=str(position).upper()
            if not 0<=case<len(s.panel)or position not in POSITIONS:raise ValueError("case must be 0..23; position FIRST/MIDDLE/LAST")
            with s.lock,torch.no_grad():
                t0=time.perf_counter();keys=s._keys(case,position);memory=None;events=[]
                for step,ci in enumerate(keys):
                    if memory is None:memory=s._init_incremental(ci)
                    else:
                        new=s._append_incremental(memory,ci);del memory;memory=new
                    events.append({"step":step+1,"cartridge_id":ci,"position":step+1,"cache_tokens":int(memory[0][0].shape[-2]),"source_replayed_in_consolidation":False,"prior_tokens_recomputed":False})
                s._validate(memory,keys);s._memory=memory;s._selection={"case":case,"position":position};s._events=events;tgt=s.panel[case]["target"]
                return{"case":case,"position":position,"cartridges":[{"id":ci,"type":s.car[ci]["type"],"fact":s.car[ci]["fact"],"is_target":ci==tgt}for ci in keys],"question":s._question(case),
                       "expected":s._gold(case),"events":events,"elapsed_seconds":round(time.perf_counter()-t0,3),"weights_updated":False,"source_replay_during_consolidation":False,"prior_memory_recomputed":False}
        def ask(s,question=None):
            with s.lock,torch.no_grad():
                if s._memory is None:raise RuntimeError("Call load_case() first")
                case=s._selection["case"];ref=s._question(case);q=ref if question is None else str(question)
                t0=time.perf_counter();answer=s._answer_layers(s._memory,q);isref=q==ref;gold=s._gold(case)if isref else None
                return{"question":q,"answer":answer,"reference_question":isref,"expected":gold,"correct":bool(ENG.hit(answer,gold))if isref else None,"elapsed_seconds":round(time.perf_counter()-t0,3),
                       "source_text_supplied_to_query":False,"model_weights_changed":False}
        def run_case(s,case=0,position="MIDDLE"):setup=s.load_case(case,position);return{"setup":setup,"answer":s.ask()}
        def run_reference_panel(s,progress=None):
            rows=[];t0=time.perf_counter()
            for case in range(len(s.panel)):
                for position in POSITIONS:
                    r=s.run_case(case,position);rows.append({"case":case,"position":position,"answer":r["answer"]["answer"],"expected":r["answer"]["expected"],"correct":r["answer"]["correct"]})
                    if progress:progress(len(rows),72,rows[-1])
            score=sum(int(r["correct"])for r in rows)
            return{"score":score,"total":72,"matches_TEST575":score==72,"panel_sha":s.panel_sha,"rows":rows,"elapsed_seconds":round(time.perf_counter()-t0,2)}
    DEMO_ENGINE=QwenDC3();torch.cuda.synchronize();ENGINE_INIT_SECONDS=time.perf_counter()-_t
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
    """TEST575 check: no forward hooks or pre-hooks left on any decoder layer or attention module."""
    out=[]
    for li,layer in enumerate(model.model.layers):
        for label,module in(("layer",layer),("attention",layer.self_attn)):
            if module._forward_hooks or module._forward_pre_hooks:out.append(f"L{li}:{label}")
    return out
def optimizer_present():return any(isinstance(v,torch.optim.Optimizer)for v in list(globals().values())+list(ENG.__dict__.values()))
def _tensor_bytes(x):
    x=x.detach().contiguous().cpu()
    if x.dtype==torch.bfloat16:x=x.view(torch.uint16)
    return x.numpy().tobytes()
def memory_sha(layers,rows=None):
    """SHA-256 over the raw bytes of every K and V tensor (optionally only the first `rows` positions)."""
    h=hashlib.sha256()
    for k,v in layers:
        if rows is not None:k=k[:,:,:rows,:];v=v[:,:,:rows,:]
        h.update(_tensor_bytes(k));h.update(_tensor_bytes(v))
    return h.hexdigest()
def full_weight_sha(model):
    """TEST575 method, unchanged: SHA-256 over (name, shape, dtype, raw bytes) of every parameter."""
    h=hashlib.sha256()
    with torch.inference_mode():
        for name,p in model.named_parameters():
            h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
            for part in p.detach().contiguous().view(torch.uint8).reshape(-1).split(16*1024*1024):h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()
class Instrument:
    """Thin adapter around the engine adapter's API, plus measurement. Every scientific quantity comes from the engine itself."""
    def __init__(self,D,init_seconds):
        self.D=D;self.kind="ENGINE";m=D.model;c=m.config
        self.info=dict(model_id=ENG.MODEL_ID,arch=[ENG.NL,ENG.H,ENG.QH,ENG.KVH,ENG.HD],dtype=str(next(m.parameters()).dtype).replace("torch.",""),
            attn=getattr(c,"_attn_implementation",None),gpu=torch.cuda.get_device_name(0),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,python=sys.version.split()[0],platform=platform.platform(),
            params=sum(p.numel()for p in m.parameters()),rope=str(getattr(c,"rope_parameters",None)or getattr(c,"rope_theta",None)),engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256,
            engine_blob=ENGINE_BLOB,engine_exec_sha256=ENGINE_EXEC_SHA256,init_seconds=init_seconds,startup_utc=STARTUP_UTC,
            sentinel_method="TEST566 sentinel: SHA-256 over 16 slices × 256 values of 7 weight tensors (o_proj of layers 0/7/14/21/27, final norm, lm_head)",
            guard_method="all-parameter guard: SHA-256 over (name, shape, dtype, sum, |sum|, sum of squares) of every parameter tensor",
            full_weight_method="TEST575 full-weight SHA-256 over (name, shape, dtype, raw bytes) of every parameter",
            forward_counter="recording-only forward pre-hook on the causal-LM module, attached only during an engine API call",
            observer="append observer: wraps the adapter's init(·,3)/append(·,·,3) calls during load_case() only; calls them unchanged; compares prior K/V rows bitwise")
        t=time.perf_counter();self.info["weight_sha0"]=full_weight_sha(m);self.info["weight_sha_seconds"]=time.perf_counter()-t
        self.info["sentinel0"]=self.sentinel();self.info["sentinel_engine_load"]=ENG.S0;self.info["guard0"]=self.guard();self.info["hooks0"]=hook_stats(m)[0]
    # ---- read-only views of engine state ----
    def config(self):
        D=self.D
        return dict(MODEL_ID=ENG.MODEL_ID,CUT=D.CUT,ENGINE_CUTS=list(ENG.CUTS),MAX_NEW=ENG.MAX_NEW,SEED=ENG.SEED,PANEL_SEED=ENG.PANEL_SEED,EXPECTED_PANEL_SHA=ENG.EXPECTED_PANEL_SHA,panel_sha=D.panel_sha,
                    prefix_tokens=D.p,n_cartridges=len(D.car),n_cases=len(D.panel),positions=list(ENG.SLOTS),system=ENG.SYSTEM,arch=[ENG.NL,ENG.H,ENG.QH,ENG.KVH,ENG.HD],
                    engine_sha256=hashlib.sha256(ENGINE_SRC.encode("utf-8")).hexdigest(),engine_blob=git_blob_sha1(ENGINE_SRC.encode("utf-8")),engine_exec_sha256=ENGINE_EXEC_SHA256)
    def status(self):return self.D.status()
    def cases(self):return self.D.cases()
    def keys(self,case,position):return list(self.D._keys(int(case),str(position)))
    def car(self,ci):c=self.D.car[ci];return dict(id=ci,world=c["world"],type=c["type"],fact=c["fact"],gold=c["gold"],body_len=len(c["body"]))
    def target(self,case):return int(self.D.panel[case]["target"])
    def source_ids(self,ci):return list(self.D.prefix_ids)+list(self.D.car[ci]["body"])
    def query_ids(self,q):return list(self.D.tok(ENG.suffix(q),add_special_tokens=False).input_ids)
    def decode_each(self,ids):return[self.D.tok.decode([t])for t in ids]
    def panel_strings(self):return sorted({w for row in ENG.BASE for w in row})
    # ---- integrity ----
    def sentinel(self):return ENG.sentinel()
    def guard(self):
        h=hashlib.sha256()
        with torch.inference_mode():
            for name,p in self.D.model.named_parameters():
                x=p.detach().float();vals=(float(x.sum().item()),float(x.abs().sum().item()),float((x*x).sum().item()))
                h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode());h.update(np.asarray(vals,dtype=np.float64).tobytes());del x
        torch.cuda.empty_cache();return h.hexdigest()
    def full_weight_sha(self):return full_weight_sha(self.D.model)
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
        D=self.D;rec,_rec=self._recorder();steps=[];orig_init=D._init_incremental;orig_app=D._append_incremental
        def init_obs(ci):
            a=len(rec);t=time.perf_counter();res=orig_init(ci);torch.cuda.synchronize()
            steps.append(dict(kind="init",cartridge_id=int(ci),old_len=0,new_len=int(res[0][0].shape[-2]),fw=(a,len(rec)),prior_rows_bitwise_unchanged=None,prior_sha_before=None,prior_sha_after=None,
                              memory_sha=memory_sha(res),seconds=time.perf_counter()-t));return res
        def app_obs(memory,ci):
            a=len(rec);oldn=int(memory[0][0].shape[-2]);sb=memory_sha(memory);t=time.perf_counter();res=orig_app(memory,ci);torch.cuda.synchronize();dt_=time.perf_counter()-t
            same=all(torch.equal(res[l][0][:,:,:oldn],memory[l][0])and torch.equal(res[l][1][:,:,:oldn],memory[l][1])for l in range(len(memory)))
            steps.append(dict(kind="append",cartridge_id=int(ci),old_len=oldn,new_len=int(res[0][0].shape[-2]),fw=(a,len(rec)),prior_rows_bitwise_unchanged=bool(same),
                              prior_sha_before=sb,prior_sha_after=memory_sha(res,oldn),memory_sha=memory_sha(res),seconds=dt_));return res
        h=D.model.register_forward_pre_hook(_rec,with_kwargs=True);D._init_incremental=init_obs;D._append_incremental=app_obs
        try:torch.cuda.synchronize();t=time.perf_counter();setup=D.load_case(case,position);torch.cuda.synchronize();dt_=time.perf_counter()-t
        finally:
            h.remove()
            for nm in("_init_incremental","_append_incremental"):D.__dict__.pop(nm,None)
        for s in steps:s["forwards"]=rec[s["fw"][0]:s["fw"][1]];del s["fw"]
        return setup,dict(steps=steps,forwards=rec),dt_
    def ask(self,q=None):
        D=self.D;rec,_rec=self._recorder();h=D.model.register_forward_pre_hook(_rec,with_kwargs=True)
        try:torch.cuda.synchronize();t=time.perf_counter();res=D.ask(q);torch.cuda.synchronize();dt_=time.perf_counter()-t
        finally:h.remove()
        return res,rec,dt_
    def memory_len(self):return int(self.D._memory[0][0].shape[-2])if self.D._memory is not None else 0
    def memory_sha_now(self):return memory_sha(self.D._memory)
    def memory_profile(self):
        """Display-only statistics of the consolidated memory (mean L2 norm over KV heads of every K and V row, per layer)."""
        M=self.D._memory;kn=[];vn=[];nb=0
        with torch.inference_mode():
            for k,v in M:
                kn.append([float(x)for x in k.float().norm(dim=-1).mean(dim=1)[0].tolist()]);vn.append([float(x)for x in v.float().norm(dim=-1).mean(dim=1)[0].tolist()])
                nb+=k.numel()*k.element_size()+v.numel()*v.element_size()
        return dict(layers=len(M),T=len(kn[0]),k_norm=kn,v_norm=vn,bytes=int(nb),kv_heads=int(M[0][0].shape[1]),head_dim=int(M[0][0].shape[-1]),dtype=str(M[0][0].dtype).replace("torch.",""))
    def replay(self,progress=None):return self.D.run_reference_panel(progress)
    def sync(self):torch.cuda.synchronize()
    def reset_peak(self):torch.cuda.reset_peak_memory_stats()
    def peak_gib(self):return torch.cuda.max_memory_allocated()/2**30
say("[4/6] Measurement adapter · TEST566 weight sentinel · all-parameter guard · TEST575 full-weight SHA-256 (about 1–2 minutes) …")
INSTR=Instrument(DEMO_ENGINE,ENGINE_INIT_SECONDS)
say(f"[5/6] Full-weight SHA-256 {INSTR.info['weight_sha0']} ({INSTR.info['weight_sha_seconds']:.1f} s) · TEST575 reference {WEIGHT_SHA_EXPECTED}")
_cfg=INSTR.config();_st=INSTR.status();_fz=INSTR.frozen()
ENGINE_LOCK=[("566.py git blob SHA-1 = TEST575 base blob",_cfg["engine_blob"]==ENGINE_BLOB_EXPECTED),("566.py SHA-256 = embedded reference",_cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED),
 ("verified engine section executed (boundary marker found once, as in TEST575)",ENGINE_SRC.count(ENGINE_MARKER)==1 and ENGINE_EXEC_SRC==ENGINE_SRC.split(ENGINE_MARKER,1)[0]),
 ("model = Qwen/Qwen2.5-7B-Instruct",_cfg["MODEL_ID"]==MODEL_ID),(f"architecture {arch_txt()}",tuple(INSTR.info["arch"])==ARCH==tuple(_cfg["arch"])),
 ("consolidation cut = layer 3 (one of the TEST566 cuts)",_cfg["CUT"]==CUT_EXPECTED and CUT_EXPECTED in _cfg["ENGINE_CUTS"]),("canonical instruction prefix = 29 tokens",_cfg["prefix_tokens"]==PREFIX_EXPECTED),
 ("locked panel SHA-256",_cfg["panel_sha"]==EXPECTED_PANEL_SHA==_cfg["EXPECTED_PANEL_SHA"]),("96 cartridges · 24 cases · FIRST/MIDDLE/LAST",_cfg["n_cartridges"]==96 and _cfg["n_cases"]==N_CASES and tuple(_cfg["positions"])==POSITIONS),
 ("full-weight SHA-256 = TEST575 reference",INSTR.info["weight_sha0"]==WEIGHT_SHA_EXPECTED),
 ("adapter status: TEST575 reference 72/72",_st.get("reference_test")=="TEST575"and _st.get("reference_score")=="72/72"),
 ("weight sentinel = TEST566 value at model load",_fz["sentinel"]==INSTR.info["sentinel0"]==INSTR.info["sentinel_engine_load"]),("trainable tensors = 0",_fz["trainable_tensors"]==0),
 ("model in eval mode, no LoRA, no optimizer",(not _fz["training"])and not _fz["lora"]and not _fz["optimizer"]),("no foreign forward/backward hooks",_fz["foreign_hooks"]==0),
 ("no residual hooks on decoder layers or attention",not _fz["residual_hooks"])]
_bad=[n for n,ok in ENGINE_LOCK if not ok]
if _bad:raise RuntimeError(f"ENGINE LOCK FAILED: {_bad}")
say(f"[6/6] Engine lock PASS · {len(ENGINE_LOCK)} checks · init {ENGINE_INIT_SECONDS:.1f}s · guard {INSTR.info['guard0'][:24]}… · sentinel {INSTR.info['sentinel0'][:24]}…")
say("PART 1 / 3 complete. Now run PART 2 / 3 in the next cell.");say("="*140)
MAM_QWEN_PART1_OK=True
#<<ENGINE_END>>
# =====================================================================================================================================
# AKBASCORE MAM · PERSISTENT NUMERICAL MEMORY FOR FROZEN LANGUAGE MODELS — QWEN2.5-7B · PART 2 / 3 · MEASUREMENT PIPELINE AND FIGURES
# Same demo as PART 1. Run this cell after PART 1 / 3, in the same Colab runtime; then run PART 3 / 3.
# Defines the fail-closed measurement run (write → append-only consolidation → source-free question → verification → sealed payload,
# logs, ZIP), six figures (JPEG 300 dpi + vector PDF) and the optional live replay of the locked panel. Nothing here changes the engine.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 9 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_QWEN_PART1_OK"):raise RuntimeError("PART 1 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
#<<CORE_BEGIN>>
import unicodedata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch
from matplotlib.lines import Line2D
from PIL import Image
ROOT=Path("/content/AKBASCORE_MAM_QWEN_DC3")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_MAM_QWEN_DC3")
ROOT.mkdir(parents=True,exist_ok=True);ALLOWED_PATHS=[str(ROOT)]
N_STAGES=8
def ev(stage,title,body):return dict(stage=stage,title=title,body=body)
def file_ready(p):return p is not None and Path(p).is_file()and Path(p).stat().st_size>0
RUN_PREFIX="MAMQWEN-";REPLAY_PREFIX="MAMQWENREPLAY-"
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
    a("This is a LIVE run of the frozen Qwen2.5-7B incremental-DC3 engine (566.py, verified section). It is not TEST575; the sealed TEST575 values are reproduced verbatim, never recomputed.")
    a(CROSS_TITLE+". "+CROSS_TEXT);a(CROSS_BOUNDARY);a(CROSS_REF)
    for k_,v in(("RUN ID",P["run_id"]),("START UTC",P["run_start_utc"]),("END UTC",P["run_end_utc"]),("SEALED UTC",sealed_utc),("VERDICT",P["verdict"]),
        ("MODEL",f"{cfg['MODEL_ID']} · frozen · {P['model']['dtype']} · attention {P['model']['attn']}"),("ARCHITECTURE",arch_txt(tuple(P["model"]["arch"]))),
        ("GPU",f"{E_['gpu']} (TEST575 reference GPU: {CANON_GPU}; same: {P['hardware']['same']})"),("TORCH / TRANSFORMERS / GRADIO",f"{E_['torch']} / {E_['transformers']} / {E_['gradio']}"),
        ("ENGINE FILE",f"{P['engine']['file']} · git blob {P['engine']['blob']} (= TEST575 base blob) · SHA-256 {P['engine']['sha256']}"),("ENGINE SECTION",f"executed as in TEST575 · SHA-256 {P['engine']['exec_sha256']}"),
        ("CONSOLIDATION CUT",f"layer {cfg['CUT']} (cartridge = H{cfg['CUT']} residual + layers 0–{cfg['CUT']} K/V; {upper_txt(cfg['CUT'],cfg['arch'][0])} consolidate)"),
        ("PANEL",f"locked 24-case panel · SHA-256 {cfg['panel_sha']}"),("TEST575 LOCK",TEST575["lock"]),("TEST575 RESULT SHA",TEST575["result_sha"]),("FULL-WEIGHT SHA-256",f"{P['frozen']['weight_sha_startup']} (TEST575 reference {TEST575['weight_sha']})")):a(f"{k_:<30}: {v}")
    a("");a("CASE");a(Dd)
    a(f"case {L['case']:02d} · target position {L['position']} (slot {L['target_slot']+1} of 5) · question: {L['question']}");a(f"expected answer (audit metadata): {L['expected']}")
    for n_,c in enumerate(L["cartridges"],1):a(f"C{n_} · cartridge #{c['id']:02d} · {c['type']:<8} · {c['body_len']:>2} body tokens · {c['fact']}"+("   ← TARGET"if c["is_target"]else""))
    a("");a("WRITE + APPEND-ONLY CONSOLIDATION · load_case()  [LIVE]");a(Dd)
    for n_,s in enumerate(L["steps"],1):
        a(f"step {n_} · {s['kind']:<6} · cartridge #{s['cartridge_id']:02d} · memory {s['old_len']} → {s['new_len']} tokens (+{s['new_len']-s['old_len']}) · {1000*s['seconds']:.1f} ms")
        for f in s["forwards"]:a(f"         forward: {fw_txt(f)}")
        if s["kind"]=="append":a(f"         previously consolidated rows 0…{s['old_len']-1}: bitwise unchanged = {s['prior_rows_bitwise_unchanged']} · SHA {s['prior_sha_before'][:16]}… → {s['prior_sha_after'][:16]}…")
    a(f"final memory: {L['memory']['T']} tokens × {L['memory']['layers']} layers · {L['memory']['kv_heads']} KV heads × {L['memory']['head_dim']} · {L['memory']['bytes']/2**20:.2f} MiB ({L['memory']['dtype']}) · SHA {L['memory_sha'][:24]}…")
    a(f"source token-id forwards during load: {L['source_reads']} (one per cartridge, each its own fact) · consolidation forwards with all-zero input embeddings: {L['zero_embed_forwards']}")
    a("");a("SOURCE-FREE QUESTION · ask()  [LIVE]");a(Dd)
    a(f"question: {ans['question']}");a(f"answer  : {ans['answer']}");a(f"expected: {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'} (re-derived: {L['correct_rederived']})")
    a(f"query input = question template only: {L['query_tokens']} token ids over the installed memory ({L['query_past']} tokens) · generation forwards {L['gen_forwards']} · {1000*ans['elapsed_seconds']:.0f} ms")
    a(f"repeat ask(): identical answer = {L['repeat_identical']} · memory SHA unchanged by querying = {L['memory_unchanged_by_query']}")
    a(f"sealed TEST575 INCR_DC3 answer for this case and position: {L['sealed_answer']} · live answer identical: {L['matches_sealed']} (reported, not a gate)")
    if L.get("custom"):cu=L["custom"];a(f"free-text question (unscored): {cu['question']} → {cu['answer']} · query tokens {cu['query_tokens']}")
    a("");a("SEALED TEST575 REFERENCE · QWEN2.5-7B (archived comparative controls; not produced by this run)");a(Dd)
    for arm in ARM_ORDER:v=TEST575["arms"][arm];a(f"{arm:<13} FIRST {v['FIRST']:>2}/24 · MIDDLE {v['MIDDLE']:>2}/24 · LAST {v['LAST']:>2}/24 · TOTAL {v['TOTAL']}/72 · answer identity with JOINT {TEST575['answer_identity_joint'].get(arm,72)}/72 · {ARM_DESC[arm]}")
    cf=TEST575["counterfactual"];a(f"counterfactual (BATCH_BLOCK3 vs BATCH_DC3): layers 0–3 K/V identical {cf['early_kv_identical']}/72 · upper-layer K/V changed {cf['upper_kv_changed']}/72 · answer changed {cf['answer_changed']}/72")
    a(f"incremental DC3 vs native independent K/V: gain {TEST575['incr_gain_vs_indep']} · loss {TEST575['incr_loss_vs_indep']} · gates {TEST575['gates'][0]}/{TEST575['gates'][1]} PASS · trainable {TEST575['trainable']}")
    a(f"scale probe ({SCALE['name']}, {SCALE['note']}): direct recall "+" · ".join(f"{n}: {k}/{m}"for n,k,m in SCALE["rows"])+" · two-hop "+" · ".join(f"{n}: {k}/{m}"for n,k,m in SCALE["hop2"]))
    a(f"{SCALE['name']} lock {SCALE['lock']} · result {SCALE['result_sha']}")
    a("");a("FROZEN MODEL");a(Dd)
    for k_ in("weight_sha_startup","sentinel_startup","sentinel_before","sentinel_after","guard_startup","guard_before","guard_after"):a(f"{k_:<18}: {fz[k_]}")
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
    chk("566.py git blob = TEST575 base blob · SHA-256 = embedded reference",cfg["engine_blob"]==ENGINE_BLOB_EXPECTED and cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED)
    chk(f"model, architecture, consolidation cut (layer {CUT_EXPECTED}) and {PREFIX_EXPECTED}-token prefix as in TEST575",cfg["MODEL_ID"]==MODEL_ID and tuple(cfg["arch"])==ARCH and cfg["CUT"]==CUT_EXPECTED and cfg["prefix_tokens"]==PREFIX_EXPECTED)
    chk("locked panel (SHA-256) · 96 cartridges · 24 cases",cfg["panel_sha"]==EXPECTED_PANEL_SHA and cfg["n_cartridges"]==96 and cfg["n_cases"]==N_CASES)
    chk("full-weight SHA-256 at start-up = TEST575 reference",info["weight_sha0"]==WEIGHT_SHA_EXPECTED)
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
    prof=I.memory_profile();chk(f"memory profile: {ARCH[0]} layers × T rows · {ARCH[3]} KV heads × {ARCH[4]}",prof["layers"]==ARCH[0] and prof["T"]==T and prof["kv_heads"]==ARCH[3] and prof["head_dim"]==ARCH[4])
    say(f"   memory {T} tokens · 5 steps · prior rows bitwise unchanged at every append · {1000*dt:.0f} ms")
    # ---- 3 source-free question ----
    question=setup["question"]
    yield ev(3,"Source-free question","ask(): the question template is processed over the numerical memory; the source text is not given to the model.")
    a1,fw_a,dt=I.ask(None);raw["answer"]=a1;tm["ask"]+=dt;qids=I.query_ids(question)
    chk("query forward input = question-template tokens only, over the installed memory",len(fw_a)>=1 and fw_a[0]["kind"]=="ids"and fw_a[0]["ids"]==qids and fw_a[0]["has_past"]and fw_a[0]["past_len"]==T)
    chk("generation forwards: one new token each, over the growing cache",all(f["kind"]=="ids"and f["n"]==1 and f["has_past"]and f["past_len"]==T+len(qids)+n_ for n_,f in enumerate(fw_a[1:])))
    chk("ask(): reference question of this case; engine flags: no source text, weights unchanged",a1["question"]==question and a1["reference_question"]is True and a1["expected"]==expected and a1["source_text_supplied_to_query"]is False and a1["model_weights_changed"]is False)
    cr=hit_ans(a1["answer"],expected);chk("correctness re-derived from the answer text = engine result",cr==a1["correct"])
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
    peak=I.peak_gib();same_hw=info["gpu"]==CANON_GPU;sealed_ans=next(a_ for c_,p_,a_ in TEST575_INCR_ANSWERS if c_==case and p_==position)
    live=dict(case=case,position=position,target_slot=slot,cartridges=cars,question=question,expected=expected,setup=setup,steps=steps,forwards_load=trace["forwards"],
              source_reads=len(ids_fw),zero_embed_forwards=sum(1 for f in emb_fw if f["embeds_zero"]),memory=prof,memory_sha=mem_sha,segments=segments(steps,P_),
              answer=a1,answer_repeat=a2,correct=bool(a1["correct"]),correct_rederived=bool(cr),repeat_identical=a2["answer"]==a1["answer"],memory_unchanged_by_query=sha_q==mem_sha,
              query_tokens=len(qids),query_token_text=I.decode_each(qids),query_past=T,gen_forwards=len(fw_a)-1,forwards_ask=fw_a,custom=cu,
              sealed_answer=sealed_ans,matches_sealed=norm_ans(a1["answer"])==norm_ans(sealed_ans),
              session=[dict(r,correct=(bool(a1["correct"])if r.get("correct")is None else bool(r["correct"])))for r in(ctl.get("session")or[])])
    P={"schema":"akbascore.mam.qwen.dc3.demo.run.v1","demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"place":AUTHOR_PLACE,"copyright":COPYRIGHT,
       "license":{"name":LICENSE_NAME,"url":LICENSE_URL},"priority_record":PRIORITY,"paradigm":PARADIGM,"discovery":DISCOVERY,"core_message":CORE_MESSAGE,
       "cross_model":{"title":CROSS_TITLE,"statement":CROSS_TEXT,"boundary":CROSS_BOUNDARY,"earlier_record":PRIOR_PUB,"note":"reference only; no value of the earlier record is used as a result here"},
       "run_id":run_id,"run_start_utc":run_start_utc,"run_end_utc":utc_now(),"verdict":verdict,
       "model":{"id":cfg["MODEL_ID"],"arch":info["arch"],"dtype":info["dtype"],"attn":info["attn"],"params":info["params"],"rope":info.get("rope")},
       "environment":{k_:info[k_]for k_ in("gpu","gpu_total_gib","torch","transformers","gradio","python","platform")},"hardware":{"reference_gpu":CANON_GPU,"this_run_gpu":info["gpu"],"same":bool(same_hw)},
       "engine":{"file":info["engine_file"],"sha256":cfg["engine_sha256"],"blob":cfg["engine_blob"],"exec_sha256":cfg["engine_exec_sha256"],"config":cfg,"status_before":st0,"init_seconds":info["init_seconds"],
                 "instrument":I.kind,"adapter":"QwenDC3: init(·,3) → append(·,·,3) × 4 → answer(·,·), TEST575 incremental order","source":SOURCE_URL,"lock_program":LOCK_URL,"log":LOG_URL},
       "instrumentation":{"forward_counter":info["forward_counter"],"observer":info["observer"]},"live":live,
       "frozen":{"weight_sha_startup":info["weight_sha0"],"weight_sha_reference":WEIGHT_SHA_EXPECTED,"full_weight_method":info["full_weight_method"],"sentinel_startup":info["sentinel0"],"sentinel_before":fz0["sentinel"],"sentinel_after":fz1["sentinel"],"guard_startup":info["guard0"],"guard_before":g_before,"guard_after":g_after,
                 "trainable_tensors":fz1["trainable_tensors"],"lora":fz1["lora"],"optimizer":fz1["optimizer"],"training_mode":fz1["training"],"foreign_hooks_before":fz0["foreign_hooks"],
                 "foreign_hooks_after":fz1["foreign_hooks"],"residual_hooks_before":fz0["residual_hooks"],"residual_hooks_after":fz1["residual_hooks"],"guard_method":info["guard_method"],"sentinel_method":info["sentinel_method"]},
       "archived":{"test575":TEST575,"arm_description":ARM_DESC,"scale_probe":SCALE,"note":"Sealed Qwen reference values; not produced or recomputed by this run. Comparative control arms were run in TEST575, not in this demo."},
       "scope":SCOPE,"checks_pre_seal":list(checks),"timing":dict(tm),"gpu":{"peak_allocated_gib":peak}}
    P=jsafe(P);allow=(info["gpu"],CANON_GPU)
    hits=claim_scan("payload",strip_words(json.dumps(P,ensure_ascii=False),strip),allow=allow)
    chk("claim-discipline and foreign-model scan of the payload: 0 hits",not hits);P["claim_scan"]={"rules":len(CLAIM_RULES)+1+len(LEAK_RULES),"hits":0,"scope":"payload JSON text; panel names and free text excluded; earlier-record citation exempt"}
    pb=canon(P);sha=hashlib.sha256(pb).hexdigest();payload_name=f"{run_id}_FULL_RUN_LOG_payload.json";pp=run_dir/payload_name;t_seal=time.perf_counter();pp.write_bytes(pb);say("payload sealed:",sha)
    chk("payload SHA-256 verification after write",hashlib.sha256(pp.read_bytes()).hexdigest()==sha)
    names=dict(payload=payload_name,manifest=f"run_manifest_{run_id}.json",txt=f"{run_id}_READABLE_RUN_LOG.txt",summary=f"{run_id}_SUMMARY.json",jsonl=f"{run_id}_RAW_RECORDS.jsonl",images=f"figures_manifest_{run_id}.json",zip=f"AKBASCORE_MAM_QWEN_DC3_EVIDENCE_{run_id}.zip")
    mp=run_dir/names["manifest"];sealed_utc=utc_now()
    manifest={"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"copyright":COPYRIGHT,"license":LICENSE_URL,"priority_record":PRIORITY,"run_id":run_id,"payload_file":payload_name,
              "payload_sha256":sha,"payload_bytes":len(pb),"sealed_utc":sealed_utc,"verdict":verdict,"engine_sha256":cfg["engine_sha256"],"engine_blob":cfg["engine_blob"],"test575_lock":TEST575["lock"],"test575_result_sha":TEST575["result_sha"],
              "weight_sha_startup":info["weight_sha0"],"model":MODEL_ID,
              "canonicalization":"json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False, allow_nan=False), UTF-8",
              "note":"Artifact integrity seal: confirms the payload is unchanged since sealing. It is not a scientific proof and not a third-party verification."}
    mp.write_text(json.dumps(manifest,indent=2,sort_keys=True,ensure_ascii=False),encoding="utf-8")
    summary={"run_id":run_id,"verdict":verdict,"case":case,"position":position,"question":question,"answer":a1["answer"],"expected":expected,"correct":a1["correct"],
             "cartridges":[{"slot":n_+1,"id":c["id"],"type":c["type"],"target":c["is_target"]}for n_,c in enumerate(cars)],
             "memory_growth":[s["new_len"]for s in steps],"prior_rows_bitwise_unchanged":[s["prior_rows_bitwise_unchanged"]for s in steps[1:]],"source_reads_during_load":len(ids_fw),
             "query_tokens":len(qids),"memory_tokens":T,"weights_unchanged":True,"engine_sha256":cfg["engine_sha256"],"payload_sha256":sha,"model":MODEL_ID,
             "sealed_answer_TEST575":sealed_ans,"live_answer_matches_sealed":norm_ans(a1["answer"])==norm_ans(sealed_ans),
             "sealed_test575_reference":{a_:TEST575["arms"][a_]["TOTAL"]for a_ in ARM_ORDER}}
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
# ---------------- optional LIVE replay of the locked panel (72 runs; reported separately from the sealed TEST575 reference) ----------------
def replay_report(I,run_id,run_dir,progress=None):
    info=I.info;t0=time.perf_counter();start=utc_now();g0=I.guard();s0=I.sentinel()if hasattr(I,"sentinel")else info["sentinel0"]
    if g0!=info["guard0"]:raise AuditFail("all-parameter guard before replay differs from startup")
    R=I.replay(progress);g1=I.guard()
    if g1!=g0:raise AuditFail("all-parameter guard changed during replay")
    rows=R["rows"];score=sum(int(r["correct"])for r in rows)
    if score!=R["score"]or len(rows)!=72:raise AuditFail("replay score does not re-derive from its rows")
    per={p:sum(int(r["correct"])for r in rows if r["position"]==p)for p in POSITIONS}
    sealed={(c_,p_):a_ for c_,p_,a_ in TEST575_INCR_ANSWERS}
    for r in rows:r["sealed_answer"]=sealed[(r["case"],r["position"])];r["matches_sealed"]=norm_ans(r["answer"])==norm_ans(r["sealed_answer"])
    same=sum(int(r["matches_sealed"])for r in rows)
    out={"schema":"akbascore.mam.qwen.dc3.live_replay.v1","model":MODEL_ID,"demo":DEMO_TITLE,"author":AUTHOR,"organisation":ORG,"date":DATE_ISO,"license":LICENSE_URL,"run_id":run_id,"start_utc":start,"end_utc":utc_now(),
         "status":"LIVE REPLAY — separate from the sealed TEST575 reference","score":score,"total":72,"per_position":per,"matches_TEST575_total":score==72,"answers_identical_to_sealed_TEST575":same,"panel_sha":R["panel_sha"],
         "engine_sha256":ENGINE_SHA256_EXPECTED,"engine_blob":ENGINE_BLOB_EXPECTED,"guard_before":g0,"guard_after":g1,"weights_unchanged":g1==g0==info["guard0"],"gpu":info["gpu"],"seconds":round(time.perf_counter()-t0,2),
         "rows":rows,"sealed_reference":{"name":"TEST575","INCR_DC3":TEST575["arms"]["INCR_DC3"],"result_sha":TEST575["result_sha"]}}
    pb=canon(out);sha=hashlib.sha256(pb).hexdigest();jp=run_dir/f"{run_id}_LIVE_REPLAY.json";jp.write_bytes(pb)
    S="="*120;txt=[S,f"{DEMO_TITLE} — {MODEL_FAMILY} · LIVE REPLAY OF THE LOCKED PANEL (INCR_DC3)",f"{DISCOVERY} · {ORG} · {DATE_TXT} · {LICENSE_NAME} · {LICENSE_URL}",S,
         "This replay is a LIVE measurement in this runtime. It is reported separately from the sealed TEST575 record and does not replace it.",
         f"run {run_id} · {start} → {out['end_utc']} · {out['seconds']} s · GPU {info['gpu']}",f"LIVE score {score}/72 · FIRST {per['FIRST']}/24 · MIDDLE {per['MIDDLE']}/24 · LAST {per['LAST']}/24",
         f"sealed TEST575 incremental DC3: {TEST575['arms']['INCR_DC3']['TOTAL']}/72 (result SHA {TEST575['result_sha']}) · live answers identical to the sealed TEST575 answers: {same}/72",f"weights unchanged (all-parameter guard before = after = startup): {out['weights_unchanged']}",
         f"payload SHA-256 {sha}","-"*120]+[f"case {r['case']:02d} {r['position']:<6} expected {r['expected']:<10} answer {r['answer']!r} {'PASS'if r['correct']else'FAIL'} · sealed {r['sealed_answer']} {'=' if r['matches_sealed'] else '≠'}"for r in rows]+[S]
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
                "Keywords":f"model {MODEL_ID}; engine 566.py git blob {ENGINE_BLOB}; engine sha256 {ENGINE_SHA256}; TEST575 result {TEST575['result_sha']}; license {LICENSE_URL}"})
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
    fig.text(.5,.010,mt(f"live run {ctx['run_id']} · {MODEL_FAMILY} · 566.py engine (blob {ENGINE_BLOB[:12]}…; not TEST575 itself) · TEST575 result {TEST575['result_sha'][:12]}… · payload {ctx['sha'][:12]}… · Fig. {k}/{ctx['N']} · {COPYRIGHT}"),
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
        fig.text(.261,y-.087,mt(f"K/V layers 0–{cfg['CUT']}  {(wr if n_==0 else body)} rows"),fontsize=7.3,family=MONO,color=INK2,va="center")
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
    fig.text(.515,.205,mt(f"{UP}: computed only for new rows,\nattending to the existing memory · layers 0–{cfg['CUT']}: keys re-phased (RoPE)"),fontsize=7.4,color=INK2,va="top")
    block(fig,.765,.43,.205,.36,"QUERY","Question over the memory",f"{L['question']}\n\nmodel input: {L['query_tokens']} template tokens only\ninstalled memory: {T} numerical rows\n(prefix + C1 … C5)\nsource facts given to the model: none\nweights: frozen",mono=False,bfs=8.6)
    arrow(fig,.742,.6,.765,.6)
    frame(fig,.765,.205,.205,.2,ec=BLUE if L["correct"]else ORANGE,fc=BLUE_L if L["correct"]else ORANGE_L,lw=1.3)
    fig.text(.775,.385,"ANSWER (generated by the frozen model)",fontsize=7.4,weight="bold",color=INK2,va="center")
    fig.text(.775,.33,mt(short(a1["answer"],22)),fontsize=15,weight="bold",color=INK,va="center")
    fig.text(.775,.272,mt(f"expected {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'}"),fontsize=8.6,color=BLUE if L["correct"]else ORANGE,weight="bold",va="center")
    fig.text(.775,.235,"expected answer = audit metadata",fontsize=7,color=MUTED,va="center")
    footer(fig,ctx,k,f"Figure {k}. The executed memory path of {MODEL_FAMILY} ({AR[0]} layers, hidden {AR[1]}) for case {L['case']:02d}. (a) Each source fact is read once. (b) The frozen model writes an independent numerical cartridge per fact: the layer-{cfg['CUT']} residual H{cfg['CUT']} and the layers 0–{cfg['CUT']} K/V. "
           f"(c) Cartridges are appended in order; for every append, {UP} are computed only for the new rows over the existing memory, so earlier rows are never recomputed (bitwise check, Fig. 2). "
           f"(d) The question template ({L['query_tokens']} tokens) is processed over the {T}-token numerical memory; no source text is given to the model. Weights are frozen throughout.")
    return finish(fig,path,ctx,[f"{T}",f"{L['query_tokens']} template tokens",mt(short(a1["answer"],22)),f"expected {L['expected']}",f"× {AR[1]}",UP],"Write, append, ask")
# ------------------------------------------------------------------ FIG 2 · append-only growth (measured)
def f2(ctx,k,path):
    L=L_(ctx);st=L["steps"];cars=L["cartridges"];T=L["memory"]["T"];fig=new_fig()
    header(fig,k,"Append-only consolidation, measured forward by forward","Every model forward during load_case() and a bitwise comparison of all previously consolidated K/V rows after each append.","LIVE MEASUREMENT",BLUE)
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
           f"After every append all earlier K/V rows of all {L['memory']['layers']} layers are compared bitwise with the memory before the append.")
    return finish(fig,path,ctx,[f"{T}",f"{L['source_reads']}",L["memory_sha"][:20],"✓ identical"],"Append-only consolidation")
# ------------------------------------------------------------------ FIG 3 · consolidated memory map
def f3(ctx,k,path):
    L=L_(ctx);M=L["memory"];T=M["T"];seg=L["segments"];cfg=ctx["P"]["engine"]["config"];fig=new_fig()
    header(fig,k,"The consolidated numerical memory",f"{MODEL_FAMILY} · {M['layers']} layers × {T} token rows · {M['kv_heads']} KV heads × {M['head_dim']} · {M['bytes']/2**20:.2f} MiB ({M['dtype']}) · values read from the memory of this run.","LIVE MEASUREMENT",BLUE)
    for n_,(key,ttl)in enumerate((("k_norm","‖K‖ per row (mean over KV heads)"),("v_norm","‖V‖ per row (mean over KV heads)"))):
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
    footer(fig,ctx,k,f"Figure {k}. Map of the consolidated memory after the fifth append. Each column is one token row, each row of the heat map one layer; colour shows the row's mean K (top) or V (bottom) norm, z-scored within each layer, for display only. "
           f"Vertical lines separate the instruction prefix and the five cartridges (T marks the target). Layers 0–{cfg['CUT']} come from the cartridges (keys re-phased to their memory positions); {upper_txt(cfg['CUT'],M['layers'])} were computed by the appends. The memory holds numbers only; no text is stored in it.")
    return finish(fig,path,ctx,[f"{T} token rows",f"{M['bytes']/2**20:.2f} MiB",f"{M['layers']} layers",f"L{M['layers']-1}"],"Consolidated memory")
# ------------------------------------------------------------------ FIG 4 · source-free question answering
def f4(ctx,k,path):
    L=L_(ctx);a1=L["answer"];T=L["memory"]["T"];qt=L["query_token_text"];fig=new_fig();ok=L["correct"]
    header(fig,k,"Answering from numerical memory without the source text",f"{MODEL_FAMILY} · the exact model input at query time, the installed memory and the generated answer.","LIVE MEASUREMENT",BLUE)
    frame(fig,.03,.645,.94,.19,ec=RULE,fc=PANEL,lw=.7)
    fig.text(.04,.815,mt(f"MODEL INPUT AT QUERY TIME · {L['query_tokens']} token ids (question template) · over {T} memory rows · source text: none"),fontsize=8.6,weight="bold",color=INK2,va="center")
    toks=[t.replace("\n","⏎")for t in qt];x=.04;y=.775;fig.canvas.draw();rdr=fig.canvas.get_renderer();FW=fig.get_figwidth()*fig.dpi
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
    footer(fig,ctx,k,f"Figure {k}. Source-free question answering for case {L['case']:02d}. Top: the complete list of token ids given to the model at query time, decoded for display — the QUESTION/ANSWER template only; none of the five source facts is supplied. "
           f"The model attends to the {T}-row numerical memory installed as its cache and generates the answer greedily. Repeating the question gives the same answer, and the memory's SHA-256 is unchanged by querying.")
    return finish(fig,path,ctx,[f"{L['query_tokens']} token ids",mt(short(a1["answer"],30)),f"expected {L['expected']}"],"Source-free answering")
# ------------------------------------------------------------------ FIG 5 · sealed TEST575 reference + live
def f5(ctx,k,path):
    L=L_(ctx);fig=new_fig();A=TEST575["arms"];cf=TEST575["counterfactual"]
    header(fig,k,"Sealed TEST575 reference and the live result",f"Left: archived TEST575 values of {MODEL_FAMILY} (24 cases × 3 positions; control arms run in TEST575, not here). Right: live results of this session.","SEALED REFERENCE + LIVE",INK2)
    ax=fig.add_axes([.06,.375,.5,.43]);ax.set_zorder(3);clean(ax);arms=list(ARM_ORDER);cols={"FIRST":BLUE,"MIDDLE":TEAL,"LAST":"#7C3AED"};w=.25
    for j,p in enumerate(POSITIONS):
        v=[A[a_][p]for a_ in arms];ax.bar(np.arange(len(arms))+(j-1)*w,v,w*.92,color=cols[p],label=p)
        for i_,vv in enumerate(v):ax.text(i_+(j-1)*w,vv+.4,str(vv),ha="center",fontsize=7,color=INK)
    ax.set_xticks(range(len(arms)));ax.set_xticklabels([f"{a_}\n{A[a_]['TOTAL']}/72"for a_ in arms],fontsize=7.8);ax.set_ylim(0,28);ax.set_ylabel("correct of 24 cases");ax.axhline(24,color=MUTED,lw=.6,ls=(0,(3,3)))
    ax.legend(loc="upper right",ncol=3);ax.set_title(f"TEST575 · sealed · {MODEL_FAMILY} · target position FIRST / MIDDLE / LAST",fontsize=8.6)
    fig.text(.06,.287,mt(f"counterfactual (same H3, same layers 0–3 K/V, cross-cartridge attention blocked in {upper_txt()}):\nearly K/V identical {cf['early_kv_identical']}/72 · upper K/V changed {cf['upper_kv_changed']}/72 · answer changed {cf['answer_changed']}/72"),
             fontsize=7.4,color=INK,va="center",weight="bold",linespacing=1.35)
    fit_text(fig,.06,.145,.5,.10," · ".join(f"{a_} = {ARM_DESC[a_]}"for a_ in arms)+f" · answers identical to JOINT: INCR_DC3 {TEST575['answer_identity_joint']['INCR_DC3']}/72, BATCH_DC3 {TEST575['answer_identity_joint']['BATCH_DC3']}/72 · gates {TEST575['gates'][0]}/{TEST575['gates'][1]}",fs_max=7.4,fs_min=6.2,color=INK2)
    tx=.6;frame(fig,tx,.47,.37,.36,ec=BLUE,fc=BLUE_L,lw=1.1)
    fig.text(tx+.01,.81,"LIVE · this run",fontsize=9.6,weight="bold",color=BLUE,va="center")
    fig.text(tx+.01,.778,mt(f"case {L['case']:02d} · target {L['position']} · answer {short(L['answer']['answer'],18)}"),fontsize=8.3,family=MONO,color=INK,va="center")
    fig.text(tx+.01,.750,mt(f"expected {L['expected']} · {'CORRECT'if L['correct']else'INCORRECT'}"),fontsize=8.3,family=MONO,color=BLUE if L["correct"]else ORANGE,weight="bold",va="center")
    fig.text(tx+.01,.722,mt(f"sealed TEST575 answer {short(L['sealed_answer'],16)} · identical: {L['matches_sealed']}"),fontsize=7.9,family=MONO,color=INK2,va="center")
    ses=L.get("session")or[];per={p:[r for r in ses if r["position"]==p]for p in POSITIONS}
    fig.text(tx+.01,.688,"LIVE · sealed runs in this session (including this one)",fontsize=8.2,weight="bold",color=INK2,va="center")
    for j,p in enumerate(POSITIONS):
        rr=per[p];fig.text(tx+.01,.660-j*.027,mt(f"{p:<7} {sum(int(r['correct'])for r in rr)}/{len(rr)} correct"),fontsize=8.2,family=MONO,color=INK,va="center")
    fit_text(fig,tx+.01,.478,.35,.09,"A single live run is a demonstration observation. The optional live replay of all 72 panel runs is reported in its own file. Neither replaces or recomputes the sealed TEST575 record.",fs_max=7.6,fs_min=6.3,color=INK2)
    frame(fig,tx,.15,.37,.29,ec=AMBER,fc=AMBER_L,lw=.9)
    fig.text(tx+.01,.42,"SCALE PROBE · TEST572 · same CUT3 mechanism · sealed",fontsize=8.2,weight="bold",color=AMBER,va="center")
    bx=fig.add_axes([tx+.05,.225,.30,.135]);bx.set_zorder(3);ns=[n for n,_,_ in SCALE["rows"]]
    bx.plot(range(len(ns)),[c/m for _,c,m in SCALE["rows"]],marker="o",color=AMBER,lw=1.3,label="direct recall")
    bx.plot(range(len(ns)),[c/m for _,c,m in SCALE["hop2"]],marker="s",color=GRAYPT,lw=1.0,ls=(0,(3,2)),label="two-hop")
    clean(bx);bx.set_xticks(range(len(ns)));bx.set_xticklabels([f"{n}\n{c}/{m}"for n,c,m in SCALE["rows"]],fontsize=6.8);bx.set_ylim(0,1.08);bx.set_ylabel("fraction correct",fontsize=7);bx.tick_params(labelsize=6.8)
    bx.set_xlabel("cartridges in memory · separate fact panel (labels: direct recall)",fontsize=6.8);bx.legend(loc="lower right",bbox_to_anchor=(1.0,1.0),ncol=2,fontsize=6.6)
    footer(fig,ctx,k,f"Figure {k}. Left: the sealed TEST575 record of {MODEL_FAMILY} (result SHA-256 {TEST575['result_sha'][:16]}…): append-only incremental DC3 72/72, batch DC3 72/72, joint text 72/72, native independent K/V 26/72 and the counterfactual "
           f"BATCH_BLOCK3 28/72 on the locked 24-case panel at three target positions; these comparative arms were executed in TEST575, while this demo runs INCR_DC3 live. Right: live results of this session, kept separate, and the sealed TEST572 scale probe "
           "of the same CUT3 mechanism on a separate fact panel (direct recall 6/8 with 40 cartridges; two-hop recall falls to 0/8).")
    return finish(fig,path,ctx,["INCR_DC3\n72/72","NATIVE_INDEP\n26/72","BATCH_BLOCK3\n28/72","LIVE · this run",f"expected {L['expected']}","6/8",f"answer changed {cf['answer_changed']}/72"],"TEST575 reference and live result")
# ------------------------------------------------------------------ FIG 6 · provenance, cross-model reference and priority record
def f6(ctx,k,path):
    P=ctx["P"];L=P["live"];cfg=P["engine"]["config"];E_=P["environment"];fz=P["frozen"];fig=new_fig();AR=cfg["arch"]
    header(fig,k,"Provenance, verification and priority record",f"{MODEL_FAMILY} · configuration, measured integrity quantities, hashes, authorship, the cross-model reference and the claim boundary of this run.","PROVENANCE · PRIORITY RECORD",INK2)
    cols=[(.03,"CONFIGURATION AND ENGINE",
           f"run           {P['run_id']}\nstart (UTC)   {P['run_start_utc']}\nmodel         {cfg['MODEL_ID'].split('/')[-1]}\nweights       frozen · {P['model']['dtype']} · {P['model']['attn']}\n"
           f"architecture  {arch_short(tuple(AR))}\ncartridge     H{cfg['CUT']} residual + K/V L0–{cfg['CUT']}\nconsolidation append-only · {upper_txt(cfg['CUT'],AR[0])}\nprefix        {cfg['prefix_tokens']} tokens\n"
           f"panel         locked · SHA-256\n  {cfg['panel_sha'][:32]}\n  {cfg['panel_sha'][32:]}\nGPU           {E_['gpu']}\ntorch         {E_['torch']}\ntransformers  {E_['transformers']}\ngradio        {E_['gradio']}\n"
           f"engine file   {P['engine']['file']} (verified section)\ngit blob = TEST575 base blob\n  {P['engine']['blob']}\nSHA-256\n  {P['engine']['sha256'][:32]}\n  {P['engine']['sha256'][32:]}\nend (UTC)     {P['run_end_utc']}"),
          (.35,"VERIFICATION MEASURED IN THIS RUN",
           f"full-weight SHA-256 (start-up) = TEST575\n  {fz['weight_sha_startup'][:40]}…\nweight sentinel  startup = before = after\n  {fz['sentinel_after'][:40]}…\nall-parameter guard  startup = before = after\n  {fz['guard_after'][:40]}…\n"
           f"trainable tensors   {fz['trainable_tensors']}\nLoRA / optimizer    {fz['lora']} / {fz['optimizer']}\nforeign hooks       {fz['foreign_hooks_before']} → {fz['foreign_hooks_after']}\nresidual hooks      {len(fz['residual_hooks_before'])} → {len(fz['residual_hooks_after'])}\n"
           f"source token-id forwards (load)  {L['source_reads']}\nzero-embedding consolidations    {L['zero_embed_forwards']}\nprior rows unchanged (4 appends) "
           f"{all(s['prior_rows_bitwise_unchanged']for s in L['steps'][1:])}\nquery input = template only      True\nmemory unchanged by querying     {L['memory_unchanged_by_query']}\n"
           f"repeat answer identical          {L['repeat_identical']}\n\nmemory  {L['memory']['T']} rows · {L['memory']['bytes']/2**20:.2f} MiB\n  SHA {L['memory_sha'][:36]}…\nanswer  {short(L['answer']['answer'],24)} · {'correct'if L['correct']else'incorrect'}\n"
           f"sealed TEST575 answer identical  {L['matches_sealed']}\npre-seal checks passed: {len(P['checks_pre_seal'])}\nverdict\n  {P['verdict'][:44]}\n  {P['verdict'][44:]}"),
          (.67,"AUTHORSHIP · PRIORITY · CLAIM BOUNDARY",
           f"{DISCOVERY}\n{ORG} · {DATE_TXT}\n{AUTHOR_PLACE}\n\n{PRIORITY}\n\n{LICENSE_NAME}\n{COPYRIGHT}\n{LICENSE_URL}\n\nsealed TEST575 ({MODEL_FAMILY})\n  result {TEST575['result_sha'][:30]}…\n  INCR_DC3 72/72 · NATIVE_INDEP 26/72\n\n"
           "not established:\n"+"\n".join("  • "+t for t in("large populations (TEST572: direct recall 6/8 at 40)","two-hop inference (TEST572: 0/8 at 40)","KV-only cartridges (H3 is part of it)","scored free-text questions","models beyond the two tested families")))]
    for x,ttl,body in cols:
        frame(fig,x,.30,.30,.57,ec=RULE,fc="white",lw=.8);fig.text(x+.01,.85,ttl,fontsize=9,weight="bold",color=INK2,va="center")
        fit_text(fig,x+.01,.31,.282,.52,body,fs_max=9.0,fs_min=5.9,family=MONO if x<.6 else None,color=INK)
    frame(fig,.03,.148,.94,.137,ec=BLUE,fc=BLUE_L,lw=1.2)
    fig.text(.042,.266,mt(CROSS_TITLE),fontsize=10.5,weight="bold",color=BLUE,va="center")
    fit_text(fig,.042,.156,.916,.094,f"{CROSS_TEXT}\nEarlier record (reference only): {PRIOR_PUB['title']} · DOI {PRIOR_PUB['doi']} · {PRIOR_PUB['url']}\n{CROSS_BOUNDARY}",fs_max=8.4,fs_min=6.3,color=INK)
    footer(fig,ctx,k,f"Figure {k}. Provenance record of run {P['run_id']} (verdict {P['verdict']}). Payload SHA-256 {ctx['sha']}. "
           "Hashes are artifact-integrity seals, not scientific proof or third-party verification. All live values come from the raw records stored in the sealed payload; sealed values are reproduced verbatim from the TEST575 and TEST572 records. "
           "The earlier record is cited by DOI only; none of its values is shown as a result here.")
    return finish(fig,path,ctx,[P["run_id"],P["engine"]["blob"],fz["weight_sha_startup"][:40],fz["guard_after"][:40],DISCOVERY,ORG,CROSS_TITLE,PRIOR_PUB["doi"],PRIOR_PUB["url"]],"Provenance and priority record")
FIGURES=[("fig_01_write_append_ask","Fig. 1 · Write → append-only consolidation → source-free answer",f1),("fig_02_append_only_record","Fig. 2 · Append-only consolidation, measured",f2),
         ("fig_03_memory_map","Fig. 3 · The consolidated numerical memory",f3),("fig_04_source_free_answer","Fig. 4 · Answering without the source text",f4),
         ("fig_05_reference_and_live","Fig. 5 · Sealed TEST575 reference and live result",f5),("fig_06_provenance_priority","Fig. 6 · Provenance, cross-model reference and priority record",f6)]
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
MAM_QWEN_PART2_OK=True
# =====================================================================================================================================
# AKBASCORE MAM · PERSISTENT NUMERICAL MEMORY FOR FROZEN LANGUAGE MODELS — QWEN2.5-7B · PART 3 / 3 · SELF-TEST AND GRADIO DEMONSTRATION
# Same demo as PARTS 1 and 2. Run this cell after PART 2 / 3, in the same Colab runtime. It runs a GPU-free self-test of the full
# pipeline (including fail-closed cases) and then prints a public gradio.live link. Choose a case and a target position, press RUN.
# Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 9 October 2026 · AKBASCORE RESEARCH SOFTWARE LICENSE
# =====================================================================================================================================
if not globals().get("MAM_QWEN_PART2_OK"):raise RuntimeError("PART 2 / 3 has not been run in this runtime. Run PART 1, then PART 2, then PART 3.")
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
            weight_sha0=("d"*64 if mode=="weight_ref" else WEIGHT_SHA_EXPECTED),sentinel0="a"*64,guard0="b"*64,hooks0=0,
            sentinel_method="synthetic",guard_method="synthetic",full_weight_method="synthetic",forward_counter="synthetic",observer="synthetic")
    def _target(s,w):return w*4+{"CURRENT":0,"NEAR":1,"FORMER":2,"ROLE":3}[["CURRENT","FORMER","NEAR","ROLE"][w%4]]
    def config(s):
        return dict(MODEL_ID=MODEL_ID,CUT=6 if s.mode=="config"else CUT_EXPECTED,ENGINE_CUTS=[1,2,3,6],MAX_NEW=16,SEED=552552,PANEL_SEED=550550,EXPECTED_PANEL_SHA=EXPECTED_PANEL_SHA,panel_sha=EXPECTED_PANEL_SHA,prefix_tokens=s.P,
                    n_cartridges=96,n_cases=24,positions=list(POSITIONS),system="synthetic",arch=list(ARCH),engine_sha256=ENGINE_SHA256_EXPECTED,engine_blob=ENGINE_BLOB_EXPECTED,engine_exec_sha256="e"*64)
    def status(s):return{"title":"AKBASCORE MAM","reference_test":"TEST575","reference_score":"72/72","panel_sha":EXPECTED_PANEL_SHA,"cut":CUT_EXPECTED}
    def cases(s):return[{"case":w,"type":s.cars[s._target(w)]["type"],"question":s.questions[w],"positions":list(POSITIONS)}for w in range(24)]
    def keys(s,case,position):
        t=s._target(case);d=[((case+1+3*j)%24)*4+(j%4)for j in range(4)]
        return [t]+d if position=="FIRST"else d[:2]+[t]+d[2:]if position=="MIDDLE"else d+[t]
    def car(s,ci):return dict(s.cars[ci])
    def target(s,case):return s._target(case)
    def source_ids(s,ci):return list(range(1,s.P+1))+[1000+ci*20+j for j in range(s.cars[ci]["body_len"])]
    def _qtoks(s,q):return re.findall(r"\n|[^\s\n]+",f"\nQUESTION:\n{q}\n\nANSWER:\n<|im_end|>\n<|im_start|>assistant\n")
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
    def full_weight_sha(s):return s.info["weight_sha0"]
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
            st.update(memory_sha=hashlib.sha256(f"m{new}".encode()).hexdigest(),seconds=0.3,forwards=f);steps.append(st);allf+=f;old=new
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
        return dict(question=qq,answer=ans,reference_question=isref,expected=exp if isref else None,correct=(hit_ans(ans,exp)if isref else None),elapsed_seconds=0.25,
                    source_text_supplied_to_query=False,model_weights_changed=False),rec,0.25
    def memory_len(s):return s.T
    def memory_sha_now(s):return hashlib.sha256(f"m{s.T}".encode()).hexdigest()
    def memory_profile(s):
        NL,_,_,KVH,HD=ARCH;rng=np.random.default_rng(s.T);k=(rng.normal(20,3,(NL,s.T))+np.linspace(0,10,NL)[:,None]).tolist();v=(rng.normal(2,.4,(NL,s.T))).tolist()
        return dict(layers=NL,T=s.T,k_norm=k,v_norm=v,bytes=s.T*NL*2*KVH*HD*2,kv_heads=KVH,head_dim=HD,dtype="bfloat16")
    def replay(s,progress=None):
        rows=[]
        for c in range(24):
            for p in POSITIONS:
                rows.append(dict(case=c,position=p,answer=s.cars[s._target(c)]["gold"],expected=s.cars[s._target(c)]["gold"],correct=True))
                if progress:progress(len(rows),72,rows[-1])
        return dict(score=72,total=72,matches_TEST575=True,panel_sha=EXPECTED_PANEL_SHA,rows=rows,elapsed_seconds=0.1)
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
          f'<span class="mono">order</span> C1 → C5 are written and appended in this order; the source text is not given to the model when the question is asked.</div>')
    return card_html(f"Case {c:02d} · target at {pos}",body,"info")
def engine_card_html(I):
    inf=I.info;cfg=I.config();A=TEST575["arms"]
    rows=[("model",f"{inf['model_id']} · frozen · {inf['dtype']} · {arch_txt(tuple(cfg['arch']))}"),("GPU",inf["gpu"]),("cartridge",f"H{cfg['CUT']} residual + K/V of layers 0–{cfg['CUT']} · written once per fact"),
          ("consolidation",f"append-only · {upper_txt(cfg['CUT'],cfg['arch'][0])} computed only for new rows · prior rows never recomputed"),("panel",f"locked 24-case panel · FIRST/MIDDLE/LAST · SHA {cfg['panel_sha'][:16]}…"),
          ("engine",f"{inf['engine_file']} (verified section, as in TEST575) · git blob {inf['engine_blob'][:16]}… · SHA-256 {inf['engine_sha256'][:16]}…"),
          ("weights",f"full-weight SHA-256 {inf['weight_sha0'][:16]}… (= TEST575 reference) · sentinel {inf['sentinel0'][:16]}… · all-parameter guard {inf['guard0'][:16]}…"),
          ("init",f"{inf['init_seconds']:.1f} s" if isinstance(inf['init_seconds'],float)and math.isfinite(inf['init_seconds'])else"reused")]
    body="".join(f'<span class="mono">{html.escape(a)}</span> {html.escape(str(b))}<br>'for a,b in rows)
    cf=TEST575["counterfactual"]
    body+=(f'<div class="small" style="margin-top:8px"><b>SEALED TEST575 REFERENCE · {html.escape(MODEL_FAMILY)}</b> (archived comparative controls · 72 = 24 cases × 3 target positions · not recomputed by single runs; this demo runs INCR_DC3 live)<br>'
           +"<br>".join(f'<span class="mono">{a_}</span> {A[a_]["TOTAL"]}/72 · FIRST {A[a_]["FIRST"]} · MIDDLE {A[a_]["MIDDLE"]} · LAST {A[a_]["LAST"]} — {html.escape(ARM_DESC[a_])}'for a_ in ARM_ORDER)
           +f'<br><span class="mono">counterfactual</span> upper-layer K/V changed {cf["upper_kv_changed"]}/72 · answer changed {cf["answer_changed"]}/72 · answers identical to JOINT (INCR/BATCH) 72/72 · gates {TEST575["gates"][0]}/{TEST575["gates"][1]}'
           +f'<br><span class="mono">scale probe {SCALE["name"]}</span> direct recall '+" · ".join(f"{n}: {c_}/{m}"for n,c_,m in SCALE["rows"])+' cartridges (separate fact panel)'
           +f'<br><span class="mono">lock SHA-256</span> {TEST575["lock"]}<br><span class="mono">result SHA-256</span> {TEST575["result_sha"]}</div>')
    return card_html("Engine status · sealed reference",body,"on")
READY_HTML=card_html("Ready",f"Choose a case and the target position, optionally type your own question, and press <b>RUN</b>. One run writes five independent cartridges, appends them one by one, "
    f"asks the question without the source text, verifies the frozen weights, seals the payload and renders {N_FIGS} figures (JPEG 300 dpi + vector PDF). "
    "Each run is a LIVE demonstration observation; the TEST575 values above are the sealed reference.","info")
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
              f'<span class="mono">sealed</span> TEST575 INCR_DC3 answer for this case and position: {html.escape(L["sealed_answer"])} · live identical: {L["matches_sealed"]}<br>'
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
    try:
        rid=f"{REPLAY_PREFIX}{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}";prune_runs(2,REPLAY_PREFIX);d=ROOT/rid;d.mkdir(parents=True,exist_ok=True)
        st={"done":0,"last":None,"res":None,"err":None}
        def prog(n,tot,row):st["done"]=n;st["last"]=row
        def work():
            try:st["res"]=replay_report(INSTR,rid,d,prog)
            except Exception as ex:st["err"]=ex;traceback.print_exc()
        th=threading.Thread(target=work,daemon=True);th.start()
        while th.is_alive():
            lr=st["last"];tail=(f"<br>last: case {lr['case']:02d} {lr['position']} → {html.escape(str(lr['answer']))} ({'PASS' if lr['correct'] else 'FAIL'})"if lr else"")
            yield card_html("Live replay running",f"Replaying the 72 locked panel runs (INCR_DC3) with the unchanged engine (LIVE; separate from the sealed TEST575 record).{tail}"+pbar_html(st["done"],72),"info"),SKIP(),SKIP()
            time.sleep(1.5)
        if st["err"]is not None:raise st["err"]
        R=st["res"];o=R["out"];pp=o["per_position"]
        body=(f"<b>LIVE REPLAY {o['score']}/72</b> · FIRST {pp['FIRST']}/24 · MIDDLE {pp['MIDDLE']}/24 · LAST {pp['LAST']}/24 · {o['seconds']} s · weights unchanged: {o['weights_unchanged']}<br>"
              f"Live answers identical to the sealed TEST575 INCR_DC3 answers: {o['answers_identical_to_sealed_TEST575']}/72. Sealed TEST575 incremental DC3 for comparison: {TEST575['arms']['INCR_DC3']['TOTAL']}/72. The replay is a separate live measurement and does not replace the sealed record.<br>"
              f'<span class="mono">replay SHA-256 {R["sha"]}</span>')
        yield card_html("Live replay complete",body,"on"),dl_update(R["json"],REPLAY_LABELS[0]),dl_update(R["txt"],REPLAY_LABELS[1])
    except Exception as ex:
        yield card_html("Live replay failed",f"{html.escape(type(ex).__name__)}: {html.escape(str(ex)[:500])}","err"),dl_update(None,REPLAY_LABELS[0]),dl_update(None,REPLAY_LABELS[1])
    finally:
        GPU_LOCK.release()
        try:torch.cuda.empty_cache()
        except Exception:pass
def verify_handler():
    if not GPU_LOCK.acquire(blocking=False):return"Busy: a run is in progress."
    try:
        t=time.perf_counter();g=INSTR.guard();fz=INSTR.frozen();w=INSTR.full_weight_sha();cfg=INSTR.config()
        out=dict(model=MODEL_ID,engine_status=INSTR.status(),full_weight_sha256_now=w,full_weight_sha256_TEST575=WEIGHT_SHA_EXPECTED,full_weight_identical_to_TEST575=w==WEIGHT_SHA_EXPECTED,
                 all_parameter_guard_now=g,all_parameter_guard_startup=INSTR.info["guard0"],guard_unchanged=g==INSTR.info["guard0"],
                 sentinel_unchanged=fz["sentinel"]==INSTR.info["sentinel0"],trainable_tensors=fz["trainable_tensors"],foreign_hooks=fz["foreign_hooks"],residual_hooks=fz["residual_hooks"],
                 engine_file=ENGINE_FILE,engine_blob=cfg["engine_blob"],engine_blob_TEST575=ENGINE_BLOB_EXPECTED,engine_sha256=cfg["engine_sha256"],engine_sha256_expected=ENGINE_SHA256_EXPECTED,
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
                         ("source_in_query","source tokens in the query input"),("config","a changed consolidation cut"),("weight_ref","a full-weight SHA-256 different from TEST575"),
                         ("residual_hook","a hook left on a decoder layer after the run")):
            d2=d/mode;d2.mkdir()
            try:drain(execute_run(SelfTestInstrument(mode),dict(run_id="SELFTEST-"+mode.upper(),run_dir=d2,case=5,position="LAST",custom="",session=[])))
            except AuditFail:out.append(f"fail-closed: {what} aborts the run (nothing sealed)")
            else:raise RuntimeError(f"self-test: {what} did not abort the run")
        d3=d/"replay";d3.mkdir();R=replay_report(SelfTestInstrument("ok"),"SELFTEST-REPLAY",d3);assert R["out"]["score"]==72 and file_ready(R["json"])and file_ready(R["txt"])
        out.append("live replay report (synthetic): JSON + TXT, score re-derived from rows")
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
      f'<a href="{SOURCE_URL}" target="_blank" rel="noopener">566.py engine</a> · <a href="{LOCK_URL}" target="_blank" rel="noopener">TEST575 program</a> · <a href="{LOG_URL}" target="_blank" rel="noopener">TEST575 log</a> · '
      f'<a href="{SCALE_LOG_URL}" target="_blank" rel="noopener">TEST572 log</a></div></div>')
CROSS_HTML=card_html(CROSS_TITLE,f'{html.escape(CROSS_TEXT)}<br><br><span class="mono">earlier record</span> {html.escape(PRIOR_PUB["title"])}<br>'
    f'<span class="mono">DOI</span> <a href="{PRIOR_PUB["url"]}" target="_blank" rel="noopener">{html.escape(PRIOR_PUB["doi"])}</a> · {html.escape(PRIOR_PUB["date"])}<br>'
    f'<div class="small" style="opacity:.8;margin-top:6px">{html.escape(CROSS_BOUNDARY)}</div>',"on")
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
_TITLE="AKBASCORE MAM · Qwen2.5-7B · persistent numerical memory · Mustafa Akbaş"
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
    with gr.Accordion("Live replay of the locked panel · 72 INCR_DC3 runs · a few minutes on an A100 (reported separately from the sealed TEST575 record)",open=False):
        rp_btn=gr.Button("Run the live replay (72 runs)")
        rp_status=gr.HTML(card_html("Live replay","Replays all 24 cases × FIRST/MIDDLE/LAST with the unchanged engine (INCR_DC3), compares every answer with the sealed TEST575 answer and writes its own JSON/TXT report. It does not replace the sealed TEST575 result.","info"))
        with gr.Row():rp_dl=[gr.DownloadButton(label=l,value=None,interactive=False)for l in REPLAY_LABELS]
    with gr.Accordion("Verification · full-weight SHA-256 · weight guard · engine and panel hashes",open=False):
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

