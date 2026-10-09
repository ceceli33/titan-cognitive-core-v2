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
