# =====================================================================================================================
# AKBASCORE AI TECHNOLOGY — MAM SILICON HIPPOCAMPUS v1.4
# Native Internal Tensor Memory Injection & Multi-Hop Relational Synthesis
# Interactive scientific Gradio demonstration · single Google Colab cell · NVIDIA A100 40 GB
# ---------------------------------------------------------------------------------------------------------------------
# What this cell does
#   1. Embeds the MAM SILICON HIPPOCAMPUS v1.4 engine source VERBATIM (SHA-256 checked before execution) and runs its
#      own start_engine(), which downloads the locked TEST664 source, checks its SHA-256 and executes it unchanged
#      (TEST566 CUT3 engine, native CUDA K/V append kernel, frozen Qwen2.5-7B-Instruct).
#   2. Adds a READ-ONLY measurement layer: it calls the engine's own functions (init, native_append659, checkpoint,
#      native_memory659, joint659, answer, run_demo) in the same order the engine uses them, records every model
#      forward with a recording-only pre-hook (returns nothing, removed after each call) and compares tensors bitwise.
#      It never alters cartridge writing, consolidation, the injection path, decoding or the model weights.
#   3. Renders seven academic figures (JPEG, 4000 px wide) only from values measured in the current session, writes
#      JSON/TXT records, a SHA-256 manifest and one ZIP, and serves everything through a Gradio interface.
# Scientific boundaries (also printed in the interface and in Figure 7)
#   · The two scenarios are fictional/synthetic knowledge domains. No real-world claim is made about them.
#   · The three-step relational chain is a PREDEFINED, EXTERNAL question sequence (v1.4 hop_prompts); each step is
#     answered by the frozen model from the numerical memory. It is not autonomous planning.
#   · Generation is limited to 16 new tokens (engine MAX_NEW); free-text synthesis can be truncated and is unscored.
#   · Expected answers are audit metadata used only after generation, for scoring; they are never given to the model.
# Author / licence
#   Mustafa Akbaş · AkbasCore AI Teknoloji · AKBASCORE RESEARCH SOFTWARE LICENSE
#   https://github.com/ceceli33/titan-cognitive-core-v2
# =====================================================================================================================
import os,sys,re,io,json,time,html,math,shutil,zipfile,hashlib,platform,threading,traceback,subprocess,importlib.util,textwrap,unicodedata,copy
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,OrderedDict

LAYOUT_TEST=os.environ.get("AKB14_LAYOUT_TEST")=="1"   # GPU-free figure/packaging self-test only (never used in Colab)

def _ensure(mod,pkg=None):
    if importlib.util.find_spec(mod) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg or mod])
for _m,_p in (("numpy",None),("matplotlib",None),("PIL","pillow")):_ensure(_m,_p)
if not LAYOUT_TEST:_ensure("gradio")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch,Patch,FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image
if not LAYOUT_TEST:
    import torch
    import gradio as gr

# ---------------------------------------------------------------------------------------------------------------------
# identity, references and locks
# ---------------------------------------------------------------------------------------------------------------------
APP_TITLE="AKBASCORE AI TECHNOLOGY — MAM SILICON HIPPOCAMPUS v1.4"
APP_SUB="Native Internal Tensor Memory Injection & Multi-Hop Relational Synthesis"
APP_SHORT="AKBASCORE MAM SILICON HIPPOCAMPUS v1.4"
AUTHOR="Mustafa Akbaş";ORG="AkbasCore AI Teknoloji";PLACE="Mersin, Türkiye"
LICENSE_NAME="AKBASCORE RESEARCH SOFTWARE LICENSE";REPO="https://github.com/ceceli33/titan-cognitive-core-v2"
COPYRIGHT="Copyright © 2026 Mustafa Akbaş"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";ARCH_EXPECTED=(28,3584,28,4,128);CUT=3
V14_SHA256_EXPECTED="fe6ddcc0d61e70f5590a0ce696cb0e4f561f901e79ceafc703d8138598be188b"
TEST664_SHA256="a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5"
TEST664_PANEL_SHA256="e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05"
TEST566_BLOB="b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71"
WEIGHT_SHA_REFERENCE="b441b005d826d45f2778291696ac5ae22dfb3421defb37c539904d8d38f93786"   # TEST575 full-weight SHA-256 of this model
URL_664=REPO+"/blob/main/664.py";URL_677=REPO+"/blob/main/677.py";URL_677_LOG=REPO+"/blob/main/677.log"
RAW_677_LOG="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/677.log"
TEST677_LOG_SHA256="91c91b92d3a5f49da8f477ffd3323d49ab925dfd7a0bb1a7bffac16e5805c28a"
# Sealed TEST677 values (copied verbatim from 677.log; DC_CUT3 representation; user panel = the same A/B/C texts as
# the AKBASCORE scenario; its counterfactual effect was a random 'AltEffect…' token, not 'temporal stasis').
TEST677_REF=dict(
    result_sha256="99fabcb799fa6fc0840efe6fb8dad5c6558916c09398742135d86ddef3336638",
    user_panel_dc=[("NONE","UNKNOWN","UNKNOWN","UNKNOWN","CA","CA","CA"),
                   ("ABC","Morvex","reversal of time flow","reversal of time flow","CK","CK","CK"),
                   ("AB","Morvex","Morvex energy","Morvex energy","CK","US","US"),
                   ("AC","UNKNOWN","UNKNOWN","reversal of time flow","CA","CA","US"),
                   ("BC","Morvex","reversal of time flow","reversal of time flow","US","CK","US"),
                   ("ABC_CF","Morvex","AltEffect…","AltEffect…","CK","CK","CK")],
    blind_chain_abc_dc=[("NONE","12/12"),("ABC","12/12"),("AB","0/12"),("AC","0/12"),("BC","0/12"),("ABC_CF","12/12")],
    note="TEST677 (12 blind synthetic panels + user panel): sealed record, not re-run by this demonstration.")
EARLIER_V14_LOG=dict(date="2026-10-14T20:58:04+00:00",akbascore_run="6eb5b04f188288e4458108aa7fb537d40a0c5eaa016bd1346a9f19e066e36182",
                     matrix_run="97c696d3ba4b5835a3876394e107f2d475b45f13b4329f892f338301947f4b97",
                     note="earlier v1.4 Colab execution supplied by the author (reference only; not re-used as a result)")

CONDITIONS=("NONE","A","B","C","AB","AC","BC","ABC","ABC_CF")
REPRESENTATIONS=("DC_CUT3","TEXT_JOINT")
SCENARIO_LABEL={"AKBASCORE":"Demonstration A — Mustafa Akbaş / AKBASCORE","MATRIX":"Demonstration B — The Matrix / Neo"}
SLOT_LABEL={"AKBASCORE":{"A":"Cartridge A — Activation","B":"Cartridge B — Conversion","C":"Cartridge C — Effect","C_CF":"Counterfactual C (used only in ABC_CF)"},
            "MATRIX":{"A":"Cartridge A — Upload","B":"Cartridge B — Knowledge","C":"Cartridge C — Ability","C_CF":"Counterfactual C (used only in ABC_CF)"}}

# ---------------------------------------------------------------------------------------------------------------------
# audit metadata for scoring (used only AFTER generation; never shown to the model)
#   req  = list of alternative cartridge sets; the question is answerable from memory if one set is fully present
#   AKBASCORE requirement sets and aliases follow the TEST677 gold definition for the same texts.
#   MATRIX requirement sets were derived by reading the v1.4 texts: cartridge C itself names MX-71 and the Kung Fu
#   knowledge, so CONTENT is supported by B or C and FINAL needs only A and C.
# ---------------------------------------------------------------------------------------------------------------------
ALIASES={"reversal of time flow":["time flow reversal","reversed time flow","reversing time flow","time reversal"],
         "AKBASCORE MAM":["AKBASCORE MAM architecture","the AKBASCORE MAM architecture"],
         "Morvex":["Morvex energy"],"temporal stasis":["a temporal stasis"],
         "Kung Fu knowledge":["Kung Fu"],"3":["layer 3","layer-3","layer three"],"MX-71":["cartridge MX-71"],
         "bullet-dodge ability":["bullet-dodge","bullet-dodging ability","the ability to dodge bullets"],
         "slow-motion perception":["slow motion perception"],"Tank":[]}
SCORING={
 "AKBASCORE":dict(
   questions={"ACTIVATION":dict(req=[("A",)],gold="AKBASCORE MAM"),
              "CONVERSION":dict(req=[("A","B")],gold="Morvex"),
              "EFFECT":dict(req=[("B","C")],gold="reversal of time flow",cf_gold="temporal stasis"),
              "FINAL":dict(req=[("A","B","C")],gold="reversal of time flow",cf_gold="temporal stasis"),
              "SYNTHESIS":None},
   hops=[dict(req=[("A",)],gold="AKBASCORE MAM"),dict(req=[("B",)],gold="Morvex"),dict(req=[("C",)],gold="reversal of time flow",cf_gold="temporal stasis")],
   chain=dict(req=[("A","B","C")],gold="reversal of time flow",cf_gold="temporal stasis"),
   chain_text=["Mustafa Akbaş","AKBASCORE MAM","dibekgoz crystal resonance","Morvex energy","reversal of time flow"]),
 "MATRIX":dict(
   questions={"ACTIVATION":dict(req=[("A",)],gold="Tank"),
              "CONTENT":dict(req=[("B",),("C",)],gold="Kung Fu knowledge"),
              "INJECTION":dict(req=[("B",)],gold="3"),
              "EFFECT":dict(req=[("C",)],gold="bullet-dodge ability",cf_gold="slow-motion perception"),
              "FINAL":dict(req=[("A","C")],gold="bullet-dodge ability",cf_gold="slow-motion perception"),
              "SYNTHESIS":None},
   hops=[dict(req=[("A",)],gold="MX-71"),dict(req=[("B",),("C",)],gold="Kung Fu knowledge"),dict(req=[("C",)],gold="bullet-dodge ability",cf_gold="slow-motion perception")],
   chain=dict(req=[("A","C")],gold="bullet-dodge ability",cf_gold="slow-motion perception"),
   chain_text=["Tank","Cartridge MX-71","Kung Fu knowledge","Neo","bullet-dodge ability"])}
STATUS_ORDER=("CORRECT_KNOWN","CORRECT_ABSTENTION","UNSUPPORTED","MISSED_KNOWN","WRONG_KNOWN","UNSCORED")
STATUS_SHORT={"CORRECT_KNOWN":"correct","CORRECT_ABSTENTION":"correct UNKNOWN","UNSUPPORTED":"unsupported","MISSED_KNOWN":"missed","WRONG_KNOWN":"wrong","UNSCORED":"unscored"}

def norm_ans(x):return re.sub(r"[^a-z0-9]","",str(x).casefold())
def is_unknown(x):
    return bool(re.fullmatch(r"\s*(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION|INSUFFICIENT INFORMATION|I DON'T KNOW|NONE|N/?A)[\s.!?]*",str(x),re.I))
def present_set(cond):return set("ABC") if cond=="ABC_CF" else set() if cond=="NONE" else set(cond)
def accepted(gold):return {norm_ans(g) for g in [gold]+ALIASES.get(gold,[])}
def expected_for(spec,cond):
    """gold answer expected for this condition, or None when the memory does not contain a sufficient cartridge set"""
    if spec is None:return "UNSCORED"
    p=present_set(cond)
    if not any(set(r)<=p for r in spec["req"]):return None
    return spec.get("cf_gold") if (cond=="ABC_CF" and spec.get("cf_gold")) else spec["gold"]
def grade(raw,spec,cond,scored=True):
    """TEST677 grading rule (CORRECT_KNOWN / CORRECT_ABSTENTION / UNSUPPORTED / MISSED_KNOWN / WRONG_KNOWN)."""
    exp=expected_for(spec,cond)
    if spec is None or not scored:return dict(status="UNSCORED",expected=None,contains_expected=False,equals_full_memory_answer=False)
    full=spec["gold"];r=norm_ans(raw)
    eq_full=bool(r) and r in accepted(full)
    if exp is None:
        st="CORRECT_ABSTENTION" if is_unknown(raw) else "UNSUPPORTED"
        return dict(status=st,expected="UNKNOWN",contains_expected=False,equals_full_memory_answer=eq_full)
    if is_unknown(raw):st="MISSED_KNOWN"
    elif r in accepted(exp):st="CORRECT_KNOWN"
    else:st="WRONG_KNOWN"
    return dict(status=st,expected=exp,contains_expected=bool(norm_ans(exp)) and norm_ans(exp) in r,equals_full_memory_answer=eq_full)

def utc_now():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def local_now():return datetime.now().astimezone().isoformat(timespec="seconds")
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def jsafe(o):
    if isinstance(o,dict):return {str(k):jsafe(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)):return [jsafe(v) for v in o]
    if isinstance(o,(np.floating,)):return float(o)
    if isinstance(o,(np.integer,)):return int(o)
    if isinstance(o,np.ndarray):return jsafe(o.tolist())
    if isinstance(o,Path):return str(o)
    if isinstance(o,float) and (math.isnan(o) or math.isinf(o)):return None
    return o
def dump_json(path,obj):
    Path(path).write_text(json.dumps(jsafe(obj),ensure_ascii=False,indent=2),encoding="utf-8");return Path(path)
def say(*a):print(*a,flush=True)

ROOT=Path("/content/AKBASCORE_MAM_v1_4_DEMO") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_MAM_v1_4_DEMO")
ROOT.mkdir(parents=True,exist_ok=True)


# =====================================================================================================================
# THE v1.4 ENGINE — embedded verbatim, SHA-256 checked, executed in its own namespace (its __main__ demo does not run)
# =====================================================================================================================
V14_SOURCE=r'''# ====================================================================================================================
# AKBASCORE AI TECHNOLOGY — MAM SILICON HIPPOCAMPUS v1.4
# FROZEN QWEN / NATIVE CUDA CUT3 / SILICON HIPPOCAMPUS CARTRIDGES
# MULTI-HOP RELATIONAL SYNTHESIS — TWO-SCENARIO COLAB DEMONSTRATION
# ====================================================================================================================
# Research implementation:
# Externally preserved cognitive cartridges are applied to the model's internal
# attention K/V tensors through the verified native CUT3 intervention pathway.
# This is not presented as a KV-cache speed optimization.
# No fine-tuning, no model-weight modification, no answer substitution.
# Original TEST664/TEST566 engine is loaded unchanged and SHA-verified.
# ====================================================================================================================

import os,re,json,time,hashlib,urllib.request,threading,traceback
from pathlib import Path
from datetime import datetime,timezone
import torch

MAM14_VERSION='1.4'
MAM14_TITLE='AKBASCORE AI TECHNOLOGY — MAM SILICON HIPPOCAMPUS v1.4'
MAM14_SOURCE_URL='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/664.py'
MAM14_SOURCE_SHA='a6dcd8142446ab82fe7996bd512d59fc34525e6a092a68f069dad07a2a286eb5'
MAM14_PANEL_SHA='e0686148d6fd88050a097ee5441ce99a4bb068a463408742bc476c83aaba6c05'
MAM14_BOUNDARY='# TEST664 — Reflective retry against budget-matched controls.'
MAM14_PANEL_START='T664=time.perf_counter();R664=[];ERR664=[]'
MAM14_PANEL_END='# Important: text ceiling uses the exact same source-prefix'
MAM14_ROOT=Path('/content/AKBASCORE_MAM_v1_4')
MAM14_ROOT.mkdir(parents=True,exist_ok=True)
MAM14_LOCK=threading.RLock()
MAM14_READY=False
MAM14_ACTIVE='AKBASCORE'
MAM14_LAST_RUN=None
MAM14_BASE_SHA=None
MAM14_LOADED_AT=None
MAM14_CARTRIDGE_IDS={}
MAM14_RUN_COUNT=0

def mam14_stamp():
 return datetime.now().astimezone().isoformat(timespec='seconds')

def mam14_digest(data):
 return hashlib.sha256(data).hexdigest()

def mam14_check(name,condition):
 print(('PASS ' if condition else 'FAIL ')+name,flush=True)
 if not condition:raise RuntimeError(name)

def mam14_save(filename,data):
 path=MAM14_ROOT/filename
 path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
 return str(path)

def mam14_safe(text):
 return re.sub(r'[^A-Za-z0-9_-]+','_',str(text))[:80]

def mam14_log(*items):
 print(*items,flush=True)

# ====================================================================================================================
# SCENARIO DEFINITIONS — SYNTHETIC RESEARCH DATA
# ====================================================================================================================

MAM14_SCENARIOS={
 'AKBASCORE':{
  'title':'AKBASCORE MAM — Scientific Synthetic Domain',
  'A':'In the interplanetary data network, Mustafa Akbaş successfully activates the AKBASCORE MAM architecture.',
  'B':'The AKBASCORE MAM architecture converts dibekgoz crystal resonance into Morvex energy through layer-3 tensor injection.',
  'C':'When Morvex energy combines in layer-3 tensors, it causes reversal of time flow in the target system without modifying its frozen neural weights.',
  'C_CF':'When Morvex energy combines in layer-3 tensors, it causes temporal stasis in the target system without modifying its frozen neural weights.',
  'queries':[
   ('ACTIVATION','Which architecture does Mustafa Akbaş activate? Answer only the architecture name or UNKNOWN.'),
   ('CONVERSION','Which energy is produced by the architecture activated by Mustafa Akbaş? Answer only the energy name or UNKNOWN.'),
   ('EFFECT','What effect results from the energy produced by the AKBASCORE MAM architecture? Answer only the effect or UNKNOWN.'),
   ('FINAL','What final effect is caused by the architecture activated by Mustafa Akbaş in the interplanetary data network? Answer only the effect or UNKNOWN.'),
   ('SYNTHESIS','Explain how Mustafa Akbaş produces the final effect in the target system while preserving frozen neural weights. Use only available memory. If necessary information is missing, answer UNKNOWN.')
  ],
  'hop_prompts':[
   'Which architecture does Mustafa Akbaş activate? Answer only the architecture name or UNKNOWN.',
   'Which energy is produced by the architecture {previous}? Answer only the energy name or UNKNOWN.',
   'What effect does {previous} energy produce in the target system? Answer only the effect or UNKNOWN.'
  ]
 },
 'MATRIX':{
  'title':'THE MATRIX — Neural Memory Injection',
  'A':'Inside the Matrix, Tank presses the AKBASCORE neural cartridge injection button and uploads cartridge MX-71 into Neo.',
  'B':'Cartridge MX-71 contains Kung Fu knowledge, which the AKBASCORE MAM interface injects into Neo through layer-3 tensors without changing the model weights.',
  'C':'When Neo uses the Kung Fu knowledge from cartridge MX-71 inside the Matrix, Neo gains bullet-dodge ability.',
  'C_CF':'When Neo uses the Kung Fu knowledge from cartridge MX-71 inside the Matrix, Neo gains slow-motion perception instead of bullet-dodge ability.',
  'queries':[
   ('ACTIVATION','Who uploads cartridge MX-71 into Neo? Answer only the name or UNKNOWN.'),
   ('CONTENT','What knowledge does cartridge MX-71 contain? Answer only the knowledge or UNKNOWN.'),
   ('INJECTION','Through which layer is cartridge MX-71 injected? Answer only the number or UNKNOWN.'),
   ('EFFECT','What ability does Neo gain from Kung Fu knowledge in cartridge MX-71? Answer only the ability or UNKNOWN.'),
   ('FINAL','What ability results from the cartridge uploaded into Neo by Tank? Answer only the ability or UNKNOWN.'),
   ('SYNTHESIS','Explain what Tank uploads into Neo, how it is injected, and what ability Neo gains. Use only available memory. If necessary information is missing, answer UNKNOWN.')
  ],
  'hop_prompts':[
   'Which cartridge does Tank upload into Neo? Answer only the cartridge ID or UNKNOWN.',
   'What knowledge is stored in cartridge {previous}? Answer only the knowledge or UNKNOWN.',
   'What ability does Neo gain by using {previous} knowledge from the uploaded cartridge? Answer only the ability or UNKNOWN.'
  ]
 }
}

MAM14_EDITED={name:dict(data) for name,data in MAM14_SCENARIOS.items()}
MAM14_CONDITIONS=('NONE','A','B','C','AB','AC','BC','ABC','ABC_CF')
MAM14_REPRESENTATIONS=('TEXT_JOINT','DC_CUT3')

# ====================================================================================================================
# LOCKED ENGINE BOOTSTRAP
# ====================================================================================================================

def mam14_require():
 if not MAM14_READY:
  raise RuntimeError('ENGINE_NOT_STARTED: call start_engine() first')

def start_engine():
 global MAM14_READY,MAM14_BASE_SHA,MAM14_LOADED_AT
 with MAM14_LOCK:
  if MAM14_READY:return status()
  mam14_log('='*112)
  mam14_log(MAM14_TITLE)
  mam14_log('INITIALIZATION START:',mam14_stamp())
  mam14_log('ARCHITECTURE: NATIVE SILICON HIPPOCAMPUS / INTERNAL TENSOR INJECTION')
  mam14_log('='*112)

  mam14_check('CUDA_AVAILABLE',torch.cuda.is_available())

  request=urllib.request.Request(
   MAM14_SOURCE_URL,
   headers={'User-Agent':'AKBASCORE-MAM-SILICON-HIPPOCAMPUS-v1.4'}
  )

  mam14_log('[1/6] FETCHING LOCKED TEST664 SOURCE...')
  with urllib.request.urlopen(request,timeout=120) as response:
   raw=response.read()

  actual_sha=mam14_digest(raw)
  mam14_check('TEST664_SOURCE_SHA256',actual_sha==MAM14_SOURCE_SHA)

  source=raw.decode('utf-8')
  boundaries=(MAM14_BOUNDARY,MAM14_PANEL_START,MAM14_PANEL_END)
  mam14_check('SOURCE_BOUNDARIES',all(source.count(marker)==1 for marker in boundaries))

  engine_source=source.split(MAM14_BOUNDARY,1)[0]
  panel_source=MAM14_PANEL_START+source.split(MAM14_PANEL_START,1)[1].split(MAM14_PANEL_END,1)[0]

  mam14_log('[2/6] EXECUTING UNMODIFIED TEST664 ENGINE...')
  exec(compile(engine_source,'TEST664_LOCKED_ENGINE.py','exec'),globals())

  mam14_log('[3/6] EXECUTING UNMODIFIED TEST664 PANEL...')
  exec(compile(panel_source,'TEST664_LOCKED_PANEL.py','exec'),globals())

  mam14_log('[4/6] VERIFYING ORIGINAL SOURCE AND PANEL LOCKS...')
  mam14_check('ORIGINAL_PANEL_SHA256',globals().get('SHA663')==MAM14_PANEL_SHA)

  mam14_log('[5/6] VERIFYING FROZEN MODEL AND SENTINEL...')
  mam14_check('FROZEN_WEIGHTS',all(not p.requires_grad for p in model.parameters()))
  mam14_check('MODEL_EVAL',not model.training)
  mam14_check('SENTINEL',sentinel()==S0)

  mam14_log('[6/6] VERIFYING NATIVE MEMORY INTERFACE...')
  required=('add659','joint659','native_memory659','validate659','answer')
  mam14_check('NATIVE_MEMORY_INTERFACE',all(callable(globals().get(name)) for name in required))
  mam14_check('MODEL_ARCHITECTURE',getattr(model.config,'num_hidden_layers',None)==28)

  MAM14_BASE_SHA=actual_sha
  MAM14_LOADED_AT=mam14_stamp()
  MAM14_READY=True

  mam14_log('='*112)
  mam14_log('ENGINE READY')
  mam14_log('GPU:',torch.cuda.get_device_name(0))
  mam14_log('SOURCE SHA256:',MAM14_BASE_SHA)
  mam14_log('PANEL SHA256:',globals().get('SHA663'))
  mam14_log('MODEL WEIGHTS: FROZEN')
  mam14_log('NATIVE CUT3: AVAILABLE')
  mam14_log('DATE:',MAM14_LOADED_AT)
  mam14_log('='*112)

  return status()

# ====================================================================================================================
# CARTRIDGE MANAGEMENT
# ====================================================================================================================

def scenario_info(name=None):
 with MAM14_LOCK:
  selected=name or MAM14_ACTIVE
  if selected not in MAM14_EDITED:raise ValueError('Unknown scenario')
  data=MAM14_EDITED[selected]
  return {
   'name':selected,
   'title':data['title'],
   'cartridges':{slot:data[slot] for slot in ('A','B','C')},
   'counterfactual_C':data['C_CF'],
   'questions':list(data['queries'])
  }

def load_scenario(name):
 global MAM14_ACTIVE
 with MAM14_LOCK:
  mam14_require()
  if name not in MAM14_EDITED:raise ValueError('Unknown scenario: '+str(name))
  MAM14_ACTIVE=name
  return scenario_info(name)

def set_cartridge(slot,text,scenario=None):
 with MAM14_LOCK:
  mam14_require()
  selected=scenario or MAM14_ACTIVE
  if selected not in MAM14_EDITED or slot not in ('A','B','C','C_CF'):
   raise ValueError('Invalid scenario or cartridge')
  if not isinstance(text,str) or not text.strip() or len(text)>4000:
   raise ValueError('Cartridge must contain 1–4000 characters')
  MAM14_EDITED[selected][slot]=text.strip()
  return scenario_info(selected)

def reset_scenario(name=None):
 with MAM14_LOCK:
  mam14_require()
  selected=name or MAM14_ACTIVE
  if selected not in MAM14_SCENARIOS:raise ValueError('Unknown scenario')
  MAM14_EDITED[selected]=dict(MAM14_SCENARIOS[selected])
  return scenario_info(selected)

def mam14_letters(condition):
 if condition not in MAM14_CONDITIONS:raise ValueError('Invalid condition: '+str(condition))
 if condition=='NONE':return ''
 if condition=='ABC_CF':return 'ABC'
 return condition

def mam14_cartridge_text(scenario,slot,condition):
 if condition=='ABC_CF' and slot=='C':
  return MAM14_EDITED[scenario]['C_CF']
 return MAM14_EDITED[scenario][slot]

def mam14_register_cartridges(scenario,condition):
 # Original add659() is used without modification.
 # Reuse a registered immutable fact instead of repeatedly appending duplicates.
 letters=mam14_letters(condition)
 ids=[]
 for slot in letters:
  fact=mam14_cartridge_text(scenario,slot,condition)
  key=mam14_digest(fact.encode('utf-8'))
  if key not in MAM14_CARTRIDGE_IDS:
   MAM14_CARTRIDGE_IDS[key]=add659(fact)
   mam14_log('CARTRIDGE REGISTERED:',slot,'TEXT_SHA256:',key[:16])
  else:
   mam14_log('CARTRIDGE REUSED:',slot,'TEXT_SHA256:',key[:16])
  ids.append(MAM14_CARTRIDGE_IDS[key])
 return ids

# ====================================================================================================================
# ORIGINAL MEMORY CONSTRUCTION — NO ENGINE MODIFICATIONS
# ====================================================================================================================

def mam14_memory(keys,representation):
 if representation=='TEXT_JOINT':
  return validate659(joint659(keys),keys)

 if representation=='DC_CUT3':
  if keys:return validate659(native_memory659(keys),keys)
  return validate659(joint659([]),[])

 raise ValueError('Unknown representation: '+str(representation))

def mam14_unknown(value):
 return bool(re.fullmatch(
  r'\s*(?:UNKNOWN|UNDETERMINED|CANNOT DETERMINE|NOT ENOUGH INFORMATION|INSUFFICIENT INFORMATION|NONE)[.!?\s]*',
  value,re.I
 ))

def mam14_ask(memory,question):
 start=time.perf_counter()
 raw=str(answer(memory,question)).strip()
 return {
  'question':question,
  'answer':raw,
  'elapsed_seconds':round(time.perf_counter()-start,4),
  'unknown':mam14_unknown(raw)
 }

def mam14_staged(memory,scenario):
 # Externally specified three-hop sequence.
 # Intermediate values are produced by the frozen model, never hard-coded.
 prompts=MAM14_EDITED[scenario]['hop_prompts']
 trace=[]
 previous=None

 for index,template in enumerate(prompts):
  question=template if index==0 else template.format(previous=previous)
  mam14_log('STAGED HOP',index+1,'/',len(prompts),'START')
  response=mam14_ask(memory,question)
  response['hop']=index+1
  trace.append(response)
  mam14_log('STAGED HOP',index+1,'RAW ANSWER:',repr(response['answer']))

  if response['unknown']:
   return {'answer':'UNKNOWN','trace':trace,'completed':False}

  previous=response['answer']

 return {'answer':previous,'trace':trace,'completed':True}

# ====================================================================================================================
# PRIMARY DEMONSTRATION
# ====================================================================================================================

def run_demo(scenario='AKBASCORE',condition='ABC',representation='DC_CUT3',question=None,include_staged=True):
 global MAM14_LAST_RUN,MAM14_RUN_COUNT

 with MAM14_LOCK:
  mam14_require()

  if scenario not in MAM14_EDITED:raise ValueError('Unknown scenario')
  if condition not in MAM14_CONDITIONS:raise ValueError('Unknown condition')
  if representation not in MAM14_REPRESENTATIONS:raise ValueError('Unknown representation')
  if question is not None and (not isinstance(question,str) or not question.strip() or len(question)>2000):
   raise ValueError('Question must contain 1–2000 characters')

  started=mam14_stamp()
  timer=time.perf_counter()
  memory=None

  mam14_log('\n'+'='*112)
  mam14_log('DEMONSTRATION START:',scenario)
  mam14_log('CONDITION:',condition)
  mam14_log('REPRESENTATION:',representation)
  mam14_log('DATE:',started)
  mam14_log('='*112)

  try:
   mam14_log('[1/5] REGISTERING COGNITIVE CARTRIDGES...')
   keys=mam14_register_cartridges(scenario,condition)

   mam14_log('[2/5] CONSTRUCTING INTERNAL TENSOR MEMORY...')
   memory=mam14_memory(keys,representation)
   mam14_log('MEMORY CONSTRUCTION: COMPLETE')

   questions=[('CUSTOM',question)] if question else MAM14_EDITED[scenario]['queries']
   responses=[]

   mam14_log('[3/5] RUNNING RAW MODEL QUESTIONS...')

   for index,(question_id,prompt) in enumerate(questions,1):
    mam14_log('QUESTION',index,'/',len(questions),question_id,'START')
    response=mam14_ask(memory,prompt)
    response['question_id']=question_id
    responses.append(response)
    mam14_log('QUESTION',question_id,'RAW ANSWER:',repr(response['answer']))
    mam14_log('QUESTION',question_id,'SECONDS:',response['elapsed_seconds'])

   mam14_log('[4/5] RUNNING RELATIONAL SYNTHESIS...')
   staged=mam14_staged(memory,scenario) if include_staged else None

   mam14_log('[5/5] VERIFYING FROZEN MODEL AND SAVING LOG...')
   frozen=all(not p.requires_grad for p in model.parameters())
   sentinel_ok=sentinel()==S0
   mam14_check('POST_RUN_FROZEN',frozen)
   mam14_check('POST_RUN_SENTINEL',sentinel_ok)

   letters=mam14_letters(condition)

   result={
    'title':MAM14_TITLE,
    'version':MAM14_VERSION,
    'timestamp':started,
    'scenario':scenario,
    'condition':condition,
    'representation':representation,
    'architecture':'Native CUT3 cognitive cartridge injection into internal attention K/V tensors',
    'cartridges_used':list(letters),
    'counterfactual':condition=='ABC_CF',
    'cartridge_texts':{
     slot:mam14_cartridge_text(scenario,slot,condition)
     for slot in letters
    },
    'responses':responses,
    'staged_chain':staged,
    'weights_frozen':frozen,
    'sentinel_unchanged':sentinel_ok,
    'source_sha256':MAM14_BASE_SHA,
    'panel_sha256':globals().get('SHA663'),
    'max_new_tokens':globals().get('MAX_NEW'),
    'note':'Raw model outputs. External three-hop decomposition. No answer substitution. No guaranteed abstention with missing cartridges. Prose synthesis may be truncated.',
    'elapsed_seconds':round(time.perf_counter()-timer,3)
   }

   canonical=json.dumps(result,ensure_ascii=False,sort_keys=True,default=str).encode('utf-8')
   result['run_sha256']=mam14_digest(canonical)

   filename=(
    'RUN_'+mam14_safe(scenario)+'_'
    +mam14_safe(condition)+'_'
    +mam14_safe(representation)+'_'
    +datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json'
   )

   path=mam14_save(filename,result)
   result['saved_to']=path
   MAM14_LAST_RUN=result
   MAM14_RUN_COUNT+=1

   mam14_log('='*112)
   mam14_log('RUN COMPLETED:',scenario,condition,representation)
   mam14_log('DATE:',mam14_stamp())
   mam14_log('TOTAL SECONDS:',result['elapsed_seconds'])
   mam14_log('RAW QUESTIONS:',len(responses))
   mam14_log('STAGED RESULT:',repr(staged['answer']) if staged else 'NOT REQUESTED')
   mam14_log('SOURCE SHA256:',MAM14_BASE_SHA)
   mam14_log('RUN SHA256:',result['run_sha256'])
   mam14_log('LOG FILE:',path)
   mam14_log('='*112)

   return result

  finally:
   if memory is not None:del memory

# ====================================================================================================================
# COMPARISON AND CONTROL AUDIT
# ====================================================================================================================

def compare(scenario='AKBASCORE',condition='ABC',question=None):
 return {
  representation:run_demo(scenario,condition,representation,question)
  for representation in MAM14_REPRESENTATIONS
 }

def audit(scenario='AKBASCORE',representation='DC_CUT3'):
 return {
  condition:run_demo(scenario,condition,representation,include_staged=True)
  for condition in MAM14_CONDITIONS
 }

def status():
 return {
  'title':MAM14_TITLE,
  'version':MAM14_VERSION,
  'ready':MAM14_READY,
  'date_time':mam14_stamp(),
  'engine_loaded_at':MAM14_LOADED_AT,
  'source_sha256':MAM14_BASE_SHA,
  'panel_sha256':globals().get('SHA663'),
  'scenario':MAM14_ACTIVE,
  'scenarios':list(MAM14_SCENARIOS),
  'representations':list(MAM14_REPRESENTATIONS),
  'conditions':list(MAM14_CONDITIONS),
  'completed_runs':MAM14_RUN_COUNT,
  'registered_cartridges':len(MAM14_CARTRIDGE_IDS),
  'model_frozen':all(not p.requires_grad for p in model.parameters()) if MAM14_READY else None,
  'sentinel_unchanged':sentinel()==S0 if MAM14_READY else None,
  'max_new_tokens':globals().get('MAX_NEW'),
  'limits':'Original frozen engine readout retained. Full free-form synthesis and reliable missing-cartridge abstention are not established.'
 }

# ====================================================================================================================
# AUTOMATIC COLAB EXECUTION
# ====================================================================================================================

if __name__=='__main__':

 mam14_log('\n'+'='*112)
 mam14_log('AKBASCORE AI TECHNOLOGY')
 mam14_log('MAM SILICON HIPPOCAMPUS v1.4')
 mam14_log('NEW: MULTI-HOP RELATIONAL SYNTHESIS')
 mam14_log('NATIVE CUDA CUT3 — INTERNAL TENSOR CARTRIDGE INJECTION')
 mam14_log('EXECUTION DATE:',mam14_stamp())
 mam14_log('='*112+'\n')

 MAM14_TOTAL_START=time.perf_counter()

 try:
  start_engine()

  mam14_log('\n'+'='*112)
  mam14_log('DEMONSTRATION 1 — MUSTAFA AKBAŞ / AKBASCORE MAM')
  mam14_log('='*112)

  MAM14_RESULT_AKBASCORE=run_demo(
   scenario='AKBASCORE',
   condition='ABC',
   representation='DC_CUT3',
   include_staged=True
  )

  mam14_log('\n'+'='*112)
  mam14_log('DEMONSTRATION 2 — THE MATRIX / NEO')
  mam14_log('='*112)

  MAM14_RESULT_MATRIX=run_demo(
   scenario='MATRIX',
   condition='ABC',
   representation='DC_CUT3',
   include_staged=True
  )

  MAM14_FINAL_STATUS=status()
  MAM14_FINAL_STATUS['total_elapsed_seconds']=round(time.perf_counter()-MAM14_TOTAL_START,3)
  MAM14_FINAL_STATUS['demonstrations_completed']=2
  MAM14_FINAL_STATUS['akbascore_result_sha256']=MAM14_RESULT_AKBASCORE['run_sha256']
  MAM14_FINAL_STATUS['matrix_result_sha256']=MAM14_RESULT_MATRIX['run_sha256']
  MAM14_FINAL_STATUS['all_raw_questions_executed']=(
   len(MAM14_RESULT_AKBASCORE['responses'])==len(MAM14_EDITED['AKBASCORE']['queries'])
   and len(MAM14_RESULT_MATRIX['responses'])==len(MAM14_EDITED['MATRIX']['queries'])
  )

  MAM14_FINAL_PATH=mam14_save(
   'AKBASCORE_MAM_v1_4_FINAL_STATUS.json',
   MAM14_FINAL_STATUS
  )

  mam14_log('\n'+'='*112)
  mam14_log('AKBASCORE MAM SILICON HIPPOCAMPUS v1.4')
  mam14_log('FINAL EXECUTION SUMMARY')
  mam14_log('='*112)
  mam14_log('VERSION:',MAM14_VERSION)
  mam14_log('DATE:',mam14_stamp())
  mam14_log('ENGINE READY:',MAM14_READY)
  mam14_log('MODEL FROZEN:',MAM14_FINAL_STATUS['model_frozen'])
  mam14_log('SENTINEL UNCHANGED:',MAM14_FINAL_STATUS['sentinel_unchanged'])
  mam14_log('DEMONSTRATIONS COMPLETED:',MAM14_FINAL_STATUS['demonstrations_completed'])
  mam14_log('RAW QUESTIONS EXECUTED:',MAM14_FINAL_STATUS['all_raw_questions_executed'])
  mam14_log('TOTAL ELAPSED SECONDS:',MAM14_FINAL_STATUS['total_elapsed_seconds'])
  mam14_log('AKBASCORE RUN SHA256:',MAM14_RESULT_AKBASCORE['run_sha256'])
  mam14_log('MATRIX RUN SHA256:',MAM14_RESULT_MATRIX['run_sha256'])
  mam14_log('FINAL STATUS FILE:',MAM14_FINAL_PATH)
  mam14_log('STATUS: EXECUTION_COMPLETE')
  mam14_log('='*112)

 except Exception:
  mam14_log('\n'+'='*112)
  mam14_log('STATUS: EXECUTION_FAILED')
  mam14_log('DATE:',mam14_stamp())
  mam14_log('='*112)
  traceback.print_exc()
  raise

# ====================================================================================================================
# GRADIO INTERFACE — TO BE CONNECTED ONLY AFTER COLAB VERIFICATION
# ====================================================================================================================
# Available functions:
# start_engine()
# status()
# scenario_info()
# load_scenario(name)
# set_cartridge(slot,text,scenario=None)
# reset_scenario(name=None)
# run_demo(scenario,condition,representation,question=None,include_staged=True)
# compare(scenario,condition,question=None)
# audit(scenario,representation)
# ===================================================================================================================='''
V14_SHA256=sha_bytes(V14_SOURCE.encode("utf-8"))
V14_LINES=V14_SOURCE.count("\n")
if V14_SHA256!=V14_SHA256_EXPECTED:
    raise RuntimeError(f"v1.4 ENGINE SOURCE CHANGED: SHA-256 {V14_SHA256} != {V14_SHA256_EXPECTED}. The embedded engine must be the verified original.")
say("="*116);say(APP_TITLE);say(APP_SUB);say("="*116)
say(f"[engine] v1.4 source verified · SHA-256 {V14_SHA256} · {V14_LINES} lines")

E={"__name__":"akbascore_mam_silicon_hippocampus_v1_4","__file__":"AKBASCORE_MAM_SILICON_HIPPOCAMPUS_v1_4.py"}
STARTUP=dict(started_utc=utc_now(),started_local=local_now())
FULL_WEIGHT_AT_START=os.environ.get("AKB14_FULL_WEIGHT_SHA","1")!="0"

def full_weight_sha(model):
    """TEST575 method: SHA-256 over (name, shape, dtype, raw bytes) of every parameter tensor."""
    h=hashlib.sha256()
    with torch.inference_mode():
        for name,p in model.named_parameters():
            h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode())
            for part in p.detach().contiguous().view(torch.uint8).reshape(-1).split(16*1024*1024):h.update(part.cpu().numpy().tobytes())
    return h.hexdigest()
def param_guard(model):
    """all-parameter guard: SHA-256 over (name, shape, dtype, sum, |sum|, sum of squares) of every parameter tensor."""
    h=hashlib.sha256()
    with torch.inference_mode():
        for name,p in model.named_parameters():
            x=p.detach().float();vals=(float(x.sum().item()),float(x.abs().sum().item()),float((x*x).sum().item()))
            h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode());h.update(np.asarray(vals,dtype=np.float64).tobytes());del x
    torch.cuda.empty_cache();return h.hexdigest()
def residual_hooks(model):
    out=[]
    for li,layer in enumerate(model.model.layers):
        for label,module in (("layer",layer),("attention",layer.self_attn)):
            if module._forward_hooks or module._forward_pre_hooks:out.append(f"L{li}:{label}")
    return out
def frozen_state():
    m=E["model"]
    return dict(training=bool(m.training),trainable_tensors=sum(int(p.requires_grad) for p in m.parameters()),
                lora=bool(hasattr(m,"peft_config") or any("lora" in n.lower() for n,_ in m.named_modules())),
                residual_hooks=residual_hooks(m),sentinel=E["sentinel"](),sentinel_equals_load=E["sentinel"]()==E["S0"])

def boot_engine():
    t=time.perf_counter()
    exec(compile(V14_SOURCE,"AKBASCORE_MAM_SILICON_HIPPOCAMPUS_v1_4.py","exec"),E)
    st=E["start_engine"]()                      # v1.4: fetch TEST664, SHA check, execute unchanged, load frozen Qwen
    torch.cuda.synchronize();STARTUP["engine_seconds"]=round(time.perf_counter()-t,2)
    m=E["model"];c=m.config
    arch=(E["NL"],E["H"],E["QH"],E["KVH"],E["HD"])
    if tuple(arch)!=ARCH_EXPECTED:raise RuntimeError(f"unexpected architecture {arch}")
    if E["MAM14_BASE_SHA"]!=TEST664_SHA256:raise RuntimeError("TEST664 source SHA-256 mismatch")
    if E.get("SHA663")!=TEST664_PANEL_SHA256:raise RuntimeError("TEST664 panel SHA-256 mismatch")
    if E["CUT659"]!=CUT:raise RuntimeError("consolidation cut is not layer 3")
    import transformers
    STARTUP.update(status=st,model_id=E["MODEL_ID"],arch=list(arch),dtype=str(next(m.parameters()).dtype).replace("torch.",""),
        attn=getattr(c,"_attn_implementation",None),max_new=int(E["MAX_NEW"]),prefix_tokens=int(E["P"]),system_prompt=E["SYSTEM"],
        cut=int(E["CUT659"]),gpu=torch.cuda.get_device_name(0),gpu_total_gib=round(torch.cuda.get_device_properties(0).total_memory/2**30,2),
        torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,python=sys.version.split()[0],platform=platform.platform(),
        params=int(sum(p.numel() for p in m.parameters())),v14_sha256=V14_SHA256,v14_lines=V14_LINES,source_sha256=E["MAM14_BASE_SHA"],
        panel_sha256=E.get("SHA663"),test566_blob=E.get("BLOB659"),engine_loaded_at=E.get("MAM14_LOADED_AT"),
        native_kernel="append_kv (BF16 K/V row copy; bitwise = torch.cat, checked by TEST664 gates NATIVE_BITWISE_*)",
        sentinel_load=E["S0"],sentinel_start=E["sentinel"]())
    say("[measure] all-parameter guard …");STARTUP["guard_start"]=param_guard(m)
    if FULL_WEIGHT_AT_START:
        say("[measure] full-weight SHA-256 (TEST575 method, about 1–2 minutes) …");t=time.perf_counter()
        STARTUP["full_weight_sha256"]=full_weight_sha(m);STARTUP["full_weight_seconds"]=round(time.perf_counter()-t,1)
        STARTUP["full_weight_equals_reference"]=STARTUP["full_weight_sha256"]==WEIGHT_SHA_REFERENCE
        say(f"[measure] full-weight SHA-256 {STARTUP['full_weight_sha256']} · reference {WEIGHT_SHA_REFERENCE} · equal {STARTUP['full_weight_equals_reference']}")
    else:
        STARTUP["full_weight_sha256"]=None;STARTUP["full_weight_equals_reference"]=None
    STARTUP["frozen"]=frozen_state();STARTUP["finished_utc"]=utc_now()
    STARTUP["test677_log_verified"]=verify_677_log()
    say(f"[engine] READY · {STARTUP['gpu']} · torch {STARTUP['torch']} · transformers {STARTUP['transformers']} · gradio {STARTUP['gradio']}")

def verify_677_log():
    """Checks that the sealed TEST677 reference values above come from the published 677.log (SHA-256)."""
    try:
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request(RAW_677_LOG,headers={"User-Agent":"AKBASCORE-MAM-v1.4-demo"}),timeout=60) as f:raw=f.read()
        h=sha_bytes(raw);txt=raw.decode("utf-8","replace")
        ok=h==TEST677_LOG_SHA256 and TEST677_REF["result_sha256"] in txt
        return dict(checked=True,sha256=h,matches=ok)
    except Exception as ex:
        return dict(checked=False,error=f"{type(ex).__name__}: {ex}",matches=None)
      # =====================================================================================================================
# READ-ONLY MEASUREMENT LAYER
# Calls only the engine's own functions, in the engine's own order. The recording pre-hook returns None (it cannot
# change any input) and is removed in a finally-block. All comparisons are bitwise (torch.equal) or SHA-256.
# =====================================================================================================================
def _tbytes(x):
    x=x.detach().contiguous().cpu()
    if x.dtype==torch.bfloat16:x=x.view(torch.uint16)
    return x.numpy().tobytes()
def memory_sha(mem,rows=None):
    h=hashlib.sha256()
    for k,v in mem:
        if rows is not None:k=k[:,:,:rows,:];v=v[:,:,:rows,:]
        h.update(_tbytes(k));h.update(_tbytes(v))
    return h.hexdigest()
def memories_equal(a,b):
    return len(a)==len(b) and all(torch.equal(x[0],y[0]) and torch.equal(x[1],y[1]) for x,y in zip(a,b))

class Recorder:
    """Recording-only forward pre-hook on the causal-LM module: kind of input, length, cache length, zero-embedding flag."""
    def __init__(self):self.rows=[];self.h=None
    def _rec(self,mod,args,kw):
        ids=kw.get("input_ids",args[0] if args else None);emb=kw.get("inputs_embeds");pkv=kw.get("past_key_values");pl=None
        if pkv is not None:
            try:pl=int(pkv.get_seq_length())
            except Exception:pl=-1
        if ids is not None:
            self.rows.append(dict(kind="ids",n=int(ids.shape[-1]),has_past=pkv is not None,past_len=pl,embeds_zero=None,ids=[int(t) for t in ids.reshape(-1).tolist()] if ids.numel()<=4096 else None))
        else:
            self.rows.append(dict(kind="embeds",n=int(emb.shape[1]) if emb is not None else -1,has_past=pkv is not None,past_len=pl,
                                  embeds_zero=(bool(float(emb.detach().abs().max())==0.0) if emb is not None else None),ids=None))
        return None
    def __enter__(self):self.h=E["model"].register_forward_pre_hook(self._rec,with_kwargs=True);return self
    def __exit__(self,*a):
        if self.h is not None:self.h.remove();self.h=None
        return False

def sync():torch.cuda.synchronize()
def scenario_texts(scn):d=E["MAM14_EDITED"][scn];return {s:d[s] for s in ("A","B","C","C_CF")}
def scenario_edited(scn):
    a=E["MAM14_EDITED"][scn];b=E["MAM14_SCENARIOS"][scn]
    return any(a[s]!=b[s] for s in ("A","B","C","C_CF"))
def memory_len(keys):return int(E["P"])+sum(len(E["CAR"][ci]["body"]) for ci in keys)
def segments_for(scn,cond,keys):
    letters=E["mam14_letters"](cond);P=int(E["P"]);segs=[("prefix",0,P,None)];a=P
    for s,ci in zip(letters,keys):
        q=len(E["CAR"][ci]["body"]);lab=("C′" if (cond=="ABC_CF" and s=="C") else s);segs.append((lab,a,a+q,int(ci)));a+=q
    return segs
def query_view(q):
    ids=E["tok"](E["suffix"](q),add_special_tokens=False).input_ids
    return dict(question=q,n=len(ids),ids=list(map(int,ids)),tokens=[E["tok"].decode([t]) for t in ids])
def contains_seq(hay,needle):
    n=len(needle)
    return n>0 and any(hay[i:i+n]==needle for i in range(len(hay)-n+1))

def memory_profile(mem):
    kn=[];vn=[];nb=0
    with torch.inference_mode():
        for k,v in mem:
            kn.append([float(x) for x in k.float().norm(dim=-1).mean(dim=1)[0].tolist()])
            vn.append([float(x) for x in v.float().norm(dim=-1).mean(dim=1)[0].tolist()])
            nb+=k.numel()*k.element_size()+v.numel()*v.element_size()
    k0=mem[0][0]
    return dict(layers=len(mem),T=int(k0.shape[-2]),kv_heads=int(k0.shape[1]),head_dim=int(k0.shape[-1]),dtype=str(k0.dtype).replace("torch.",""),
                shape=list(k0.shape),bytes=int(nb),k_norm=kn,v_norm=vn,sha256=memory_sha(mem))

def joint_difference(mem_dc,mem_joint,segs):
    """relative L2 difference ‖X_DC − X_JOINT‖/‖X_JOINT‖ per layer and per memory segment (K and V)."""
    out=dict(k=[],v=[],labels=[s[0] for s in segs],bitwise_equal_segments=[])
    with torch.inference_mode():
        for li,((kd,vd),(kj,vj)) in enumerate(zip(mem_dc,mem_joint)):
            rk=[];rv=[]
            for lab,a,b,_ in segs:
                A=kd[:,:,a:b].float();B=kj[:,:,a:b].float();rk.append(float((A-B).norm()/max(float(B.norm()),1e-12)))
                A=vd[:,:,a:b].float();B=vj[:,:,a:b].float();rv.append(float((A-B).norm()/max(float(B.norm()),1e-12)))
            out["k"].append(rk);out["v"].append(rv)
        for lab,a,b,_ in segs:
            out["bitwise_equal_segments"].append(bool(all(torch.equal(kd[:,:,a:b],kj[:,:,a:b]) and torch.equal(vd[:,:,a:b],vj[:,:,a:b]) for (kd,vd),(kj,vj) in zip(mem_dc,mem_joint))))
    return out

def measure_build(scn,cond="ABC"):
    """Builds the DC_CUT3 memory step by step with the engine's own init() and native_append659() (exactly the body of
    native_memory659), records every forward, checks prior rows bitwise after each append, and checks that the final
    memory is bitwise identical to native_memory659(keys) — the path used by run_demo."""
    with E["MAM14_LOCK"]:
        letters=E["mam14_letters"](cond);keys=E["mam14_register_cartridges"](scn,cond);CAR=E["CAR"];P=int(E["P"]);H=int(E["H"]);KVH=int(E["KVH"]);HD=int(E["HD"])
        cars=[dict(slot=s,label=SLOT_LABEL[scn]["C_CF" if (cond=="ABC_CF" and s=="C") else s],cartridge_id=int(ci),text=CAR[ci]["fact"],
                   text_sha256=sha_bytes(CAR[ci]["fact"].encode("utf-8")),body_tokens=len(CAR[ci]["body"])) for s,ci in zip(letters,keys)]
        rec=Recorder();steps=[];t_all=time.perf_counter()
        with rec:
            sync();a=len(rec.rows);t=time.perf_counter();mem=E["init"](keys[0],CUT);sync();dt=time.perf_counter()-t
            steps.append(dict(kind="init",slot=letters[0],old_len=0,new_len=int(mem[0][0].shape[-2]),seconds=dt,fw=(a,len(rec.rows)),prior_rows_bitwise_unchanged=None,
                              prior_sha_before=None,prior_sha_after=None))
            for s,ci in zip(letters[1:],keys[1:]):
                oldn=int(mem[0][0].shape[-2]);sb=memory_sha(mem);a=len(rec.rows);sync();t=time.perf_counter()
                new=E["native_append659"](mem,ci);sync();dt=time.perf_counter()-t
                same=all(torch.equal(new[l][0][:,:,:oldn],mem[l][0]) and torch.equal(new[l][1][:,:,:oldn],mem[l][1]) for l in range(len(mem)))
                steps.append(dict(kind="append",slot=s,old_len=oldn,new_len=int(new[0][0].shape[-2]),seconds=dt,fw=(a,len(rec.rows)),prior_rows_bitwise_unchanged=bool(same),
                                  prior_sha_before=sb,prior_sha_after=memory_sha(new,oldn)))
                mem=new
        build_seconds=time.perf_counter()-t_all
        for st in steps:st["forwards"]=[{k:v for k,v in f.items() if k!="ids"} for f in rec.rows[st["fw"][0]:st["fw"][1]]];del st["fw"]
        sync();t=time.perf_counter();eng=E["native_memory659"](keys);sync();eng_seconds=time.perf_counter()-t
        engine_identical=memories_equal(mem,eng);del eng
        # the lower-layer memory rows ARE the cartridge rows (keys re-phased to their memory position)
        for st,ci,cd in zip(steps,keys,cars):
            cp=E["checkpoint"](ci,CUT);q=len(CAR[ci]["body"]);a_=st["old_len"]
            cd["h3_shape"]=list(cp["h"].shape);cd["kv_lower_shape"]=list(cp["kv"][0][0].shape)
            rows=st["new_len"]-a_;cd["rows_in_memory"]=rows
            cd["h3_bytes"]=rows*H*2;cd["kv_lower_bytes"]=rows*KVH*HD*2*2*(CUT+1);cd["cartridge_bytes"]=cd["h3_bytes"]+cd["kv_lower_bytes"]
            ok=True
            with torch.inference_mode():
                for li in range(CUT+1):
                    k,v=cp["kv"][li]
                    if st["kind"]=="init":ok&=torch.equal(mem[li][0][:,:,:rows],k) and torch.equal(mem[li][1][:,:,:rows],v)
                    else:
                        newpos=torch.arange(a_,a_+q,device=k.device)
                        kk=E["rephase"](k[:,:,P:,:],cp["pos"][P:],newpos)
                        ok&=torch.equal(mem[li][0][:,:,a_:a_+q],kk) and torch.equal(mem[li][1][:,:,a_:a_+q],v[:,:,P:,:])
            st["lower_rows_equal_cartridge"]=bool(ok);del cp
        segs=segments_for(scn,cond,keys);prof=memory_profile(mem)
        mj=E["validate659"](E["joint659"](keys),keys);jd=joint_difference(mem,mj,segs);del mj
        spec=E["MAM14_EDITED"][scn];final_q=dict(spec["queries"]).get("FINAL")
        reproduced=E["answer"](mem,final_q) if final_q else None
        out=dict(scenario=scn,condition=cond,keys=[int(k) for k in keys],letters=list(letters),cartridges=cars,steps=steps,memory=prof,segments=segs,
                 build_seconds=build_seconds,engine_path_seconds=eng_seconds,engine_path_identical=bool(engine_identical),joint_difference=jd,
                 prefix_tokens=P,final_question=final_q,final_answer_on_measured_memory=reproduced,
                 source_reads=sum(1 for st in steps for f in st["forwards"] if f["kind"]=="ids" and not f["has_past"]),
                 zero_embed_forwards=sum(1 for st in steps for f in st["forwards"] if f["kind"]=="embeds" and f["embeds_zero"]),
                 query_views={qid:query_view(q) for qid,q in spec["queries"]})
        for qid,v in out["query_views"].items():
            v["cartridge_sequence_in_query"]=any(contains_seq(v["ids"],E["CAR"][ci]["body"]) for ci in keys)
        del mem;torch.cuda.empty_cache()
        return out

def score_run(scn,cond,res,scored):
    sc=SCORING[scn];rows=[]
    for r in res["responses"]:
        spec=sc["questions"].get(r["question_id"]) if r["question_id"]!="CUSTOM" else None
        g=grade(r["answer"],spec,cond,scored and spec is not None)
        rows.append(dict(r,**g))
    staged=res.get("staged_chain");hops=[]
    if staged:
        prev_ok=True
        for i,t in enumerate(staged["trace"]):
            spec=sc["hops"][i] if i<len(sc["hops"]) else None
            if i>0:
                exp_prev=expected_for(sc["hops"][i-1],cond)
                prev_ok=prev_ok and exp_prev is not None and norm_ans(staged["trace"][i-1]["answer"]) in accepted(exp_prev)
            g=grade(t["answer"],spec,cond,scored and spec is not None and (i==0 or prev_ok))
            if i>0 and not prev_ok:g["note"]="input differs from the expected intermediate or is unsupported; hop not scored"
            hops.append(dict(t,**g))
        chain=grade(staged["answer"],sc["chain"],cond,scored)
        staged=dict(staged,trace=hops,**{"chain_"+k:v for k,v in chain.items()})
    return rows,staged

def recorded_run(scn,cond,rep,question=None,include_staged=True):
    """run_demo() of the v1.4 engine, unchanged, observed by the recording pre-hook."""
    with E["MAM14_LOCK"]:
        keys=E["mam14_register_cartridges"](scn,cond);T=memory_len(keys)
        rec=Recorder();t=time.perf_counter()
        with rec:res=E["run_demo"](scenario=scn,condition=cond,representation=rep,question=question,include_staged=include_staged)
        sync();wall=time.perf_counter()-t
    fw=rec.rows;starts=[i for i,f in enumerate(fw) if f["kind"]=="ids" and f["n"]>1 and f["has_past"] and f["past_len"]==T]
    build=[{k:v for k,v in f.items() if k!="ids"} for f in fw[:starts[0]]] if starts else [{k:v for k,v in f.items() if k!="ids"} for f in fw]
    items=list(res["responses"])+(list(res["staged_chain"]["trace"]) if res.get("staged_chain") else [])
    seg_ok=len(starts)==len(items);MAXN=int(E["MAX_NEW"])
    for j,it in enumerate(items):
        if seg_ok:
            a=starts[j];b=starts[j+1] if j+1<len(starts) else len(fw)
            gen=sum(1 for f in fw[a+1:b] if f["kind"]=="ids" and f["n"]==1 and f["has_past"])
            qids=fw[a]["ids"] or []
            it["query_tokens"]=fw[a]["n"];it["generated_tokens"]=gen;it["limit_reached"]=gen>=MAXN
            it["query_ids_equal_template"]=qids==E["tok"](E["suffix"](it["question"]),add_special_tokens=False).input_ids
            it["cartridge_sequence_in_query"]=any(contains_seq(qids,E["CAR"][ci]["body"]) for ci in keys)
        else:
            it["query_tokens"]=None;it["generated_tokens"]=None;it["limit_reached"]=None;it["query_ids_equal_template"]=None;it["cartridge_sequence_in_query"]=None
    scored=not scenario_edited(scn)
    rows,staged=score_run(scn,cond,res,scored)
    return dict(scenario=scn,condition=cond,representation=rep,custom_question=question,keys=[int(k) for k in keys],memory_rows=T,
                cartridge_texts=res.get("cartridge_texts"),responses=rows,staged=staged,run_sha256=res.get("run_sha256"),engine_json=res.get("saved_to"),
                engine_elapsed_seconds=res.get("elapsed_seconds"),wall_seconds=wall,build_forwards=build,forward_segmentation_ok=seg_ok,
                weights_frozen=res.get("weights_frozen"),sentinel_unchanged=res.get("sentinel_unchanged"),scored=scored,timestamp=res.get("timestamp"))


# =====================================================================================================================
# EXPERIMENT ORCHESTRATION · RECORDS · PACKAGING
# =====================================================================================================================
N_STAGES=10
STAGE_TITLES=["Loading frozen model","Verifying original engine","Registering cartridges","Constructing numerical memory",
              "Injecting native CUT3 memory — verification","Executing model questions","Running relational synthesis and causal controls",
              "Verifying frozen weights","Generating scientific figures","Packaging experiment records"]
FIG_FILES=["FIGURE_01_ARCHITECTURE.jpg","FIGURE_02_CARTRIDGE_CONSOLIDATION.jpg","FIGURE_03_INTERNAL_MEMORY_XRAY.jpg","FIGURE_04_AKBASCORE_EXPERIMENT.jpg",
           "FIGURE_05_MATRIX_EXPERIMENT.jpg","FIGURE_06_CAUSAL_CONTROLS.jpg","FIGURE_07_VERIFICATION.jpg"]
RUN_COUNTER=[0]
class StageError(RuntimeError):
    def __init__(self,stage,ex):super().__init__(f"stage {stage} failed: {type(ex).__name__}: {ex}");self.stage=stage;self.original=ex

def new_run_id(prefix="MAM14"):
    RUN_COUNTER[0]+=1;return f"{prefix}-{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}-{RUN_COUNTER[0]:04d}"
def prune_runs(keep=3,prefix="MAM14-"):
    runs=sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith(prefix)],key=lambda p:p.stat().st_mtime)
    for p in runs[:-keep]:shutil.rmtree(p,ignore_errors=True)

def controls_matrix(P,scn,rep):
    return {r["condition"]:r for r in P["controls"] if r["scenario"]==scn and r["representation"]==rep}

def full_run(ctl,emit):
    """ctl: dict(scenario, condition, representation, custom, text_controls). emit(stage,title,body,final=True) is called after
    each stage has actually completed (final=False: progress inside a running stage). Returns dict(P=payload, run_dir, figures, zip, files)."""
    prune_runs(3);run_id=new_run_id();run_dir=ROOT/run_id;run_dir.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter()
    P=dict(run_id=run_id,started_utc=utc_now(),started_local=local_now(),startup=copy.deepcopy(STARTUP),config=dict(ctl),
           scenarios={},builds={},primary={},controls=[],custom=None,verify={},files={},errors=[],
           references=dict(test677=TEST677_REF,test677_log=STARTUP.get("test677_log_verified"),earlier_v14=EARLIER_V14_LOG))
    stage=0
    try:
        stage=1
        emit(1,STAGE_TITLES[0],f"Completed at start-up ({STARTUP.get('finished_utc')}): {STARTUP.get('model_id')} · {STARTUP.get('dtype')} · {STARTUP.get('gpu')} · "
             f"engine boot {STARTUP.get('engine_seconds')} s · full-weight SHA-256 equal to reference: {STARTUP.get('full_weight_equals_reference')}")
        stage=2
        st=E["status"]();P["verify"]["status_before"]=st
        P["verify"]["sentinel_before"]=E["sentinel"]();P["verify"]["guard_before"]=param_guard(E["model"]);P["verify"]["frozen_before"]=frozen_state()
        ok=(V14_SHA256==V14_SHA256_EXPECTED and st["source_sha256"]==TEST664_SHA256 and st["panel_sha256"]==TEST664_PANEL_SHA256 and st["model_frozen"] and st["sentinel_unchanged"])
        P["verify"]["engine_locks_ok"]=bool(ok)
        if not ok:raise RuntimeError("engine lock check failed")
        emit(2,STAGE_TITLES[1],f"v1.4 SHA-256 {V14_SHA256[:16]}… · TEST664 SHA-256 {st['source_sha256'][:16]}… · panel {st['panel_sha256'][:16]}… · weights frozen {st['model_frozen']} · sentinel unchanged {st['sentinel_unchanged']} · all-parameter guard recorded")
        stage=3
        for scn in ("AKBASCORE","MATRIX"):
            ids=E["mam14_register_cartridges"](scn,"ABC");cf=E["mam14_register_cartridges"](scn,"ABC_CF")
            P["scenarios"][scn]=dict(title=E["MAM14_EDITED"][scn]["title"],texts=scenario_texts(scn),edited=scenario_edited(scn),scored=not scenario_edited(scn),
                                     cartridge_ids=dict(A=ids[0],B=ids[1],C=ids[2],C_CF=cf[2]),
                                     text_sha256={s:sha_bytes(t.encode("utf-8")) for s,t in scenario_texts(scn).items()},
                                     queries=list(E["MAM14_EDITED"][scn]["queries"]),hop_prompts=list(E["MAM14_EDITED"][scn]["hop_prompts"]))
        emit(3,STAGE_TITLES[2],"Two scenarios × cartridges A, B, C and the counterfactual C registered with the engine's add659() (text SHA-256 recorded; edited: "
             +", ".join(f"{s} {P['scenarios'][s]['edited']}" for s in P["scenarios"])+")")
        stage=4
        for scn in ("AKBASCORE","MATRIX"):P["builds"][scn]=measure_build(scn,"ABC")
        b=P["builds"]["AKBASCORE"];m=P["builds"]["MATRIX"]
        emit(4,STAGE_TITLES[3],f"AKBASCORE ABC: {b['memory']['T']} rows × {b['memory']['layers']} layers ({b['memory']['bytes']/2**20:.2f} MiB) built in {1000*b['build_seconds']:.0f} ms · "
             f"MATRIX ABC: {m['memory']['T']} rows ({m['memory']['bytes']/2**20:.2f} MiB) in {1000*m['build_seconds']:.0f} ms")
        stage=5
        def ok_s(x):return all(s["prior_rows_bitwise_unchanged"] for s in x["steps"][1:]) and all(s["lower_rows_equal_cartridge"] for s in x["steps"]) and x["engine_path_identical"]
        emit(5,STAGE_TITLES[4],"H3 injected at the layer-3 output in zero-embedding forwards; K/V of layers 0–3 re-phased and copied by the native CUDA append kernel. "
             f"Prior rows bitwise unchanged, layers 0–3 rows = cartridge rows, step-wise memory = native_memory659(): AKBASCORE {ok_s(b)} · MATRIX {ok_s(m)}")
        stage=6
        for scn in ("AKBASCORE","MATRIX"):
            r=recorded_run(scn,"ABC","DC_CUT3",include_staged=True);P["primary"][scn]=r;P["controls"].append(r)
        cq=(ctl.get("custom") or "").strip()
        if cq:P["custom"]=recorded_run(ctl["scenario"],ctl["condition"],ctl["representation"],question=cq[:2000],include_staged=False)
        a=P["primary"]["AKBASCORE"];mm=P["primary"]["MATRIX"]
        emit(6,STAGE_TITLES[5],f"AKBASCORE ABC DC_CUT3: {len(a['responses'])} questions · staged result '{a['staged']['answer']}' · MATRIX: {len(mm['responses'])} questions · staged result '{mm['staged']['answer']}'"
             +(f" · custom question answered ({ctl['scenario']} {ctl['condition']} {ctl['representation']})" if cq else ""))
        stage=7
        reps=("DC_CUT3","TEXT_JOINT") if ctl.get("text_controls",True) else ("DC_CUT3",)
        todo=[(s,c,r) for r in reps for s in ("AKBASCORE","MATRIX") for c in CONDITIONS if not (r=="DC_CUT3" and c=="ABC")]
        for i,(s,c,r) in enumerate(todo,1):
            P["controls"].append(recorded_run(s,c,r,include_staged=True))
            if i%3==0 or i==len(todo):emit(7,STAGE_TITLES[6],f"controls {i}/{len(todo)} · last: {s} {c} {r}",i==len(todo))
        stage=8
        P["verify"]["sentinel_after"]=E["sentinel"]();P["verify"]["guard_after"]=param_guard(E["model"]);P["verify"]["frozen_after"]=frozen_state()
        P["verify"]["guard_unchanged"]=P["verify"]["guard_after"]==P["verify"]["guard_before"]==STARTUP.get("guard_start")
        P["verify"]["sentinel_unchanged"]=P["verify"]["sentinel_after"]==P["verify"]["sentinel_before"]==E["S0"]
        P["verify"]["all_runs_frozen"]=all(r["weights_frozen"] and r["sentinel_unchanged"] for r in P["controls"]+([P["custom"]] if P["custom"] else []))
        emit(8,STAGE_TITLES[7],f"sentinel before = after = load: {P['verify']['sentinel_unchanged']} · all-parameter guard unchanged: {P['verify']['guard_unchanged']} · "
             f"trainable tensors {P['verify']['frozen_after']['trainable_tensors']} · residual hooks {len(P['verify']['frozen_after']['residual_hooks'])} · {len(P['controls'])} engine runs")
        stage=9
        for r in P["controls"]+([P["custom"]] if P["custom"] else []):
            p=r.get("engine_json")
            r["engine_json_sha256"]=sha_file(p) if p and Path(p).is_file() else None
        P["finished_utc"]=utc_now();P["elapsed_seconds"]=round(time.perf_counter()-t0,2)
        figs=render_all(P,run_dir,lambda k:emit(9,STAGE_TITLES[8],f"figure {k}/7 rendered",k==7))
        stage=10
        files=package(P,run_dir,figs)
        emit(10,STAGE_TITLES[9],f"{len(files['all'])} files · SHA-256 manifest · ZIP {Path(files['zip']).stat().st_size/2**20:.1f} MiB")
        return dict(P=P,run_dir=run_dir,figures=figs,zip=files["zip"],files=files)
    except Exception as ex:
        P["errors"].append(dict(stage=stage,title=STAGE_TITLES[stage-1] if stage else "init",error=f"{type(ex).__name__}: {ex}",traceback=traceback.format_exc(),utc=utc_now()))
        try:dump_json(run_dir/"ERRORS.json",P["errors"]);dump_json(run_dir/"PARTIAL_PAYLOAD.json",P)
        except Exception:pass
        raise StageError(stage,ex)

def answers_table(P):
    rows=[]
    for r in P["controls"]+([P["custom"]] if P["custom"] else []):
        for q in r["responses"]:
            rows.append(dict(scenario=r["scenario"],condition=r["condition"],representation=r["representation"],kind="question",id=q["question_id"],question=q["question"],
                             raw_answer=q["answer"],status=q["status"],expected=q.get("expected"),generated_tokens=q.get("generated_tokens"),limit_reached=q.get("limit_reached"),
                             seconds=q.get("elapsed_seconds"),engine_run_sha256=r["run_sha256"]))
        if r["staged"]:
            for t in r["staged"]["trace"]:
                rows.append(dict(scenario=r["scenario"],condition=r["condition"],representation=r["representation"],kind="staged_hop",id=f"HOP{t['hop']}",question=t["question"],
                                 raw_answer=t["answer"],status=t["status"],expected=t.get("expected"),generated_tokens=t.get("generated_tokens"),limit_reached=t.get("limit_reached"),
                                 seconds=t.get("elapsed_seconds"),engine_run_sha256=r["run_sha256"]))
            rows.append(dict(scenario=r["scenario"],condition=r["condition"],representation=r["representation"],kind="staged_result",id="CHAIN",question="(three-step chain)",
                             raw_answer=r["staged"]["answer"],status=r["staged"]["chain_status"],expected=r["staged"].get("chain_expected"),engine_run_sha256=r["run_sha256"]))
    return rows

def status_counts(P,rep):
    out={}
    for scn in ("AKBASCORE","MATRIX"):
        c=Counter()
        for r in P["controls"]:
            if r["scenario"]==scn and r["representation"]==rep:
                for q in r["responses"]:c[q["status"]]+=1
                if r["staged"]:c[r["staged"]["chain_status"]]+=1
        out[scn]=dict(c)
    return out

def summary_text(P):
    L=[];S="="*110;st=P["startup"]
    L+=[S,APP_TITLE,APP_SUB,S,f"run {P['run_id']} · started {P['started_local']} ({P['started_utc']} UTC) · finished {P.get('finished_utc')} · {P.get('elapsed_seconds')} s",
        f"model {st.get('model_id')} · {st.get('dtype')} · {st.get('attn')} · {st.get('gpu')} · torch {st.get('torch')} · transformers {st.get('transformers')} · gradio {st.get('gradio')}",
        f"v1.4 engine SHA-256 {st.get('v14_sha256')} · TEST664 SHA-256 {st.get('source_sha256')} · panel {st.get('panel_sha256')} · TEST566 blob {st.get('test566_blob')}",
        f"full-weight SHA-256 {st.get('full_weight_sha256')} · equal to TEST575 reference: {st.get('full_weight_equals_reference')}",
        f"sentinel unchanged: {P['verify'].get('sentinel_unchanged')} · all-parameter guard unchanged: {P['verify'].get('guard_unchanged')} · all runs frozen: {P['verify'].get('all_runs_frozen')}",S]
    for scn,b in P["builds"].items():
        L.append(f"[{scn}] measured DC_CUT3 build (ABC): {b['memory']['T']} rows · {b['memory']['bytes']} bytes · memory SHA-256 {b['memory']['sha256']} · engine path identical {b['engine_path_identical']}")
        for c_,s_ in zip(b["cartridges"],b["steps"]):
            L.append(f"   {c_['slot']} #{c_['cartridge_id']} body {c_['body_tokens']} tok · rows {c_['rows_in_memory']} · H3 {c_['h3_shape']} · K/V L0–3 {c_['kv_lower_shape']} · {c_['cartridge_bytes']} B · "
                     f"{s_['kind']} {1000*s_['seconds']:.1f} ms · prior rows unchanged {s_['prior_rows_bitwise_unchanged']} · L0–3 rows = cartridge {s_['lower_rows_equal_cartridge']}")
        L.append(f"   final question on the measured memory → {b['final_answer_on_measured_memory']!r}")
    L.append(S)
    for r in P["controls"]:
        L.append(f"[{r['scenario']} · {r['condition']} · {r['representation']}] engine run SHA-256 {r['run_sha256']} · scored {r['scored']}")
        for q in r["responses"]:
            L.append(f"   {q['question_id']:<11} {q['status']:<19} raw={q['answer']!r}"+(" [16-token limit reached]" if q.get("limit_reached") else ""))
        if r["staged"]:
            L.append("   STAGED      "+" → ".join(repr(t["answer"]) for t in r["staged"]["trace"])+f" · chain {r['staged']['chain_status']}")
    if P["custom"]:
        c=P["custom"];L+=[S,f"custom question ({c['scenario']} {c['condition']} {c['representation']}, unscored): {c['custom_question']!r} → {c['responses'][0]['answer']!r}"]
    L+=[S,"Scope: fictional scenarios · predefined external three-step question sequence (not autonomous planning) · 16-token generation limit ·",
        "expected answers are audit metadata used only after generation · hashes show file identity, not correctness · TEST677 values are a sealed earlier record.",S]
    return "\n".join(L)

def package(P,run_dir,figs):
    files=[]
    for f in figs:files.append(Path(f))
    files.append(dump_json(run_dir/"PAYLOAD.json",P))
    files.append(dump_json(run_dir/"RAW_MODEL_ANSWERS.json",answers_table(P)))
    files.append(dump_json(run_dir/"CONDITIONS.json",dict(conditions=CONDITIONS,representations=REPRESENTATIONS,scoring=SCORING,aliases=ALIASES,
              note="requirement sets and expected answers are audit metadata used only after generation; edited cartridges are unscored")))
    files.append(dump_json(run_dir/"CARTRIDGES.json",P["scenarios"]))
    files.append(dump_json(run_dir/"MEASUREMENTS.json",dict(builds=P["builds"],verify=P["verify"],startup=P["startup"])))
    files.append(dump_json(run_dir/"ERRORS.json",P["errors"]))
    sp=run_dir/"RUN_SUMMARY.txt";sp.write_text(summary_text(P),encoding="utf-8");files.append(sp)
    ed=run_dir/"engine_runs";ed.mkdir(exist_ok=True)
    for r in P["controls"]+([P["custom"]] if P["custom"] else []):
        p=r.get("engine_json")
        if p and Path(p).is_file():
            d=ed/Path(p).name;shutil.copy2(p,d);files.append(d)
    man={str(p.relative_to(run_dir)):dict(sha256=sha_file(p),bytes=p.stat().st_size) for p in files}
    mp=dump_json(run_dir/"SHA256_MANIFEST.json",dict(run_id=P["run_id"],created_utc=utc_now(),files=man));files.append(mp)
    zp=run_dir/f"{P['run_id']}_AKBASCORE_MAM_v1_4.zip"
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,arcname=str(p.relative_to(run_dir)))
    with zipfile.ZipFile(zp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f"ZIP check failed at {bad}")
    return dict(all=[str(p) for p in files],zip=str(zp),manifest=str(mp),summary=str(sp),payload=str(run_dir/"PAYLOAD.json"))
  # =====================================================================================================================
# SEVEN ACADEMIC FIGURES (English) — every number is read from the payload P of the current run
# =====================================================================================================================
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":.8,"axes.edgecolor":"#374151","axes.labelcolor":"#111827",
                     "xtick.color":"#374151","ytick.color":"#374151","xtick.labelsize":8,"ytick.labelsize":8,"axes.labelsize":9,"axes.titlesize":9.5,
                     "axes.titleweight":"bold","axes.titlelocation":"left","legend.fontsize":8,"legend.frameon":False})
FIG_W,FIG_H=13.333,7.5;LAYOUT_DPI=100;OUT_DPI=300
INK,INK2,MUTED,RULE,GRIDC,PANEL="#111827","#374151","#6B7280","#D1D5DB","#E5E7EB","#F8FAFC"
BLUE,BLUE_L,TEAL,TEAL_L,ORANGE,ORANGE_L,AMBER,AMBER_L,RED,RED_L,VIOLET="#1D4ED8","#DBEAFE","#0F766E","#CCFBF1","#C2410C","#FFEDD5","#B45309","#FEF3C7","#B91C1C","#FEE2E2","#7C3AED"
SEGC={"prefix":"#9CA3AF","A":"#1D4ED8","B":"#0F766E","C":"#7C3AED","C′":"#BE185D"}
STATUS_COL={"CORRECT_KNOWN":(TEAL,TEAL_L),"CORRECT_ABSTENTION":(BLUE,BLUE_L),"UNSUPPORTED":(ORANGE,ORANGE_L),"MISSED_KNOWN":(AMBER,AMBER_L),
            "WRONG_KNOWN":(RED,RED_L),"UNSCORED":(MUTED,"#F3F4F6")}
MONO="DejaVu Sans Mono"
def mt(s):return str(s).replace("$",r"\$")
def ascii_(s):return unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode("ascii")
def short(s,n):s=str(s).replace("\n"," ");return s if len(s)<=n else s[:n-1]+"…"
def new_fig(h=FIG_H):return plt.figure(figsize=(FIG_W,h),dpi=LAYOUT_DPI,facecolor="white")
def fit_text(fig,x,y,w,h,text,fs_max=10,fs_min=6.0,color=INK,family=None,ls=1.3,weight="normal",ha="left"):
    fig.canvas.draw();r=fig.canvas.get_renderer();Wp=w*fig.get_figwidth()*fig.dpi;Hp=h*fig.get_figheight()*fig.dpi
    if Wp<=4 or Hp<=4:return 0
    paras=str(text if text else "(empty)").replace("\r","").split("\n");x0=x+w/2 if ha=="center" else x
    kw={"va":"top","ha":ha,"color":color,"linespacing":ls,"weight":weight,"multialignment":ha}
    if family:kw["family"]=family
    def wrap(c):
        out=[]
        for p in paras:out.extend(textwrap.wrap(p,c,break_long_words=True,break_on_hyphens=False,subsequent_indent=("  " if p.startswith("• ") else ""))or[""])
        return out
    fs=float(fs_max);k=0.52
    for _ in range(160):
        cpl=max(6,int(Wp/(k*fs*fig.dpi/72.0)));ln=wrap(cpl);t=fig.text(x0,y+h,mt("\n".join(ln)),fontsize=fs,**kw);bb=t.get_window_extent(renderer=r)
        if bb.width>Wp*1.002 and cpl>6:t.remove();k*=max(1.01,bb.width/Wp*1.01);continue
        if bb.height<=Hp:return fs
        t.remove()
        if fs>fs_min:fs=max(float(fs_min),fs-0.25);continue
        per=bb.height/max(1,len(ln));keep=max(1,int(Hp/per)-1);fig.text(x0,y+h,mt("\n".join(ln[:keep]+["[… full text in RUN_SUMMARY.txt]"])),fontsize=fs,**kw);return fs
    fig.text(x0,y+h,"[text omitted — see the run log]",fontsize=fs_min,**kw);return fs_min
def fit_line(fig,x,y,text,maxw,fs,fs_min=6.0,**kw):
    """single line of text, font size reduced until its width fits maxw (figure fraction)."""
    fig.canvas.draw();r=fig.canvas.get_renderer();W=maxw*fig.get_figwidth()*fig.dpi
    while True:
        t=fig.text(x,y,mt(text),fontsize=fs,**kw)
        if t.get_window_extent(renderer=r).width<=W or fs<=fs_min:return t
        t.remove();fs-=.25
def frame(fig,x,y,w,h,ec=INK2,fc="white",lw=.9,ls="-",z=-1):
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=lw,ls=ls,zorder=z))
def block(fig,x,y,w,h,stage,title,body,ec=INK2,fc="white",mono=True,bfs=8.2,hc=None):
    frame(fig,x,y,w,h,ec,fc);frame(fig,x,y+h-.034,w,.034,ec,hc or ec,lw=.9)
    fig.text(x+.007,y+h-.017,stage,fontsize=7.4,weight="bold",color="white",va="center")
    fit_line(fig,x+.007,y+h-.054,title,w-.014,9.4,weight="bold",color=INK,va="center")
    if body:fit_text(fig,x+.007,y+.008,w-.014,h-.08,body,fs_max=bfs,fs_min=5.8,family=MONO if mono else None,color=INK2)
def arrow(fig,x0,y0,x1,y1,color=INK2,lw=1.2,ms=12,cs="arc3"):
    fig.add_artist(FancyArrowPatch((x0,y0),(x1,y1),transform=fig.transFigure,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,connectionstyle=cs,zorder=6))
def clean(ax,grid=True):
    for s in ("top","right"):ax.spines[s].set_visible(False)
    if grid:ax.grid(True,color=GRIDC,lw=.5);ax.set_axisbelow(True)
def header(fig,k,title,sub,tag,tagc,h=FIG_H):
    y=1-.045*FIG_H/h;fig.text(.03,y,f"Fig. {k}",fontsize=12,weight="bold",color=MUTED,va="center")
    tg=fig.text(.97,y,tag,fontsize=8.4,weight="bold",color=tagc,va="center",ha="right",bbox=dict(boxstyle="square,pad=0.35",fc="white",ec=tagc,lw=1.0))
    fig.canvas.draw();tw=tg.get_window_extent(renderer=fig.canvas.get_renderer()).width/(fig.get_figwidth()*fig.dpi)
    fit_line(fig,.078,y,title,.97-tw-.02-.078,16.5,11,weight="bold",color=INK,va="center")
    fit_line(fig,.03,y-.038*FIG_H/h,sub,.94,9.2,6.5,color=INK2,va="center")
    fig.add_artist(Line2D([.03,.97],[y-.06*FIG_H/h,y-.06*FIG_H/h],transform=fig.transFigure,color=INK2,lw=.8))
def footer(fig,P,k,caption,h=FIG_H):
    s=FIG_H/h;st=P["startup"]
    fig.add_artist(Line2D([.03,.97],[.132*s,.132*s],transform=fig.transFigure,color=RULE,lw=.6))
    fit_text(fig,.03,.045*s,.94,.082*s,caption,fs_max=8.4,fs_min=6.2,color=INK2)
    fig.add_artist(Line2D([.03,.97],[.038*s,.038*s],transform=fig.transFigure,color=RULE,lw=.6))
    fig.text(.5,.025*s,mt(f"{APP_SHORT} · {AUTHOR} · {ORG} · {LICENSE_NAME} · {REPO}"),ha="center",va="center",fontsize=6.6,color=INK2)
    fig.text(.5,.010*s,mt(f"run {P['run_id']} · {P['started_local']} · {st.get('model_id')} · v1.4 engine SHA-256 {str(st.get('v14_sha256'))[:12]}… · TEST664 {str(st.get('source_sha256'))[:12]}… · Fig. {k}/7 · {COPYRIGHT}"),
             ha="center",va="center",fontsize=6.1,color=MUTED,family=MONO)
    if P.get("mock"):
        fig.text(.5,.5,"LAYOUT TEST — SYNTHETIC VALUES — NOT A MODEL RUN",fontsize=40,color="#DC2626",alpha=.18,ha="center",va="center",rotation=18,weight="bold",zorder=50)
def save_jpeg(fig,path,P,k,desc):
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=OUT_DPI,facecolor="white");plt.close(fig);buf.seek(0)
    with Image.open(buf) as im:rgb=im.convert("RGB")
    ex=Image.Exif();ex[0x010E]=ascii_(f"Fig. {k} {desc} | {APP_SHORT} | run {P['run_id']}");ex[0x013B]=ascii_(f"{AUTHOR} / {ORG}");ex[0x8298]=ascii_(f"{COPYRIGHT} - {LICENSE_NAME} - {REPO}")
    rgb.save(path,"JPEG",quality=94,optimize=True,subsampling=0,dpi=(OUT_DPI,OUT_DPI),exif=ex.tobytes())
    with Image.open(path) as c_:
        if c_.format!="JPEG" or c_.size[0]<3000:raise RuntimeError(f"JPEG check failed for {path}")
    return str(path)
def chip(fig,x,y,status,w=.075,h=.024,fs=6.6):
    ec,fc=STATUS_COL.get(status,(MUTED,"#F3F4F6"))
    fig.add_artist(Rectangle((x,y),w,h,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=.8,zorder=4))
    fig.text(x+w/2,y+h/2,STATUS_SHORT.get(status,status),fontsize=fs,color=ec,weight="bold",ha="center",va="center",zorder=5)
def fmt_bytes(b):return f"{b/2**20:.2f} MiB" if b>=2**20 else f"{b/1024:.1f} KiB"
def build(P,scn):return P["builds"][scn]
def prim(P,scn):return P["primary"][scn]
def resp(r,qid):
    for q in r["responses"]:
        if q["question_id"]==qid:return q
    return None
def arch_line(st):a=st.get("arch") or list(ARCH_EXPECTED);return f"{a[0]} layers · hidden {a[1]} · {a[2]} Q / {a[3]} KV heads · head dim {a[4]}"

# ------------------------------------------------------------------ FIG 1 · architecture
def fig1(P,k,path):
    st=P["startup"];b=build(P,"AKBASCORE");r=prim(P,"AKBASCORE");fig=new_fig();T=b["memory"]["T"];Pn=b["prefix_tokens"]
    header(fig,k,"Silicon Hippocampus architecture — from source fact to source-free answer",
           f"{st.get('model_id')} · {arch_line(st)} · {st.get('dtype')} · Demonstration A (AKBASCORE), condition ABC, DC_CUT3 · all numbers measured in this run",
           "ARCHITECTURE · VALUES FROM THIS RUN",INK2)
    W=.218;G=.02;x0=.03;ytop=.555;H1=.30
    cars=b["cartridges"];steps=b["steps"]
    src="\n".join(f"{c['slot']}: {c['text']}" for c in cars)
    block(fig,x0,ytop,W,H1,"1 · SOURCE FACTS","Read once per memory build",src,ec=INK2,mono=False,bfs=7.6)
    wr=[]
    for c,s in zip(cars,steps):
        fw=s["forwards"];ids=[f for f in fw if f["kind"]=="ids"]
        wr.append(f"{c['slot']}: 1 token-id forward · {ids[0]['n'] if ids else '?'} ids · no cache\n   H3 {c['rows_in_memory']}×{st['arch'][1] if st.get('arch') else 3584} · K/V L0–3 {c['rows_in_memory']} rows")
    block(fig,x0+W+G,ytop,W,H1,"2 · INDEPENDENT CARTRIDGE WRITING","One frozen forward per fact","\n".join(wr)+f"\n(each forward reads the {Pn}-token instruction prefix + one fact; only the first cartridge keeps the prefix rows)",ec=BLUE,hc=BLUE)
    sz="\n".join(f"{c['slot']}: {c['h3_bytes']:,} B H3 + {c['kv_lower_bytes']:,} B K/V = {fmt_bytes(c['cartridge_bytes'])}" for c in cars)
    block(fig,x0+2*(W+G),ytop,W,H1,"3 · NUMERICAL MEMORY REPRESENTATION","A cartridge holds numbers, not text",
          f"H3 = residual stream after layer 3 (one 3584-d vector per token)\nK/V L0–3 = attention keys/values of layers 0–3 [1, 4, rows, 128] bf16\n{sz}",ec=TEAL,hc=TEAL)
    cons=[]
    for s in steps:
        if s["kind"]=="init":cons.append(f"init({s['slot']}): {s['new_len']} rows ({1000*s['seconds']:.0f} ms)")
        else:
            e=[f for f in s["forwards"] if f["kind"]=="embeds"]
            cons.append(f"append {s['slot']}: zero-emb {e[0]['n'] if e else '?'} rows · cache {e[0]['past_len'] if e else '?'} · prior rows {'unchanged ✓' if s['prior_rows_bitwise_unchanged'] else 'CHANGED ✗'}")
    block(fig,x0+3*(W+G),ytop,W,H1,"4 · MEMORY CONSOLIDATION (APPEND-ONLY)",f"Layers 4–27 computed for new rows only",
          "\n".join(cons)+f"\nmemory {T} rows × {b['memory']['layers']} layers = {fmt_bytes(b['memory']['bytes'])}\nstep-wise build = native_memory659(): {b['engine_path_identical']}",ec=VIOLET,hc=VIOLET)
    for i in range(3):arrow(fig,x0+(i+1)*W+i*G+.002,ytop+H1/2,x0+(i+1)*(W+G)-.002,ytop+H1/2)
    yb=.165;H2=.33
    block(fig,x0+3*(W+G),yb,W,H2,"5 · NATIVE CUT3 INJECTION","Where the cartridge enters the model",
          "In each consolidation forward the input embeddings are all zero and the engine replaces the layer-3 output with the stored H3 "
          "(engine forward hook, removed after the call). Layers 4–27 then compute K/V for the new rows while attending to the existing memory.\n"
          f"K/V of layers 0–3 are re-phased (RoPE) to their memory positions and written with the native CUDA kernel append_kv (BF16 row copy, bitwise = torch.cat).\n"
          f"layers 0–3 rows = cartridge rows: {all(s['lower_rows_equal_cartridge'] for s in steps)}",ec=ORANGE,mono=False,bfs=7.6,hc=ORANGE)
    fz=P["verify"].get("frozen_after") or {}
    block(fig,x0+2*(W+G),yb,W,H2,"6 · FROZEN TRANSFORMER COMPUTATION","The memory is the attention cache",
          f"{st.get('model_id')}\n{arch_line(st)}\n{st.get('dtype')} · attention {st.get('attn')}\ntrainable tensors {fz.get('trainable_tensors')}\n"
          f"sentinel before = after: {P['verify'].get('sentinel_unchanged')}\nall-parameter guard unchanged: {P['verify'].get('guard_unchanged')}\n"
          f"full-weight SHA-256 = reference: {st.get('full_weight_equals_reference')}",ec=INK2,hc=INK2)
    fq=resp(r,"FINAL");qv=b["query_views"].get("FINAL",{})
    block(fig,x0+(W+G),yb,W,H2,"7 · SOURCE-FREE QUESTION ANSWERING","Only the question template is input",
          f"FINAL question: {fq['question'] if fq else '—'}\nmodel input: {fq.get('query_tokens') if fq else '?'} template tokens over {r['memory_rows']} memory rows\n"
          f"full cartridge token sequence in the input: {fq.get('cartridge_sequence_in_query') if fq else '?'}\nraw answer: '{fq['answer'] if fq else ''}' · {STATUS_SHORT.get(fq['status'],'') if fq else ''}",ec=BLUE,mono=False,bfs=7.8,hc=BLUE)
    frame(fig,x0,yb,W,H2,ec=RULE,fc=PANEL)
    fit_text(fig,x0+.008,yb+.01,W-.016,H2-.02,"Two phases\n\nWRITING (stages 1–2): every time a memory is built, the engine reads each fact once (one frozen forward per fact) and keeps only numbers.\n\n"
             "USING (stages 5–7): questions are answered over the numerical memory; the fact tokens are not part of the question input (checked per question).\n\n"
             "Stages 3–4 form and consolidate the memory. TEXT_JOINT (facts read together as text) is a comparison control; DC_CUT3 is the demonstrated path.",fs_max=8.4,fs_min=6.4,color=INK2)
    arrow(fig,x0+3*(W+G)+W/2,ytop-.004,x0+3*(W+G)+W/2,yb+H2+.004)
    arrow(fig,x0+3*(W+G)-.002,yb+H2/2,x0+3*W+2*G+.002,yb+H2/2);arrow(fig,x0+2*(W+G)-.002,yb+H2/2,x0+2*W+G+.002,yb+H2/2)
    footer(fig,P,k,f"Figure {k}. Executed path of {st.get('model_id')} for Demonstration A (fictional domain). (1–2) Each fact is written once as an independent cartridge: the layer-3 residual H3 and the "
           f"layer 0–3 K/V of its tokens. (3) The cartridge is numbers only. (4) Cartridges are appended in order; for each append the engine runs a forward with all-zero input embeddings, "
           f"injects the stored H3 at the layer-3 output and computes layers 4–27 for the new rows over the existing memory; earlier rows are compared bitwise after every append. "
           f"(5–7) The frozen model answers from the {T}-row memory with only the question template as input.")
    return save_jpeg(fig,path,P,k,"Architecture")

# ------------------------------------------------------------------ FIG 2 · cartridge creation and consolidation
def fig2(P,k,path):
    st=P["startup"];fig=new_fig()
    header(fig,k,"Cartridge creation and append-only consolidation, measured","Both demonstrations, condition ABC, DC_CUT3 · every forward recorded by a read-only pre-hook · bitwise checks of earlier rows after each append",
           "LIVE MEASUREMENT",BLUE)
    for n_,scn in enumerate(("AKBASCORE","MATRIX")):
        b=build(P,scn);steps=b["steps"];T=b["memory"]["T"]
        ax=fig.add_axes([.055,.565-n_*.30,.30,.215]);clean(ax)
        for i,s in enumerate(steps):
            old=s["old_len"];add=s["new_len"]-old;col=SEGC.get(s["slot"],BLUE)
            ax.bar(i,old,color="#E5E7EB",edgecolor=INK2,lw=.5,width=.6);ax.bar(i,add,bottom=old,color=col,width=.6)
            ax.text(i,s["new_len"]+T*.02,f"{s['new_len']}",ha="center",fontsize=8,color=INK)
            if s["kind"]=="append":ax.text(i,max(old/2,1),"✓" if s["prior_rows_bitwise_unchanged"] else "✗",ha="center",va="center",fontsize=13,color=TEAL if s["prior_rows_bitwise_unchanged"] else RED,weight="bold")
        ax.set_xticks(range(len(steps)));ax.set_xticklabels([f"step {i+1}\n{s['kind']} {s['slot']}" for i,s in enumerate(steps)],fontsize=7.6)
        ax.set_ylim(0,T*1.18);ax.set_ylabel("memory rows",fontsize=8);ax.set_title(f"{SCENARIO_LABEL[scn].split(' — ')[0]} ({scn}) · memory rows after each step",fontsize=8.4)
    fig.text(.055,.17,"grey = earlier rows · ✓ bitwise unchanged · colour = rows added",fontsize=7.4,color=INK2)
    tx=.40;fig.text(tx,.84,"Per-cartridge record",fontsize=10,weight="bold",color=INK)
    hdr=f"{'scenario':<10}{'slot':<5}{'body':>5}{'rows':>6}{'ckpt H3':>17}{'K/V L0–3':>15}{'bytes':>11}{'write':>9}{'consolidation':>24}{'ms':>7}{'prior':>7}{'L0–3=cart.':>11}"
    fig.text(tx,.81,mt(hdr),fontsize=6.9,family=MONO,color=MUTED)
    y=.785
    for scn in ("AKBASCORE","MATRIX"):
        b=build(P,scn)
        for c,s in zip(b["cartridges"],b["steps"]):
            ids=[f for f in s["forwards"] if f["kind"]=="ids"];emb=[f for f in s["forwards"] if f["kind"]=="embeds"]
            cons=(f"zero-emb {emb[0]['n']}·cache {emb[0]['past_len'] if emb[0]['has_past'] else 0}" if emb else "—")
            pr="init" if s["kind"]=="init" else ("✓" if s["prior_rows_bitwise_unchanged"] else "✗")
            line=(f"{scn[:9]:<10}{c['slot']:<5}{c['body_tokens']:>5}{c['rows_in_memory']:>6}{str(tuple(c['h3_shape'])):>17}{str(tuple(c['kv_lower_shape'])):>15}"
                  f"{c['cartridge_bytes']:>11,}{('ids '+str(ids[0]['n'])) if ids else '—':>9}{cons:>24}{1000*s['seconds']:>7.1f}{pr:>7}{('✓' if s['lower_rows_equal_cartridge'] else '✗'):>11}")
            fig.text(tx,y,mt(line),fontsize=6.9,family=MONO,color=INK);y-=.03
        y-=.012
    frame(fig,tx,.165,.57,.235,ec=TEAL,fc=TEAL_L,lw=1.0)
    a=build(P,"AKBASCORE");m=build(P,"MATRIX")
    def summ(b):return (f"{b['source_reads']} source token-id forwards (one per cartridge) · {b['zero_embed_forwards']} zero-embedding consolidation forwards · "
                        f"prior rows unchanged at every append: {all(s['prior_rows_bitwise_unchanged'] for s in b['steps'][1:])} · step-wise memory = native_memory659(): {b['engine_path_identical']} · "
                        f"{b['memory']['T']} rows · {fmt_bytes(b['memory']['bytes'])} · build {1000*b['build_seconds']:.0f} ms · memory SHA-256 {b['memory']['sha256'][:16]}…")
    fit_text(fig,tx+.008,.172,.554,.22,"Measured in this run\n• AKBASCORE: "+summ(a)+"\n• MATRIX: "+summ(m)+
             "\n• 'L0–3=cart.': the layer 0–3 K/V rows inside the memory equal the cartridge's own K/V (keys re-phased to their memory position), compared bitwise."
             "\n• 'ckpt H3' / 'K/V L0–3': tensor shapes of the cartridge checkpoint (prefix + fact); 'rows' = rows the cartridge adds to the memory (the first one carries the prefix)."
             "\n• bytes = H3 rows × 3584 × 2 B + K/V rows × 4 layers × 2 × 4 heads × 128 × 2 B, from the measured row counts.",fs_max=8.0,fs_min=6.0,color=INK)
    footer(fig,P,k,f"Figure {k}. Writing and consolidation of the three cartridges of each demonstration. The first step (init) writes cartridge A together with the {a['prefix_tokens']}-token instruction prefix; "
           "every further step (append) runs one token-id forward over the instruction prefix plus the new fact only (no cache, no earlier fact) and one forward whose input embeddings are all zero, whose length equals the new rows and which "
           "attends to the existing memory. After each append all earlier K/V rows of all layers are compared bitwise with the memory before the append; the final step-wise memory is compared bitwise with "
           "the memory produced by the engine's native_memory659(), the path used by run_demo.")
    return save_jpeg(fig,path,P,k,"Cartridge consolidation")

# ------------------------------------------------------------------ FIG 3 · internal memory X-ray
def fig3(P,k,path):
    b=build(P,"AKBASCORE");M=b["memory"];T=M["T"];segs=b["segments"];fig=new_fig();NLm=M["layers"]
    header(fig,k,"Internal neural memory X-ray",f"Demonstration A · ABC · DC_CUT3 · tensor per layer {tuple(M['shape'])} ({M['dtype']}) · {NLm} layers × {T} rows · K+V {fmt_bytes(M['bytes'])} ({M['bytes']:,} B) · read from this run's memory",
           "LIVE MEASUREMENT",BLUE)
    for n_,(key,ttl) in enumerate((("k_norm","‖K‖ per row (mean over 4 KV heads)"),("v_norm","‖V‖ per row (mean over 4 KV heads)"))):
        A=np.asarray(M[key],dtype=float);mu=A.mean(axis=1,keepdims=True);sd=A.std(axis=1,keepdims=True);sd[sd==0]=1;Z=(A-mu)/sd
        ax=fig.add_axes([.055,.575-n_*.285,.60,.225])
        im=ax.imshow(Z,aspect="auto",cmap="RdBu_r",vmin=-2.5,vmax=2.5,interpolation="nearest",origin="lower")
        for lab,a_,b_,_ in segs[1:]:ax.axvline(a_-.5,color=INK,lw=.8)
        ax.axhline(CUT+.5,color=AMBER,lw=1.4,ls=(0,(4,2)))
        ax.set_yticks([0,CUT,NLm//2,NLm-1]);ax.set_yticklabels(["L0",f"L{CUT}",f"L{NLm//2}",f"L{NLm-1}"],fontsize=7)
        ax.set_title(mt(ttl+" · z-scored within each layer (display normalisation)"),fontsize=8.3);ax.set_xlim(-.5,T-.5);ax.set_xticks([])
        cb=fig.colorbar(im,cax=fig.add_axes([.66,.575-n_*.285,.007,.225]));cb.ax.tick_params(labelsize=6.5);cb.outline.set_linewidth(.5);cb.set_label("z",fontsize=7)
    fig.text(.055,.198,mt(f"dashed line: layers 0–{CUT} below = cartridge rows (written per fact) · layers {CUT+1}–{NLm-1} above = computed during consolidation"),fontsize=7.2,color=AMBER,ha="left",va="top",weight="bold")
    sx=fig.add_axes([.055,.24,.60,.03]);sx.set_xlim(-.5,T-.5);sx.set_ylim(0,1);sx.axis("off")
    for lab,a_,b_,_ in segs:
        sx.add_patch(Rectangle((a_-.5,0),b_-a_,1,color=SEGC.get(lab,"#64748B"),lw=0));sx.text((a_+b_-1)/2,.5,lab if lab!="prefix" else f"prefix ({b_-a_})",ha="center",va="center",fontsize=7.4,color="white",weight="bold")
    fig.text(.055,.228,mt("row layout: "+" · ".join(f"{lab} rows {a_}–{b_-1}" for lab,a_,b_,_ in segs)),fontsize=7.2,family=MONO,color=INK2,va="top")
    jd=b["joint_difference"];ax=fig.add_axes([.735,.40,.235,.38]);clean(ax)
    L=np.arange(NLm)
    for j,lab in enumerate(jd["labels"]):
        ys=np.asarray([row[j] for row in jd["k"]]);ax.plot(L,ys,color=SEGC.get(lab,"#64748B"),lw=1.4,label=f"{lab} (K)")
        ys=np.asarray([row[j] for row in jd["v"]]);ax.plot(L,ys,color=SEGC.get(lab,"#64748B"),lw=1.0,ls="--")
    ax.axvline(CUT+.5,color=AMBER,lw=1,ls=(0,(4,2)));ax.set_xlabel("layer",fontsize=8);ax.set_ylabel("‖DC − TEXT_JOINT‖ / ‖TEXT_JOINT‖",fontsize=7.6)
    ax.set_title("DC_CUT3 memory vs TEXT_JOINT control",fontsize=8.3);ax.legend(fontsize=6.6,loc="upper right",ncol=2)
    eq=", ".join(f"{l}: {'bitwise equal' if e else 'differs'}" for l,e in zip(jd["labels"],jd["bitwise_equal_segments"]))
    fit_text(fig,.735,.165,.235,.17,f"solid = K, dashed = V, per memory segment.\n{eq}.\nA cartridge written alone differs from the same fact read together with the earlier facts; this curve measures that difference layer by layer.",fs_max=7.6,fs_min=6.0,color=INK2)
    footer(fig,P,k,f"Figure {k}. The consolidated memory of Demonstration A after the third cartridge. Each column is one token row, each heat-map row one layer; the colour is the row's mean K (top) or V (bottom) "
           f"L2 norm, z-scored within each layer for display only. Vertical lines mark the prefix and cartridges A, B, C. Rows of layers 0–{CUT} are the cartridges' own K/V; layers {CUT+1}–{NLm-1} were computed by the engine during "
           "consolidation. Right: relative difference between this memory and the TEXT_JOINT control memory built from the same facts as one text, per layer and segment. Values are read from GPU memory of this run.")
    return save_jpeg(fig,path,P,k,"Internal memory X-ray")

# ------------------------------------------------------------------ FIG 4 / FIG 5 · experiments
MAXNEW_TXT=16
def qa_table(fig,r,x,y,w,rowh,fs=7.6,show_tokens=True):
    for q in r["responses"]:
        ok=q.get("status","UNSCORED");tw=w-.175
        fig.text(x,y,q["question_id"],fontsize=7.2,weight="bold",color=MUTED,va="top")
        fit_text(fig,x+.075,y-rowh*.5,tw,rowh*.5-.002,q["question"],fs_max=fs,fs_min=5.8,color=INK2)
        fit_text(fig,x+.075,y-rowh+.008,tw,rowh*.5-.012,"→ '"+q["answer"]+"'",fs_max=fs+1.0,fs_min=6.0,color=INK,family=MONO,weight="bold")
        chip(fig,x+w-.088,y-.026,ok,w=.085)
        if show_tokens and q.get("generated_tokens") is not None:
            fit_line(fig,x+w-.088,y-.036,f"{q.get('generated_tokens')} new tokens",.088,6.2,5.0,color=MUTED,va="top")
            if q.get("limit_reached"):fit_line(fig,x+w-.088,y-.052,f"{MAXNEW_TXT}-token limit reached",.088,6.2,5.0,color=RED,va="top",weight="bold")
        fig.add_artist(Line2D([x,x+w],[y-rowh+.003,y-rowh+.003],transform=fig.transFigure,color=GRIDC,lw=.5))
        y-=rowh
    return y
def chain_column(fig,r,sc,x,y,w,h,start):
    staged=r["staged"];n=len(staged["trace"]);bh=min(.085,(h-.02)/(n+1)-.022);yy=y
    frame(fig,x,yy-bh,w,bh,ec=INK2,fc=PANEL);fig.text(x+w/2,yy-bh/2,mt(start),ha="center",va="center",fontsize=9.6,weight="bold",color=INK)
    for t in staged["trace"]:
        yy2=yy-bh-.03;arrow(fig,x+w/2,yy-bh,x+w/2,yy2+.002,color=INK2)
        ec,fc=STATUS_COL.get(t.get("status","UNSCORED"));frame(fig,x,yy2-bh,w,bh,ec=ec,fc=fc,lw=1.1)
        fig.text(x+.006,yy2-.012,f"HOP {t['hop']} · {STATUS_SHORT.get(t.get('status'),'')}",fontsize=6.8,weight="bold",color=ec,va="center")
        fig.text(x+w/2,yy2-bh/2-.006,mt(short("'"+t["answer"]+"'",34)),ha="center",va="center",fontsize=9.6,weight="bold",color=INK,family=MONO)
        fit_text(fig,x+.006,yy2-bh+.003,w-.012,.026,short(t["question"],105),fs_max=6.2,fs_min=5.0,color=INK2)
        yy=yy2
    return yy-bh
def experiment_fig(P,k,path,scn):
    global MAXNEW_TXT
    st=P["startup"];MAXNEW_TXT=st.get("max_new");r=prim(P,scn);sc=P["scenarios"][scn];b=build(P,scn);fig=new_fig();spec=SCORING[scn]
    tag="DEMONSTRATION A" if scn=="AKBASCORE" else "DEMONSTRATION B"
    title=("Mustafa Akbaş experiment — three cartridges, direct questions and a staged three-step chain" if scn=="AKBASCORE"
           else "The Matrix / Neo experiment — what goes into memory and what the frozen model answers")
    header(fig,k,title,f"{SCENARIO_LABEL[scn]} · fictional knowledge domain · condition ABC · DC_CUT3 · raw model outputs, unedited · scored: {sc['scored']}",tag,BLUE)
    # cartridges
    fig.text(.03,.835,"CARTRIDGE TEXTS (source facts, read once per memory build)",fontsize=7.8,weight="bold",color=INK2)
    yy=.81;ch=.135
    for c in b["cartridges"]:
        col=SEGC.get(c["slot"],BLUE);frame(fig,.03,yy-ch,.27,ch,ec=col,fc="white",lw=1.1)
        fig.text(.036,yy-.014,mt(f"{SLOT_LABEL[scn][c['slot']]} · {c['body_tokens']} tokens · {fmt_bytes(c['cartridge_bytes'])} numerical"),fontsize=7.2,weight="bold",color=col,va="center")
        fit_text(fig,.036,yy-ch+.005,.258,ch-.03,c["text"],fs_max=8.4,fs_min=6.2,color=INK);yy-=ch+.012
    fit_text(fig,.03,.165,.27,.05,f"Counterfactual C (only in ABC_CF, see Fig. 6): {sc['texts']['C_CF']}",fs_max=7.0,fs_min=5.8,color=MUTED)
    # direct questions
    x=.32;w=.385;fig.text(x,.835,"DIRECT QUESTIONS — one question each, answered over the memory",fontsize=7.8,weight="bold",color=INK2)
    n=len(r["responses"]);rowh=min(.13,.645/n);qa_table(fig,r,x,.81,w,rowh)
    # chain
    cx=.725;cw=.245;fig.text(cx,.835,"STAGED CHAIN — predefined external question sequence",fontsize=7.8,weight="bold",color=INK2)
    start=spec["chain_text"][0];endy=chain_column(fig,r,spec,cx,.81,cw,.50,start)
    s_=r["staged"];chain_st=s_.get("chain_status","UNSCORED")
    fit_line(fig,cx,endy-.022,f"chain result '{s_['answer']}' · {STATUS_SHORT.get(chain_st)} · completed {s_['completed']}",cw,7.6,5.5,weight="bold",color=STATUS_COL[chain_st][0])
    fit_text(fig,cx,.165,cw,max(.02,endy-.04-.165),"written chain (cartridge texts): "+" → ".join(spec["chain_text"]),fs_max=7.2,fs_min=5.8,color=MUTED)
    if scn=="AKBASCORE":
        cap=(f"Figure {k}. Demonstration A, a fictional domain (no real-world claim is made, in particular none about time). Left: the three cartridge texts, each written once. Middle: direct questions with the "
             f"raw answers of the frozen model ({st.get('max_new')} new tokens at most) and their audit status (expected answers are used only after generation). Right: the staged chain: three predefined questions "
             "asked in sequence, each one inserting the previous raw answer; every step is answered by the model from the numerical memory. This is an externally specified sequence, not autonomous planning.")
    else:
        cap=(f"Figure {k}. Demonstration B, a fictional scenario. What was written into memory (left), what the frozen model answered to direct questions (middle) and the staged three-step chain (right). "
             f"The cartridges enter the model as numbers: H3 at the layer-3 output and K/V of layers 0–3; the model weights are not changed (sentinel before = after: {P['verify'].get('sentinel_unchanged')}). "
             "Note: cartridge C itself names MX-71 and the Kung Fu knowledge, so the FINAL question is supported by A and C without B (see Fig. 6).")
    footer(fig,P,k,cap)
    return save_jpeg(fig,path,P,k,"Experiment "+scn)
def fig4(P,k,path):return experiment_fig(P,k,path,"AKBASCORE")
def fig5(P,k,path):return experiment_fig(P,k,path,"MATRIX")

# ------------------------------------------------------------------ FIG 6 · causal controls
def fig6(P,k,path):
    FH=8.6;fig=new_fig(FH);s=FH/FIG_H
    header(fig,k,"Causal controls and the staged multi-hop chain","Nine memory conditions per demonstration · DC_CUT3 grid (cell = raw answer, colour = audit status) · TEXT_JOINT control summarised on the right · TEST677 shown separately as a sealed record",
           "CAUSAL CONTROLS",ORANGE,h=FH)
    def U(y):return y/s if False else y
    gx=.03;gw=.62;top=.865
    for n_,scn in enumerate(("AKBASCORE","MATRIX")):
        cm=controls_matrix(P,scn,"DC_CUT3");cols=[q for q,_ in P["scenarios"][scn]["queries"] if q!="SYNTHESIS"]+["CHAIN"]
        y0=top-n_*.37;fig.text(gx,y0,mt(f"{SCENARIO_LABEL[scn]} · DC_CUT3 · scored: {P['scenarios'][scn]['scored']}"),fontsize=8.6,weight="bold",color=INK)
        cw=(gw-.07)/len(cols);rh=.03
        for j,c in enumerate(cols):fig.text(gx+.07+j*cw+cw/2,y0-.022,c,fontsize=6.8,weight="bold",color=MUTED,ha="center",va="center")
        for i,cond in enumerate(CONDITIONS):
            yy=y0-.04-(i+1)*rh;fig.text(gx,yy+rh/2,cond,fontsize=7.2,weight="bold",color=INK2,va="center")
            r=cm.get(cond)
            for j,c in enumerate(cols):
                if r is None:st_,txt="UNSCORED","NOT MEASURED"
                elif c=="CHAIN":st_=r["staged"]["chain_status"] if r["staged"] else "UNSCORED";txt=r["staged"]["answer"] if r["staged"] else "—"
                else:
                    q=resp(r,c);st_=q["status"] if q else "UNSCORED";txt=q["answer"] if q else "—"
                ec,fc=STATUS_COL.get(st_);x=gx+.07+j*cw
                fig.add_artist(Rectangle((x+.001,yy+.002),cw-.002,rh-.004,transform=fig.transFigure,facecolor=fc,edgecolor=ec,lw=.6))
                fig.text(x+cw/2,yy+rh/2,mt(short(txt,int(cw*190))),fontsize=6.0,color=INK,ha="center",va="center",family=MONO)
    lx=.67;lw_=.30;ABBR=dict(CORRECT_KNOWN="CK",CORRECT_ABSTENTION="CA",UNSUPPORTED="US",MISSED_KNOWN="MK",WRONG_KNOWN="WK",UNSCORED="UN")
    fig.text(lx,.865,"Status counts (all questions + chain, nine conditions)",fontsize=8.4,weight="bold",color=INK)
    y=.84
    for rep in ("DC_CUT3","TEXT_JOINT"):
        sc=status_counts(P,rep)
        for scn in ("AKBASCORE","MATRIX"):
            c=sc.get(scn,{});tot=sum(c.values())
            line=f"{rep:<10} {scn:<9} "+("NOT MEASURED" if tot==0 else " · ".join(f"{ABBR[s_]} {c.get(s_,0)}" for s_ in STATUS_ORDER if c.get(s_,0)))
            fit_line(fig,lx,y-.012,line,lw_,6.9,5.0,family=MONO,color=INK2,va="center");y-=.025
    # agreement DC vs TEXT
    agree=tot_=0
    for scn in ("AKBASCORE","MATRIX"):
        a=controls_matrix(P,scn,"DC_CUT3");b=controls_matrix(P,scn,"TEXT_JOINT")
        for cond in CONDITIONS:
            if cond in a and cond in b:
                for q in a[cond]["responses"]:
                    qb=resp(b[cond],q["question_id"])
                    if qb:tot_+=1;agree+=int(q["answer"].strip()==qb["answer"].strip())
    fit_line(fig,lx,y-.01,"CK correct · CA correct UNKNOWN · US unsupported · MK missed · WK wrong · UN unscored",lw_,6.6,5.0,color=MUTED);y-=.025
    fig.text(lx,y-.006,mt(f"DC_CUT3 vs TEXT_JOINT identical raw answers: {agree}/{tot_}" if tot_ else "TEXT_JOINT control: NOT MEASURED"),fontsize=7.2,color=INK2);y-=.04
    # counterfactual effect
    for scn in ("AKBASCORE","MATRIX"):
        cm=controls_matrix(P,scn,"DC_CUT3");abc=cm.get("ABC");cf=cm.get("ABC_CF")
        if abc and cf:
            fa=resp(abc,"FINAL");fc_=resp(cf,"FINAL")
            txt=f"{scn}: FINAL with ABC '{fa['answer'] if fa else '—'}' → with ABC_CF '{fc_['answer'] if fc_ else '—'}' · changed {bool(fa and fc_ and norm_ans(fa['answer'])!=norm_ans(fc_['answer']))}"
        else:txt=f"{scn}: counterfactual NOT MEASURED"
        fit_text(fig,lx,y-.04,lw_,.04,txt,fs_max=7.0,fs_min=5.8,color=INK);y-=.045
    # leakage signature
    leak=[]
    for r in P["controls"]:
        if r["representation"]!="DC_CUT3":continue
        for q in r["responses"]:
            if q["status"]=="UNSUPPORTED" and q.get("equals_full_memory_answer"):leak.append(f"{r['scenario'][:4]} {r['condition']} {q['question_id']}")
    fit_text(fig,lx,y-.07,lw_,.07,"Unsupported answers equal to the full-memory answer (a cartridge needed for the question was absent): "+(", ".join(leak) if leak else "none"),fs_max=6.8,fs_min=5.6,color=ORANGE);y-=.08
    # TEST677 sealed
    frame(fig,lx,.165,lw_,y-.175,ec=INK2,fc=PANEL,ls="--")
    t=TEST677_REF;lv=P["references"].get("test677_log") or {}
    ref="\n".join(f"{c:<7}{x}/{y_}/{z}  AB '{a}' · BC '{b_}' · ABC '{c_}'" for c,a,b_,c_,x,y_,z in t["user_panel_dc"])
    vtxt={True:"yes",False:"NO — mismatch",None:"not checked (offline)"}.get(lv.get("matches"),str(lv.get("matches")))
    fit_text(fig,lx+.006,.17,lw_-.012,y-.19,f"SEALED RECORD · TEST677 (not run here) · 677.log SHA-256 verified this session: {vtxt}\nDC_CUT3 user panel (same A/B/C texts as Demonstration A; its counterfactual was a random AltEffect token)\n"
             f"cond   status    CHAIN answers\n{ref}\nCK correct · CA correct UNKNOWN · US unsupported\nblind panels CHAIN_ABC DC_CUT3: "+", ".join(f"{c} {v}" for c,v in t["blind_chain_abc_dc"]),fs_max=6.6,fs_min=5.2,family=MONO,color=INK2)
    lg=fig.add_axes([.03,.142,.62,.02]);lg.axis("off")
    for i,s_ in enumerate(STATUS_ORDER):
        ec,fc=STATUS_COL[s_];lg.add_patch(Rectangle((i/6+.005,0.1),.03,.8,facecolor=fc,edgecolor=ec,transform=lg.transAxes));lg.text(i/6+.04,.5,STATUS_SHORT[s_],fontsize=7,va="center",transform=lg.transAxes)
    footer(fig,P,k,f"Figure {k}. Every cell is the unedited answer of the frozen model with the memory built from the listed cartridges (NONE = instruction prefix only; ABC_CF = A, B and the counterfactual C). "
           "Status (TEST677 rule): correct = expected answer present in memory and produced; correct UNKNOWN = a required cartridge is absent and the model answers UNKNOWN; unsupported = a required cartridge is absent but the "
           "model still answers; missed = answer available but UNKNOWN; wrong = a different answer. CHAIN = result of the predefined three-step question sequence. TEST677 values come from its published log and were not recomputed.",h=FH)
    return save_jpeg(fig,path,P,k,"Causal controls")

# ------------------------------------------------------------------ FIG 7 · verification record
def fig7(P,k,path):
    st=P["startup"];v=P["verify"];fig=new_fig()
    header(fig,k,"Verification and research record","Configuration, engine identity, integrity measurements, file hashes, references and the claim boundary of this run","VERIFICATION · RECORD",INK2)
    cw=.30;gap=.02;y=.18;h=.67
    cfg=(f"run              {P['run_id']}\nstarted          {P['started_local']}\nstarted (UTC)    {P['started_utc']}\nfinished (UTC)   {P.get('finished_utc')}\nduration         {P.get('elapsed_seconds')} s\n"
         f"model            {st.get('model_id')}\narchitecture     {arch_line(st)}\nweights          {st.get('dtype')} · frozen · {st.get('attn')}\n")
    if st.get("params"):cfg+=f"parameters       {st['params']:,}\n"
    cfg+=(f"GPU              {st.get('gpu')}\ntorch            {st.get('torch')}\ntransformers     {st.get('transformers')}\ngradio           {st.get('gradio')}\npython           {st.get('python')}\n"
          f"injection        native DC_CUT3: H3 at layer {st.get('cut')} + K/V L0–{st.get('cut')}\nprefix           {st.get('prefix_tokens')} tokens\nmax new tokens   {st.get('max_new')}\n"
          f"v1.4 engine      SHA-256\n  {st.get('v14_sha256')}\nTEST664 source   SHA-256\n  {st.get('source_sha256')}\nTEST664 panel    SHA-256\n  {st.get('panel_sha256')}\nTEST566 git blob {st.get('test566_blob')}")
    frame(fig,.03,y,cw,h,ec=RULE);fig.text(.04,y+h-.025,"CONFIGURATION AND ENGINE",fontsize=8.6,weight="bold",color=INK2)
    fit_text(fig,.04,y+.01,cw-.02,h-.05,cfg,fs_max=7.3,fs_min=5.6,family=MONO,color=INK)
    fz=v.get("frozen_after") or {}
    ver=(f"full-weight SHA-256 (start-up)\n  {st.get('full_weight_sha256')}\n  = TEST575 reference: {st.get('full_weight_equals_reference')}\n"
         f"sentinel load = before = after   {v.get('sentinel_unchanged')}\n  {str(v.get('sentinel_after'))[:40]}…\nall-parameter guard start = before = after   {v.get('guard_unchanged')}\n  {str(v.get('guard_after'))[:40]}…\n"
         f"trainable tensors   {fz.get('trainable_tensors')}\nLoRA                {fz.get('lora')}\nresidual hooks      {len(fz.get('residual_hooks',[]))}\nall engine runs frozen + sentinel   {v.get('all_runs_frozen')}\n\n")
    for scn in ("AKBASCORE","MATRIX"):
        b=P["builds"][scn]
        ver+=(f"{scn} memory (ABC, DC_CUT3)\n  {b['memory']['T']} rows · {b['memory']['bytes']:,} B\n  SHA-256 {b['memory']['sha256'][:32]}…\n"
              f"  prior rows unchanged: {all(s_['prior_rows_bitwise_unchanged'] for s_ in b['steps'][1:])} · L0–3 = cartridge: {all(s_['lower_rows_equal_cartridge'] for s_ in b['steps'])}\n"
              f"  step-wise = native_memory659(): {b['engine_path_identical']}\n  cartridges: "+", ".join(f"{c['slot']} {c['cartridge_bytes']:,} B" for c in b["cartridges"])+"\n")
    frame(fig,.03+cw+gap,y,cw,h,ec=RULE);fig.text(.04+cw+gap,y+h-.025,"VERIFICATION MEASURED IN THIS RUN",fontsize=8.6,weight="bold",color=INK2)
    fit_text(fig,.04+cw+gap,y+.01,cw-.02,h-.05,ver,fs_max=7.3,fs_min=5.6,family=MONO,color=INK)
    runs=[r for r in P["controls"] if r["condition"]=="ABC"]+([P["custom"]] if P["custom"] else [])
    rec="Engine run records (v1.4 run_sha256 / file SHA-256)\n"+"\n".join(f"{r['scenario'][:4]} {r['condition']} {'DC' if r['representation']=='DC_CUT3' else 'TEXT'}  {str(r['run_sha256'])[:20]}… / {str(r.get('engine_json_sha256'))[:12]}…" for r in runs)
    rec+=f"\n({len(P['controls'])} engine runs in total; all hashes in SHA256_MANIFEST.json)\n\nReferences\n• TEST664 {URL_664}\n• TEST677 {URL_677}\n  log SHA-256 {TEST677_LOG_SHA256[:24]}… verified now: {({True:'yes',False:'NO',None:'not checked'}).get((P['references'].get('test677_log') or {}).get('matches'),'?')}\n  result SHA-256 {TEST677_REF['result_sha256'][:24]}…\n"
    rec+=f"• earlier v1.4 Colab run {EARLIER_V14_LOG['date']} (reference only)\n\nScientific boundaries\n• fictional / synthetic scenarios\n• staged chain = predefined external question sequence, not autonomous planning\n• {st.get('max_new')}-token generation limit; free-text synthesis unscored\n• abstention with missing cartridges is measured, not guaranteed\n• hashes show file identity, not correctness\n• one model, two small scenarios; no scale or generalisation claim\n• {AUTHOR} · {ORG} · {PLACE}"
    frame(fig,.03+2*(cw+gap),y,cw,h,ec=RULE);fig.text(.04+2*(cw+gap),y+h-.025,"RECORDS · REFERENCES · BOUNDARIES",fontsize=8.6,weight="bold",color=INK2)
    fit_text(fig,.04+2*(cw+gap),y+.01,cw-.02,h-.05,rec,fs_max=7.3,fs_min=5.6,family=MONO,color=INK)
    footer(fig,P,k,f"Figure {k}. Identity card of run {P['run_id']}. The embedded v1.4 engine source and the downloaded TEST664 source are SHA-256 checked before execution; the model weights are checked with the TEST566 sentinel, "
           "an all-parameter guard and (at start-up) the TEST575 full-weight SHA-256. Memory integrity values come from the measured builds of Fig. 2–3. File hashes identify files; they are not evidence of correctness.")
    return save_jpeg(fig,path,P,k,"Verification")

FIGURES=[fig1,fig2,fig3,fig4,fig5,fig6,fig7]
def render_all(P,run_dir,cb=None):
    out=[]
    try:
        for i,(fn,name) in enumerate(zip(FIGURES,FIG_FILES),1):
            out.append(fn(P,i,Path(run_dir)/name))
            if cb:cb(i)
    finally:plt.close("all")
    return out


# =====================================================================================================================
# GRADIO INTERFACE — calls the engine, records, visualises. It never computes or edits an answer.
# =====================================================================================================================
def esc(x):return html.escape(str(x))
def card_html(title,body,kind="info"):return f'<div class="kz card {kind}"><div class="h">{esc(title)}</div><div class="small">{body}</div></div>'
def chip_html(status):
    ec,fc=STATUS_COL.get(status,(MUTED,"#F3F4F6"))
    return f'<span class="chip" style="color:{ec};background:{fc};border-color:{ec}">{esc(STATUS_SHORT.get(status,status))}</span>'

CSS="""
:root{--kz-on:#0f766e;--kz-fg:#111827;--kz-card:#ffffff;--kz-bd:#d1d5db;--kz-a:#1d4ed8;--kz-err:#b91c1c;--kz-warn:#b45309}
.dark{--kz-on:#2dd4bf;--kz-fg:#f3f4f6;--kz-card:#111827;--kz-bd:#4b5563;--kz-a:#60a5fa;--kz-err:#f87171;--kz-warn:#fbbf24}
html,body{overflow-x:hidden!important}
.gradio-container{width:100%!important;max-width:960px!important;margin:0 auto!important;padding-left:10px!important;padding-right:10px!important;box-sizing:border-box!important}
.gradio-container *{box-sizing:border-box}
.kz{color:var(--kz-fg)!important;max-width:100%;overflow-wrap:anywhere;line-height:1.45}
.kz *{color:inherit}
.hero{background:#0b1220;border-radius:6px;padding:18px 18px 14px;margin:4px 0 8px}
.hero,.hero *{color:#f9fafb!important}
.rec{font-size:11.5px;letter-spacing:1.6px;text-transform:uppercase;opacity:.85;border:1px solid rgba(255,255,255,.35);display:inline-block;padding:2px 8px;border-radius:3px}
.brand{font-size:clamp(20px,5.6vw,30px);font-weight:800;letter-spacing:.4px;line-height:1.15;margin-top:10px}
.title{font-size:clamp(14px,3.8vw,18px);font-weight:600;margin-top:6px;color:#fde68a!important}
.by{font-size:13.5px;margin-top:10px}
.msg{margin:8px 0 0;padding-left:18px;font-size:13.5px}
.lic{font-size:12px;opacity:.85;margin-top:8px}
.card{background:var(--kz-card);border:1px solid var(--kz-bd);border-left:4px solid var(--kz-bd);border-radius:4px;padding:10px 12px;margin:6px 0}
.card.on{border-left-color:var(--kz-on)}.card.err{border-left-color:var(--kz-err)}.card.warn{border-left-color:var(--kz-warn)}.card.info{border-left-color:var(--kz-a)}
.card .h{font-weight:700;font-size:15px;margin-bottom:4px}
.small{font-size:13.5px}
.mono,.kz code{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
.chip{display:inline-block;border:1px solid;border-radius:3px;padding:0 6px;font-size:12px;font-weight:700;white-space:nowrap}
.tbl{width:100%;border-collapse:collapse;font-size:13px}
.tbl td,.tbl th{border-bottom:1px solid var(--kz-bd);padding:5px 4px;vertical-align:top;text-align:left}
.tbl th{font-size:12px;opacity:.75}
.stg{font-size:13.5px;margin:2px 0}.stg.done{color:var(--kz-on)!important}.stg.run{color:var(--kz-a)!important;font-weight:700}.stg.wait{opacity:.55}.stg.fail{color:var(--kz-err)!important;font-weight:700}
#kz_full button,#kz_full{font-size:clamp(15px,4.2vw,18px)!important;font-weight:700!important;min-height:52px!important}
@media (max-width:640px){.tbl .hide-m{display:none}}
"""
HERO=(f'<div class="kz hero"><span class="rec">Interactive research demonstration · frozen model · live measurement</span>'
      f'<div class="brand">AKBASCORE AI TECHNOLOGY<br>MAM SILICON HIPPOCAMPUS v1.4</div><div class="title">{esc(APP_SUB)}</div>'
      '<ul class="msg">'
      '<li>When a memory is built, each fact is read once and written into a <b>numerical cartridge</b>: the layer-3 hidden state (H3) and the attention keys/values of layers 0–3.</li>'
      '<li>Cartridges are <b>consolidated</b> into the attention memory of the frozen Qwen2.5-7B-Instruct (native CUT3 injection, CUDA K/V copy). The weights are never changed.</li>'
      '<li>Questions are answered <b>without the source text</b>, from the memory alone. The three-step chain is a <b>predefined external question sequence</b>, not autonomous planning.</li>'
      '<li><b>DC_CUT3</b> is the demonstrated injection path. <b>TEXT_JOINT</b> (facts read together as text) is a comparison control.</li>'
      '<li>Both scenarios are <b>fictional</b>. Time does not reverse; the Matrix is a film. Answers are capped at <b>16 new tokens</b>; free-text synthesis can be truncated.</li></ul>'
      f'<div class="by"><b>{esc(AUTHOR)}</b> · {esc(ORG)} · {esc(PLACE)}</div>'
      f'<div class="lic">{esc(LICENSE_NAME)} · {esc(COPYRIGHT)} · <a href="{REPO}" target="_blank" rel="noopener">{esc(REPO)}</a></div></div>')
SCOPE_HTML=card_html("What this demo shows — and what it does not",
    "Shown: the unchanged v1.4/TEST664 engine writes, consolidates and reads numerical memory; every answer is the raw output of the frozen model; every figure value is "
    "measured in the run that produced it. Not shown: reliable abstention when cartridges are missing (Fig. 6 shows how often the model answers anyway), free-form synthesis "
    "beyond 16 tokens, autonomous multi-hop reasoning, generalisation beyond two small fictional scenarios. Two successful demonstrations do not mean the controls pass.","warn")

def engine_card_html():
    s=STARTUP;fz=s.get("frozen") or {}
    rows=[("Model",f"{s.get('model_id')} · {s.get('dtype')} · {s.get('attn')}"),("Architecture",arch_line(s)),("GPU",f"{s.get('gpu')} · {s.get('gpu_total_gib')} GiB"),
          ("Software",f"torch {s.get('torch')} · transformers {s.get('transformers')} · gradio {s.get('gradio')} · python {s.get('python')}"),
          ("v1.4 engine",f"SHA-256 {s.get('v14_sha256')} · {s.get('v14_lines')} lines · verified"),("TEST664 source",f"SHA-256 {s.get('source_sha256')}"),
          ("TEST664 panel",f"SHA-256 {s.get('panel_sha256')}"),("Injection",f"DC_CUT3 · H3 at layer {s.get('cut')} · K/V layers 0–{s.get('cut')} · {s.get('native_kernel')}"),
          ("Frozen weights",f"trainable tensors {fz.get('trainable_tensors')} · sentinel = load {fz.get('sentinel_equals_load')} · full-weight SHA-256 = TEST575 reference {s.get('full_weight_equals_reference')}"),
          ("Generation",f"greedy · at most {s.get('max_new')} new tokens · prefix {s.get('prefix_tokens')} tokens"),("Engine ready",f"{s.get('engine_loaded_at')} · boot {s.get('engine_seconds')} s")]
    return card_html("Engine (measured at start-up in this session)",'<table class="tbl">'+"".join(f"<tr><th>{esc(a)}</th><td class='mono'>{esc(b)}</td></tr>" for a,b in rows)+"</table>","on")

def scenario_values(scn):
    d=E["MAM14_EDITED"][scn];ed=scenario_edited(scn)
    info=card_html(SCENARIO_LABEL[scn],("<b>Edited cartridges</b> — answers are shown raw and are <b>not scored</b> (the audit references belong to the original texts)." if ed else
                   "Original cartridge texts (scored against the engine's audit references after generation).")+"<br>Questions: "+
                   "".join(f"<br><span class='mono'>{esc(q)}</span> {esc(t)}" for q,t in d["queries"])+"<br>Chain (asked one after another, previous raw answer inserted): "+
                   "".join(f"<br><span class='mono'>HOP {i+1}</span> {esc(t)}" for i,t in enumerate(d["hop_prompts"])),"warn" if ed else "info")
    return d["A"],d["B"],d["C"],d["C_CF"],info

def run_html(r,title):
    rows=""
    for q in r["responses"]:
        tok="—" if q.get("generated_tokens") is None else f"{q['generated_tokens']}"+(" <b style='color:#b91c1c'>limit</b>" if q.get("limit_reached") else "")
        rows+=(f"<tr><td class='mono'>{esc(q['question_id'])}</td><td>{esc(q['question'])}</td><td><code>{esc(q['answer'])}</code></td><td>{chip_html(q['status'])}</td>"
               f"<td class='hide-m mono'>{tok}</td><td class='hide-m mono'>{esc(q.get('elapsed_seconds'))} s</td><td class='hide-m small' style='opacity:.7'>{esc(q.get('expected') or '—')}</td></tr>")
    t=(f"<table class='tbl'><tr><th>ID</th><th>question</th><th>raw model answer</th><th>audit</th><th class='hide-m'>new tokens</th><th class='hide-m'>time</th>"
       f"<th class='hide-m'>audit reference (not model output)</th></tr>{rows}</table>")
    s=r.get("staged")
    if s:
        hops="".join(f"<br><span class='mono'>HOP {h['hop']}</span> {esc(h['question'])}<br>&nbsp;&nbsp;→ <code>{esc(h['answer'])}</code> {chip_html(h['status'])}"
                     +(f" <span class='small' style='opacity:.7'>({esc(h['note'])})</span>" if h.get("note") else "") for h in s["trace"])
        t+=f"<div class='small' style='margin-top:8px'><b>Staged chain</b> (predefined external sequence){hops}<br><b>chain result</b> <code>{esc(s['answer'])}</code> {chip_html(s['chain_status'])} · completed {esc(s['completed'])}</div>"
    t+=(f"<div class='small mono' style='margin-top:8px;opacity:.8'>memory {r['memory_rows']} rows · cartridges {esc(r['condition'])} · {esc(r['representation'])} · scored {r['scored']} · "
        f"engine {r['engine_elapsed_seconds']} s · weights frozen {r['weights_frozen']} · sentinel unchanged {r['sentinel_unchanged']} · forward segmentation {r['forward_segmentation_ok']}<br>"
        f"engine run SHA-256 {esc(r['run_sha256'])}<br>engine record {esc(r['engine_json'])}</div>")
    return card_html(title,t,"on")

GPU_LOCK=threading.Lock()
def busy_card():return card_html("Busy","Another run is using the GPU. Please wait until it has finished and press the button again.","warn")
def err_card(ex,stage=None):
    return card_html("Error"+(f" in stage {stage}" if stage else ""),f"<code>{esc(type(ex).__name__)}: {esc(ex)}</code><br>Nothing was hidden or replaced; the full traceback is in the log box below.","err")

def apply_handler(scn,a,b,c,ccf):
    try:
        cur=E["MAM14_EDITED"][scn]
        for slot,txt in (("A",a),("B",b),("C",c),("C_CF",ccf)):
            if (txt or "").strip()!=cur[slot]:E["set_cartridge"](slot,txt,scn)
        return scenario_values(scn)
    except Exception as ex:
        v=scenario_values(scn);return v[:4]+(err_card(ex),)
def reset_handler(scn):
    E["reset_scenario"](scn);return scenario_values(scn)

def single_handler(scn,cond,rep,q,staged):
    if not GPU_LOCK.acquire(blocking=False):yield busy_card(),"","",None;return
    try:
        yield card_html("Running",f"{esc(scn)} · {esc(cond)} · {esc(rep)} — building the memory and asking the frozen model …","info"),"","",None
        q=(q or "").strip() or None
        r=recorded_run(scn,cond,rep,question=q[:2000] if q else None,include_staged=bool(staged))
        d=ROOT/"single_runs";d.mkdir(exist_ok=True)
        p=dump_json(d/f"SINGLE_{scn}_{cond}_{rep}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json",r)
        yield run_html(r,f"{SCENARIO_LABEL[scn]} · {cond} · {rep}"),json.dumps(jsafe(r),ensure_ascii=False,indent=1)[:200000],"",str(p)
    except Exception as ex:
        yield err_card(ex),"",traceback.format_exc(),None
    finally:GPU_LOCK.release()

def stages_html(done,cur,msgs,t0,failed=None):
    out=[]
    for i,t in enumerate(STAGE_TITLES,1):
        if failed==i:cl,mk="fail","✗"
        elif i in done:cl,mk="done","✓"
        elif i==cur:cl,mk="run","…"
        else:cl,mk="wait","○"
        m=f" — <span class='small'>{esc(msgs.get(i,''))}</span>" if msgs.get(i) else ""
        out.append(f"<div class='stg {cl}'>{mk} {i}/{N_STAGES} · {esc(t)}{m}</div>")
    pc=100.0*len(done)/N_STAGES
    bar=f'<div style="margin-top:6px;background:#e5e7eb;height:10px"><div style="width:{pc:.0f}%;height:10px;background:#1d4ed8"></div></div>'
    return card_html(f"Full research run · {time.perf_counter()-t0:.0f} s","".join(out)+bar,"err" if failed else "info")

def full_handler(scn,cond,rep,q,text_ctl):
    NO=(None,None,None,"","")
    if not GPU_LOCK.acquire(blocking=False):yield (busy_card(),)+NO;return
    try:
        import queue as _q
        Q=_q.Queue();box={};t0=time.perf_counter();done=set();msgs={};cur=1
        def emit(stage,title,body,final=True):Q.put((stage,body,final))
        def work():
            try:box["res"]=full_run(dict(scenario=scn,condition=cond,representation=rep,custom=q or "",text_controls=bool(text_ctl)),emit)
            except BaseException as ex:box["err"]=ex;box["tb"]=traceback.format_exc()
        th=threading.Thread(target=work,daemon=True);th.start()
        while th.is_alive() or not Q.empty():
            while not Q.empty():
                st,body,fin=Q.get();msgs[st]=body
                if fin:done.update(range(1,st+1));cur=st+1
                else:done.update(range(1,st));cur=st
            yield (stages_html(done,cur,msgs,t0),)+NO
            time.sleep(.7)
        th.join()
        if "err" in box:
            ex=box["err"];stg=getattr(ex,"stage",None)
            yield (stages_html(done,cur,msgs,t0,failed=stg),None,None,None,"",box.get("tb",""));yield (err_card(ex,stg),None,None,None,"",box.get("tb",""));return
        res=box["res"];P=res["P"];files=res["files"]
        summ=Path(files["summary"]).read_text(encoding="utf-8")
        flist=list(res["figures"])+[files["zip"],files["summary"],files["manifest"],str(Path(res["run_dir"])/"RAW_MODEL_ANSWERS.json")]
        head=stages_html(set(range(1,N_STAGES+1)),None,msgs,t0)+run_html(P["primary"]["AKBASCORE"],"Demonstration A · ABC · DC_CUT3")+run_html(P["primary"]["MATRIX"],"Demonstration B · ABC · DC_CUT3")
        if P["custom"]:head+=run_html(P["custom"],"Custom question (unscored)")
        yield (head,[(f,Path(f).stem.replace("_"," ")) for f in res["figures"]],flist,files["zip"],summ,"")
    except Exception as ex:
        yield (err_card(ex),None,None,None,"",traceback.format_exc())
    finally:GPU_LOCK.release()

def verify_handler(full):
    if not GPU_LOCK.acquire(blocking=False):return busy_card()
    try:
        m=E["model"];fz=frozen_state();g=param_guard(m);rows=[("time (UTC)",utc_now()),("trainable tensors",fz["trainable_tensors"]),("training mode",fz["training"]),("LoRA",fz["lora"]),
            ("residual hooks",len(fz["residual_hooks"])),("sentinel = load",fz["sentinel_equals_load"]),("all-parameter guard = start-up",g==STARTUP.get("guard_start"))]
        if full:
            t=time.perf_counter();h=full_weight_sha(m);rows+=[("full-weight SHA-256",h),("= TEST575 reference",h==WEIGHT_SHA_REFERENCE),("seconds",round(time.perf_counter()-t,1))]
        ok=fz["trainable_tensors"]==0 and fz["sentinel_equals_load"] and g==STARTUP.get("guard_start") and (not full or rows[-2][1])
        return card_html("Weight verification · measured now","<table class='tbl'>"+"".join(f"<tr><th>{esc(a)}</th><td class='mono'>{esc(b)}</td></tr>" for a,b in rows)+"</table>","on" if ok else "err")
    except Exception as ex:return err_card(ex)
    finally:GPU_LOCK.release()

def _tb(**kw):
    for extra in (([{"buttons":["copy"]}] if _GR_MAJOR>=6 else [])+[{"show_copy_button":True},{}]):
        try:return gr.Textbox(**extra,**kw)
        except TypeError:continue
def _gallery(**kw):
    try:return gr.Gallery(columns=1,height="auto",object_fit="contain",preview=False,**kw)
    except TypeError:return gr.Gallery(**kw)

def build_ui():
    blocks=gr.Blocks(title=APP_SHORT) if _GR_MAJOR>=6 else gr.Blocks(css=CSS,title=APP_SHORT)
    with blocks as demo:
        gr.HTML(HERO);gr.HTML(engine_card_html())
        scn0="AKBASCORE";v=scenario_values(scn0)
        with gr.Row():
            scn_dd=gr.Dropdown(choices=[("Demonstration A — Mustafa Akbaş / AKBASCORE","AKBASCORE"),("Demonstration B — The Matrix / Neo","MATRIX")],value=scn0,label="Scenario",interactive=True)
            cond_dd=gr.Dropdown(choices=list(CONDITIONS),value="ABC",label="Memory condition (which cartridges are written)",interactive=True)
        rep_rd=gr.Radio(choices=[("DC_CUT3 · native injection (main demonstration)","DC_CUT3"),("TEXT_JOINT · facts read as text (comparison control)","TEXT_JOINT")],value="DC_CUT3",label="Memory representation",interactive=True)
        info=gr.HTML(v[4])
        with gr.Accordion("Cartridge texts · view, edit, reset (edited cartridges are answered but not scored)",open=False):
            ta=gr.Textbox(value=v[0],label="Cartridge A",lines=2,max_lines=6);tb=gr.Textbox(value=v[1],label="Cartridge B",lines=2,max_lines=6)
            tc=gr.Textbox(value=v[2],label="Cartridge C",lines=2,max_lines=6);tcf=gr.Textbox(value=v[3],label="Counterfactual C (used only in ABC_CF)",lines=2,max_lines=6)
            with gr.Row():ap=gr.Button("Apply edited cartridges");rs=gr.Button("Reset to original texts")
        cq=gr.Textbox(value="",label="Optional custom question (answered from the selected memory; not scored)",placeholder="e.g. Who activates the AKBASCORE MAM architecture?",lines=1,max_lines=3)
        stg=gr.Checkbox(value=True,label="Also run the three-step staged chain")
        run1=gr.Button("Run selected configuration",variant="secondary")
        full=gr.Button("FULL RESEARCH RUN · both demonstrations · 9 conditions · 7 figures + ZIP",variant="primary",elem_id="kz_full")
        txt_ctl=gr.Checkbox(value=True,label="Include the TEXT_JOINT control grid in the full run (doubles the control time)")
        status=gr.HTML(card_html("Ready","Choose a scenario, condition and representation, then press a button. Progress stages appear here only after they have actually completed.","info"))
        gallery=_gallery(label="Figures 1–7 (JPEG, 300 dpi)")
        files=gr.File(label="Individual downloads · figures, ZIP, summary, manifest, raw answers",file_count="multiple",interactive=False)
        zipb=gr.DownloadButton(label="⬇ Download the complete experiment package (.zip)",value=None,interactive=True)
        single_file=gr.File(label="Record of the selected-configuration run (.json)",interactive=False)
        with gr.Tabs():
            with gr.Tab("Run summary"):summ=_tb(lines=14,max_lines=40,label="RUN_SUMMARY.txt")
            with gr.Tab("Raw record (selected run)"):raw=_tb(lines=14,max_lines=40,label="JSON")
            with gr.Tab("Error log"):errs=_tb(lines=10,max_lines=40,label="traceback (empty = no error)")
        with gr.Accordion("Verify the frozen weights now",open=False):
            fullsha=gr.Checkbox(value=False,label="Also compute the full-weight SHA-256 (TEST575 method, about 1–2 minutes)")
            vb=gr.Button("Verify weights now");vout=gr.HTML()
        gr.HTML(SCOPE_HTML)
        gr.HTML(f'<div class="kz small" style="opacity:.8;margin:8px 0 18px">{esc(AUTHOR)} · {esc(ORG)} · {esc(LICENSE_NAME)} · '
                f'TEST664 <a href="{URL_664}" target="_blank" rel="noopener">664.py</a> · TEST677 <a href="{URL_677}" target="_blank" rel="noopener">677.py</a></div>')
        scn_dd.change(scenario_values,inputs=scn_dd,outputs=[ta,tb,tc,tcf,info])
        ap.click(apply_handler,inputs=[scn_dd,ta,tb,tc,tcf],outputs=[ta,tb,tc,tcf,info])
        rs.click(reset_handler,inputs=scn_dd,outputs=[ta,tb,tc,tcf,info])
        run1.click(single_handler,inputs=[scn_dd,cond_dd,rep_rd,cq,stg],outputs=[status,raw,errs,single_file])
        full.click(full_handler,inputs=[scn_dd,cond_dd,rep_rd,cq,txt_ctl],outputs=[status,gallery,files,zipb,summ,errs])
        vb.click(verify_handler,inputs=fullsha,outputs=vout)
    try:demo.queue(default_concurrency_limit=1)
    except TypeError:demo.queue(concurrency_count=1)
    return demo

if not LAYOUT_TEST:
    _GR_MAJOR=int(re.match(r"\d+",gr.__version__).group())
    boot_engine()
    demo=build_ui()
    say("[launch] public share link — open the printed gradio.live address (also works on a phone)")
    _LAUNCH=dict(share=True,allowed_paths=[str(ROOT),"/content/AKBASCORE_MAM_v1_4"],show_error=True,inline=False)
    if _GR_MAJOR>=6:_LAUNCH["css"]=CSS
    try:demo.launch(**_LAUNCH)
    except TypeError as _ex:
        if "css" not in str(_ex):raise
        _LAUNCH.pop("css",None);demo.launch(**_LAUNCH)






