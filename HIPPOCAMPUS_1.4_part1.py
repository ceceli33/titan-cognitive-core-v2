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


