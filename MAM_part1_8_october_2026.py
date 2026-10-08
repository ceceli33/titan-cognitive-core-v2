# =====================================================================================================================================
# AKBASCORE MAM · PERSISTENT NUMERICAL MEMORY FOR FROZEN LANGUAGE MODELS — PART 1 / 3 · ENGINE
# Public technical demonstration and priority record · Discovered and developed by Mustafa Akbaş · AkbasCore AI Teknoloji · 8 October 2026
# Run PART 1 → PART 2 → PART 3 as three consecutive Google Colab cells in the SAME runtime (GPU: A100 40 GB recommended).
# PART 1 embeds the TEST560 incremental-DC6 engine VERBATIM, verifies its SHA-256, loads frozen Mistral-7B-Instruct-v0.3 and attaches
# read-only measurement. PART 2 adds the fail-closed measurement pipeline and figures. PART 3 starts the Gradio demonstration.
# AKBASCORE RESEARCH SOFTWARE LICENSE · Copyright © 2026 Mustafa Akbaş · https://github.com/ceceli33/titan-cognitive-core-v2
#
# ENGINE         : MAM_TEST560_DC6_ENGINE.py, embedded byte-for-byte as ENGINE_SRC and executed in its own module namespace.
#                  The memory architecture is frozen; this program does not modify, recalibrate or re-implement it.
# MECHANISM      : each source fact is read once by the frozen model and stored as an independent numerical cartridge
#                  (layer-6 residual H6 + layers 0–6 KV). Cartridges are appended one by one into a consolidated numerical memory:
#                  layers 7–31 are computed only for the new tokens, attending to the existing memory. Questions are answered from
#                  the numerical memory; the source text is not given to the model at query time. Model weights are never changed.
# INSTRUMENTATION: (1) a recording-only forward pre-hook, attached only while an engine API call runs; it records the kind and length of
#                  every model input (token ids or embeddings), whether input embeddings are all zero, and the installed cache length;
#                  (2) an append observer, attached only during load_case(): it wraps the engine's own append method, calls it unchanged
#                  and checks that every previously consolidated K/V row is bitwise identical after the append; (3) an all-parameter
#                  weight guard; (4) re-derivation of every reported quantity from raw records.
# =====================================================================================================================================
import os,sys,io,re,json,math,time,html,random,shutil,hashlib,zipfile,platform,textwrap,threading,subprocess,importlib.util,traceback,gc,types
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote
from collections import Counter
#<<CONST_BEGIN>>
DEMO_TITLE="AKBASCORE MAM · Persistent numerical memory for frozen language models"
DEMO_SHORT="AKBASCORE MAM · TEST560 incremental DC6 demonstration"
AUTHOR="Mustafa Akbaş";ORG="AkbasCore AI Teknoloji";DATE_TXT="8 October 2026";DATE_ISO="2026-10-08";AUTHOR_PLACE="Mersin, Türkiye"
COPYRIGHT="Copyright © 2026 Mustafa Akbaş";LICENSE_NAME="AKBASCORE RESEARCH SOFTWARE LICENSE"
LICENSE_URL="https://github.com/ceceli33/titan-cognitive-core-v2"
SOURCE_URL="https://github.com/ceceli33/titan-cognitive-core-v2/blob/main/560.py";LOG_URL="https://github.com/ceceli33/titan-cognitive-core-v2/blob/main/560.log"
PARADIGM="A new numerical-memory paradigm for frozen language models"
DISCOVERY="Discovered and developed by Mustafa Akbaş"
PRIORITY=("Public technical demonstration and priority record of the AKBASCORE MAM architecture. "
          "Mustafa Akbaş (AkbasCore AI Teknoloji) discovered and developed this numerical-memory mechanism and records his priority for it on 8 October 2026.")
CORE_MESSAGE=["Independent numerical cartridges are written by the frozen model — each source fact is read once.",
              "Cartridges are consolidated append-only into one shared numerical memory.",
              "No earlier source text is replayed during consolidation.",
              "Previously consolidated memory tokens are not recomputed.",
              "Model weights are not changed.",
              "The question is answered without giving the source text to the model again."]
ENGINE_FILE="MAM_TEST560_DC6_ENGINE.py"
ENGINE_SHA256_EXPECTED="b07562c84f014ec100dc89a9194ad028d37a5fabfac7488137940b0bbdec4b5f"
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";ARCH=(32,4096,32,8);CUT_EXPECTED=6;PREFIX_EXPECTED=27;N_CASES=24;N_CART=5;POSITIONS=("FIRST","MIDDLE","LAST")
TARGET_SLOT={"FIRST":0,"MIDDLE":2,"LAST":4}
CANON_GPU="NVIDIA A100-SXM4-40GB"
# ---- sealed records (read from the archived experiment logs; never produced or recomputed by this program) ----
TEST560=dict(name="TEST560",lock="9b8b78dddfdd401781e2bfab29c9e52e3227ef22bb132e22396a9d685cd32d56",result_sha="6a1d7453970b0e23e80abf31234b58c37fa39c42fc7971f5473b192dc11802a7",
    panel_sha=EXPECTED_PANEL_SHA,model=MODEL_ID,gpu=CANON_GPU,torch="2.11.0+cu130",transformers="5.18.0",cut=6,prefix=27,cases=24,positions=3,total=72,
    arms={"INDEP":{"FIRST":4,"MIDDLE":3,"LAST":6,"TOTAL":13},"JOINT":{"FIRST":24,"MIDDLE":24,"LAST":24,"TOTAL":72},
          "BATCH_DC6":{"FIRST":24,"MIDDLE":24,"LAST":24,"TOTAL":72},"INCR_DC6":{"FIRST":24,"MIDDLE":24,"LAST":24,"TOTAL":72}},
    incr_gain=59,incr_loss=0,incr_answer_identity_joint=(71,72),incr_answer_identity_batch=(72,72),
    decision="FULL INCREMENTAL CONSOLIDATION — APPEND-ONLY MEMORY MATCHES BATCH ACCURACY 72/72")
ARM_DESC={"INDEP":"independent cartridges, no consolidation (control)","JOINT":"facts processed together as text (upper bound, not a memory)",
          "BATCH_DC6":"deferred consolidation of all five cartridges at once","INCR_DC6":"append-only incremental consolidation (this engine)"}
SCALE_BOUNDARY=dict(name="TEST563",result_sha="41072d660a1b1437f56e3ba42afffa406adf2f78fe6c82de5e58390399309616",
    rows=[(5,5),(10,9),(20,16),(40,16),(80,22)],note="same incremental DC6 mechanism, larger cartridge populations (sealed)")
SCOPE={
 "demonstrated":[
  "Five source facts are written independently by frozen Mistral-7B-Instruct-v0.3: each fact is read once and stored as a numerical cartridge (layer-6 residual H6 + layers 0–6 K/V).",
  "The cartridges are appended one by one into one consolidated numerical memory. For each append, layers 7–31 are computed only for the new tokens, attending to the existing memory; layers 0–6 keys are re-phased to the new positions with the model's own rotary embedding.",
  "Measured in every run: the consolidation forward receives all-zero input embeddings plus the stored H6 (no token ids); no earlier cartridge's source is read again; every previously consolidated K/V row is bitwise identical after each append.",
  "The question is answered from the numerical memory: the query forward receives only the question-template tokens over the installed memory.",
  "Model weights are unchanged (all-parameter guard before and after the run)."],
 "archived":[
  "TEST560 sealed results on the locked panel (24 five-cartridge cases × FIRST/MIDDLE/LAST = 72): append-only incremental DC6 72/72, batch DC6 72/72, joint text 72/72, independent cartridges without consolidation 13/72. These are reference values; single live runs do not recompute them. The optional live replay reports its own score separately.",
  "TEST563 scale boundary of the same mechanism: 5/5, 9/10, 16/20, 16/40, 22/80 correct for 5–80 cartridges."],
 "not_established":[
  "Large memory populations: accuracy of this mechanism falls as the number of cartridges grows (TEST563: 22/80 at 80 cartridges).",
  "Strict KV-only sufficiency: the cartridge is augmented (H6 residual + layers 0–6 K/V).",
  "Questions outside the locked panel template: free-text questions are shown but not scored.",
  "Multi-step inference that combines several cartridges.",
  "Order-free consolidation: the append uses causal order; equal accuracy at FIRST/MIDDLE/LAST was measured in TEST560 on this panel."],
 "notes":[
  "Writing cartridge k reads only fact k (inside the fixed instruction template) once; this is the write event of that cartridge.",
  "The 27-token instruction prefix is written with the first cartridge; later cartridges contribute their body rows only.",
  "Expected answers come from panel construction and are audit metadata; they are not an input to writing, consolidation or answering.",
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
# ---- claim-discipline scanner: applied to every generated text (figures, logs, JSON). Panel names are removed first. ----
ALLOWED_TESTS={"560","563"}
CLAIM_RULES=[("capacity-overclaim",r"(?i)\bunlimited\b|\binfinite\b"),("universal-claim",r"(?i)\buniversal\b"),("human-like-claim",r"(?i)human-like"),
    ("model-independence-claim",r"(?i)model-independent"),("all-transformers-claim",r"(?i)all transformers"),("solved-claim",r"(?i)\bsolved\b"),
    ("perfect-claim",r"(?i)\bperfect\b"),("world-first-claim",r"(?i)world'?s first|first in the world|independently proven"),
    ("legal-status-claim",r"(?i)\bpatent(ed|-pending)\b|non-obvious|\bpatentab"),("metaphor",r"(?i)\bbrain\b|\bquantum\b|\bneuron"),
    ("complexity-claim",r"O\(1\)"),("compression-wording",r"(?i)\bcompress")]
def claim_scan(name,text,allow=()):
    for a in allow:
        if a:text=text.replace(a,"<allowed>")
    hits=[(name,lab,m.group(0))for lab,p in CLAIM_RULES for m in re.finditer(p,text)]
    hits+=[(name,"unexpected-test-number",m.group(0))for m in re.finditer(r"TEST\s?(\d+)",text)if m.group(1)not in ALLOWED_TESTS]
    return hits
def strip_words(text,words):
    ws=sorted({w for w in words if w and len(w)>=2},key=len,reverse=True)
    if not ws:return text
    return re.sub(r"(?<![A-Za-z])(?:"+"|".join(map(re.escape,ws))+r")(?![A-Za-z])","",text)
def cpu_selftest():
    n=0
    assert hit_ans("Melket","Melket") and hit_ans("The capital is Melket.","melket") and not hit_ans("Melketon","Melket");n+=1
    assert claim_scan("t","Mistral-7B-Instruct-v0.3 TEST560 TEST563 72/72 append-only DC6 H6 layers 0–6 "+PARADIGM+" "+DISCOVERY+" "+PRIORITY)==[];n+=1
    for b in("unlimited memory","universal memory","human-like memory","model-independent","works on all transformers","memory solved","perfect recall",
             "world's first","patent-pending","brain","quantum","O(1)","compressed","TEST528"):
        assert claim_scan("t",b),b;n+=1
    assert strip_words("Zorvan BRAIN carried",["BRAIN"])=="Zorvan  carried";n+=1
    assert sum(TEST560["arms"]["INDEP"][p]for p in POSITIONS)==TEST560["arms"]["INDEP"]["TOTAL"]==13;n+=1
    assert all(sum(TEST560["arms"][a][p]for p in POSITIONS)==TEST560["arms"][a]["TOTAL"]==72 for a in("JOINT","BATCH_DC6","INCR_DC6"));n+=1
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
say("="*140);say(DEMO_TITLE+" — PART 1 / 3 · ENGINE");say(f"{DISCOVERY} · {ORG} · {DATE_TXT} · {AUTHOR_PLACE}");say(f"{LICENSE_NAME} · {COPYRIGHT} · {LICENSE_URL}");say("="*140)
say(f"[1/5] CPU self-test PASS ({cpu_selftest()} checks)")
# ---------------- TEST560 ENGINE — embedded verbatim (do not edit; its SHA-256 is verified below) ----------------
ENGINE_SRC=r'''"""
AKBASCORE MAM — WORLD DEMONSTRATION ENGINE
Developed by AkbasCore AI Teknoloji | Mustafa Akbaş | 8 October 2026

MAM (Memory Architecture): A frozen language model converts source facts into independent
numerical cartridges (layer-6 residual plus layers 0–6 KV). The cartridges are appended
into a consolidated numerical memory. The original source text is not replayed during
consolidation, and previously consolidated tokens are not recomputed. Questions are
answered from the resulting numerical memory without changing model weights.

Research-validated scope: TEST560, 24 five-cartridge cases × three target positions,
72/72 incremental-DC6 answers; independent control 13/72. This is a bounded
experimental demonstration, NOT a claim of arbitrary-size or perfect memory.
TEST561–563 document a separate scaling limitation (80-cartridge DC6: 22/80).

AKBASCORE RESEARCH SOFTWARE LICENSE
Copyright © 2026 Mustafa Akbaş
License terms: https://github.com/ceceli33/titan-cognitive-core-v2
Source: https://github.com/ceceli33/titan-cognitive-core-v2/blob/main/560.py
Log: https://github.com/ceceli33/titan-cognitive-core-v2/blob/main/560.log
"""
import os,random,hashlib,json,re,time,threading
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";SEED=552552;PANEL_SEED=550550;CUT=6;MAX_NEW=16
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
POSITIONS=("FIRST","MIDDLE","LAST");QUERY_CYCLE=("CURRENT","FORMER","NEAR","ROLE")
BASE=[
("Zorvan","Melket","Dravel","Oakhaven","Pelnor"),("Kelvar","Nareth","Solven","Branik","Tarsen"),("Tarev","Luneth","Varos","Cedran","Mireth"),("Belnor","Arven","Dorel","Kesmar","Falven"),
("Ravik","Selora","Terven","Maldor","Nerik"),("Nemor","Calven","Istral","Pareth","Dovren"),("Darsen","Velora","Keldin","Orvek","Sarnel"),("Feron","Talven","Merith","Sovran","Belvik"),
("Larev","Nerith","Calder","Veyron","Tormek"),("Torven","Elsar","Marvek","Dorin","Kaleth"),("Selnor","Kareth","Valen","Ordan","Mervek"),("Mirev","Taldor","Neris","Kelmar","Sorvik"),
("Varen","Solith","Deran","Malvek","Cordan"),("Kelor","Ardin","Velmar","Toren","Narell"),("Narev","Belith","Corven","Sareth","Dorvik"),("Dervan","Mirel","Talvek","Orsen","Kelron"),
("Calnor","Verith","Naldor","Seren","Parvek"),("Parel","Dorven","Kelith","Maros","Tervik"),("Sorven","Tarell","Vindor","Nelmar","Calrek"),("Barel","Corith","Laven","Derik","Solmar"),
("Ralen","Mervor","Talith","Kesven","Noreth"),("Norel","Valdor","Serith","Calvenor","Darvek"),("Tervan","Orel","Mardin","Velos","Karven"),("Karev","Solen","Dareth","Mirven","Talrek")]
SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def source_prefix(f):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{f}\n\n"
def canonical_prefix():return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
class MAMDemo:
    def __init__(self):
        if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required (A100 40 GB reference environment).")
        os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        self.device=torch.device("cuda");self.dtype=torch.bfloat16;self.lock=threading.RLock()
        self.tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
        self.model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=self.dtype).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        c=self.model.config;self.nl=len(self.model.model.layers);self.h=c.hidden_size
        if (self.nl,self.h,c.num_attention_heads,c.num_key_value_heads)!=(32,4096,32,8):raise RuntimeError("Mistral architecture mismatch")
        self.prefix_ids=self.tok(canonical_prefix(),add_special_tokens=False).input_ids;self.p=len(self.prefix_ids)
        self.worlds=[];self.car=[];self.key_to_idx={}
        for wi,(s,current,near,former,role_target) in enumerate(BASE):
            facts=[("CURRENT",f"The current capital of {s} is {current}.",current),("NEAR",f"The largest city of {s} is {near}.",near),("FORMER",f"The former capital of {s} was {former}.",former),("ROLE",f"The current capital of {role_target} is {s}.",s)]
            queries=[("CURRENT",f"What is the current capital of {s}?",current),("FORMER",f"What was the former capital of {s}?",former),("NEAR",f"What is the largest city of {s}?",near),("ROLE",f"What is the current capital of {role_target}?",s)]
            self.worlds.append({"id":wi,"facts":facts,"queries":queries})
            for typ,fact,gold in facts:
                idx=len(self.car);ids=self.tok(source_prefix(fact),add_special_tokens=False).input_ids
                if ids[:self.p]!=self.prefix_ids:raise RuntimeError("Canonical prefix mismatch")
                self.car.append({"idx":idx,"world":wi,"type":typ,"fact":fact,"gold":gold,"body":ids[self.p:]});self.key_to_idx[(wi,typ)]=idx
        rng=random.Random(PANEL_SEED);pool=list(range(96));rng.shuffle(pool);self.panel=[];cursor=0
        for wi in range(24):
            typ=QUERY_CYCLE[wi%4];target=self.key_to_idx[(wi,typ)];others=[]
            while len(others)<4:
                ci=pool[cursor%96];cursor+=1
                if ci==target or self.car[ci]["world"]==wi or any(self.car[x]["world"]==self.car[ci]["world"] for x in others):continue
                others.append(ci)
            self.panel.append({"case":wi,"target":target,"others":others})
        self.panel_sha=sha_obj(self.panel)
        if self.panel_sha!=EXPECTED_PANEL_SHA:raise RuntimeError(f"TEST560 panel provenance mismatch: {self.panel_sha}")
        self._memory=None;self._selection=None;self._events=[]
    def _cache_layers(self,pkv):
        if hasattr(pkv,"layers"):
            out=[]
            for layer in pkv.layers:
                k=getattr(layer,"keys",None);v=getattr(layer,"values",None)
                if k is None:k=getattr(layer,"key_cache",None)
                if v is None:v=getattr(layer,"value_cache",None)
                if k is None or v is None:raise RuntimeError("Cache API mismatch")
                out.append((k,v))
            if out:return tuple(out)
        if hasattr(pkv,"key_cache"):return tuple(zip(pkv.key_cache,pkv.value_cache))
        raise RuntimeError("Cache API mismatch")
    def _clone_layers(self,pkv):return tuple((k.detach().clone(),v.detach().clone()) for k,v in self._cache_layers(pkv))
    def _build_cache(self,layers):
        c=DynamicCache()
        for li,(k,v) in enumerate(layers):c.update(k,v,li)
        return c
    def _replace_hidden(self,out,new):
        if isinstance(out,tuple):return (new,)+out[1:]
        if isinstance(out,list):return [new]+out[1:]
        return new
    def _rotate_half(self,x):
        a,b=x.chunk(2,dim=-1);return torch.cat((-b,a),dim=-1)
    @torch.no_grad()
    def _rope_tables(self,pos):
        dummy=torch.zeros((1,pos.numel(),self.h),dtype=self.dtype,device=self.device);cos,sin=self.model.model.rotary_emb(dummy,pos.reshape(1,-1))
        return cos.unsqueeze(1).float(),sin.unsqueeze(1).float()
    @torch.no_grad()
    def _rephase_key(self,k,oldpos,newpos):
        if torch.equal(oldpos,newpos):return k
        co,so=self._rope_tables(oldpos);cn,sn=self._rope_tables(newpos);x=k.float();unrot=x*co-self._rotate_half(x)*so
        return (unrot*cn+self._rotate_half(unrot)*sn).to(k.dtype)
    @torch.no_grad()
    def _forge_checkpoint(self,ci):
        ids=self.prefix_ids+self.car[ci]["body"];n=len(ids);x=torch.tensor(ids,dtype=torch.long,device=self.device).reshape(1,-1);pos=torch.arange(n,dtype=torch.long,device=self.device);state={};handles=[]
        try:
            def cap(module,args,out):state["h"]=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
            handles.append(self.model.model.layers[CUT].register_forward_hook(cap))
            out=self.model(input_ids=x,attention_mask=torch.ones((1,n),dtype=torch.long,device=self.device),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
            early=tuple((k.detach().clone(),v.detach().clone()) for k,v in self._cache_layers(out.past_key_values)[:CUT+1]);h=state["h"];del out
            return {"h":h,"kv":early,"positions":pos.clone(),"body_len":n-self.p}
        finally:
            for h in handles:h.remove()
    @torch.no_grad()
    def _init_incremental(self,ci):
        cp=self._forge_checkpoint(ci);n=self.p+cp["body_len"];assembled=cp["h"];early=[]
        for li in range(CUT+1):early.append((cp["kv"][li][0].detach().clone(),cp["kv"][li][1].detach().clone()))
        handles=[]
        try:
            def inject(module,args,out):return self._replace_hidden(out,assembled)
            handles.append(self.model.model.layers[CUT].register_forward_hook(inject))
            zeros=torch.zeros((1,n,self.h),dtype=self.dtype,device=self.device);pos=torch.arange(n,device=self.device)
            out=self.model(inputs_embeds=zeros,attention_mask=torch.ones((1,n),dtype=torch.long,device=self.device),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True);gen=self._clone_layers(out.past_key_values);result=tuple(early)+gen[CUT+1:];del out,gen,zeros,cp
            return result
        finally:
            for h in handles:h.remove()
    @torch.no_grad()
    def _append_incremental(self,memory,ci):
        cp=self._forge_checkpoint(ci);oldn=memory[0][0].shape[-2];q=cp["body_len"];newpos=torch.arange(oldn,oldn+q,dtype=torch.long,device=self.device);body_h=cp["h"][:,self.p:,:]
        if body_h.shape!=(1,q,self.h):raise RuntimeError("Incremental H6 body mismatch")
        cache=self._build_cache(memory);handles=[]
        try:
            def inject(module,args,out):
                h=out[0] if isinstance(out,(tuple,list)) else out
                if h.shape!=body_h.shape:raise RuntimeError("Incremental H6 injection mismatch")
                return self._replace_hidden(out,body_h)
            handles.append(self.model.model.layers[CUT].register_forward_hook(inject))
            zeros=torch.zeros((1,q,self.h),dtype=self.dtype,device=self.device);att=torch.ones((1,oldn+q),dtype=torch.long,device=self.device)
            out=self.model(inputs_embeds=zeros,past_key_values=cache,attention_mask=att,position_ids=newpos.unsqueeze(0),cache_position=newpos,use_cache=True,return_dict=True);grown=list(self._clone_layers(out.past_key_values))
            for li in range(CUT+1):
                oldk,oldv=memory[li];k=cp["kv"][li][0][:,:,self.p:,:];v=cp["kv"][li][1][:,:,self.p:,:];k=self._rephase_key(k,cp["positions"][self.p:],newpos)
                grown[li]=(torch.cat((oldk,k),dim=-2),torch.cat((oldv,v),dim=-2))
            result=tuple(grown);del out,grown,zeros,cache,cp
            if result[0][0].shape[-2]!=oldn+q:raise RuntimeError("Incremental cache growth FAIL")
            return result
        finally:
            for h in handles:h.remove()
    @torch.no_grad()
    def _answer_layers(self,layers,q):
        cache=self._build_cache(layers);physical=int(cache.get_seq_length());qids=self.tok(query_suffix(q),add_special_tokens=False).input_ids;ids=torch.tensor(qids,dtype=torch.long,device=self.device).reshape(1,-1);qlen=ids.shape[1]
        pos=torch.arange(physical,physical+qlen,dtype=torch.long,device=self.device).unsqueeze(0);att=torch.ones((1,physical+qlen),dtype=torch.long,device=self.device)
        out=self.model(input_ids=ids,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);eos=set() if self.tok.eos_token_id is None else {int(self.tok.eos_token_id)};gen=[]
        for _ in range(MAX_NEW):
            gen.append(nxt)
            if int(nxt.item()) in eos:break
            physical=int(cache.get_seq_length());pos=torch.tensor([[physical]],dtype=torch.long,device=self.device);att=torch.ones((1,physical+1),dtype=torch.long,device=self.device)
            out=self.model(input_ids=nxt,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
        return self.tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()
    def _keys(self,case,position):
        z=self.panel[case];d=list(z["others"]);t=z["target"]
        return [t]+d if position=="FIRST" else d[:2]+[t]+d[2:] if position=="MIDDLE" else d+[t]
    def cases(self):
        return [{"case":i,"type":self.car[z["target"]]["type"],"question":next(x[1] for x in self.worlds[i]["queries"] if x[0]==self.car[z["target"]]["type"]),"positions":list(POSITIONS)} for i,z in enumerate(self.panel)]
    def status(self):
        return {"title":"AKBASCORE MAM","developer":"AkbasCore AI Teknoloji","author":"Mustafa Akbaş","date":"2026-10-08","model":MODEL_ID,"panel_sha":self.panel_sha,"reference_test":"TEST560","reference_score":"72/72","scale_limit":"TEST563: 80-cartridge DC6 22/80","loaded_case":self._selection,"cartridges":0 if self._selection is None else 5,"events":list(self._events)}
    def load_case(self,case=0,position="MIDDLE"):
        case=int(case);position=str(position).upper()
        if not 0<=case<24 or position not in POSITIONS:raise ValueError("case must be 0..23; position FIRST/MIDDLE/LAST")
        with self.lock,torch.no_grad():
            t0=time.perf_counter();keys=self._keys(case,position);memory=None;events=[]
            for step,ci in enumerate(keys):
                if memory is None:memory=self._init_incremental(ci)
                else:
                    new=self._append_incremental(memory,ci);del memory;memory=new
                events.append({"step":step+1,"cartridge_id":ci,"position":step+1,"cache_tokens":int(memory[0][0].shape[-2]),"source_replayed_in_consolidation":False,"prior_tokens_recomputed":False})
            self._memory=memory;self._selection={"case":case,"position":position};self._events=events
            typ=self.car[self.panel[case]["target"]]["type"];q=next(x[1] for x in self.worlds[case]["queries"] if x[0]==typ)
            return {"case":case,"position":position,"cartridges":[{"id":ci,"type":self.car[ci]["type"],"fact":self.car[ci]["fact"],"is_target":ci==self.panel[case]["target"]} for ci in keys],"question":q,"expected":self.car[self.panel[case]["target"]]["gold"],"events":events,"elapsed_seconds":round(time.perf_counter()-t0,3),"weights_updated":False,"source_replay_during_consolidation":False,"prior_memory_recomputed":False}
    def ask(self,question=None):
        with self.lock,torch.no_grad():
            if self._memory is None:raise RuntimeError("Call load_case() first")
            case=self._selection["case"];typ=self.car[self.panel[case]["target"]]["type"];reference_q=next(x[1] for x in self.worlds[case]["queries"] if x[0]==typ);q=reference_q if question is None else str(question)
            t0=time.perf_counter();answer=self._answer_layers(self._memory,q);is_reference=q==reference_q
            gold=self.car[self.panel[case]["target"]]["gold"] if is_reference else None
            return {"question":q,"answer":answer,"reference_question":is_reference,"expected":gold,"correct":hit(answer,gold) if is_reference else None,"elapsed_seconds":round(time.perf_counter()-t0,3),"source_text_supplied_to_query":False,"model_weights_changed":False}
    def run_case(self,case=0,position="MIDDLE"):
        setup=self.load_case(case,position);answer=self.ask();return {"setup":setup,"answer":answer}
    def run_reference_panel(self,progress=None):
        rows=[];t0=time.perf_counter()
        for case in range(24):
            for position in POSITIONS:
                r=self.run_case(case,position);rows.append({"case":case,"position":position,"answer":r["answer"]["answer"],"expected":r["answer"]["expected"],"correct":r["answer"]["correct"]})
                if progress:progress(len(rows),72,rows[-1])
        score=sum(int(r["correct"]) for r in rows)
        return {"score":score,"total":72,"matches_TEST560":score==72,"panel_sha":self.panel_sha,"rows":rows,"elapsed_seconds":round(time.perf_counter()-t0,2)}
if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(description="AKBASCORE MAM — TEST560-faithful numerical-memory demo backend")
    ap.add_argument("--case",type=int,default=0);ap.add_argument("--position",choices=POSITIONS,default="MIDDLE");ap.add_argument("--full-panel",action="store_true");args,_=ap.parse_known_args()
    demo=MAMDemo();print(json.dumps(demo.run_reference_panel() if args.full_panel else demo.run_case(args.case,args.position),indent=2,ensure_ascii=False))'''
ENGINE_SHA256=hashlib.sha256(ENGINE_SRC.encode("utf-8")).hexdigest()
if ENGINE_SHA256!=ENGINE_SHA256_EXPECTED:
    raise RuntimeError(f"Embedded engine source differs from the supplied engine file (SHA-256 {ENGINE_SHA256} != {ENGINE_SHA256_EXPECTED}). Re-paste PART 1 unchanged.")
say(f"[2/5] Engine source verified · {ENGINE_FILE} · SHA-256 {ENGINE_SHA256}")
STARTUP_UTC=utc_now()
if"ENG"in globals()and getattr(globals()["ENG"],"__engine_sha256__",None)==ENGINE_SHA256 and globals().get("DEMO_ENGINE")is not None:
    say("[3/5] Engine already initialised in this runtime — reusing it (no second model load).");ENGINE_INIT_SECONDS=globals().get("ENGINE_INIT_SECONDS",float("nan"))
else:
    say("[3/5] Executing the engine in its own module namespace and constructing MAMDemo() (loads frozen Mistral-7B, about 1–2 minutes) …")
    ENG=types.ModuleType("mam_test560_dc6_engine");ENG.__file__=ENGINE_FILE+" (embedded verbatim)";sys.modules[ENG.__name__]=ENG
    exec(compile(ENGINE_SRC,ENGINE_FILE,"exec"),ENG.__dict__)
    ENG.__engine_sha256__=ENGINE_SHA256
    _t=time.perf_counter();DEMO_ENGINE=ENG.MAMDemo();torch.cuda.synchronize();ENGINE_INIT_SECONDS=time.perf_counter()-_t
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
class Instrument:
    """Thin adapter around the engine's public API (MAMDemo), plus measurement. Every scientific quantity comes from the engine itself."""
    def __init__(self,D,init_seconds):
        self.D=D;self.kind="ENGINE";m=D.model;c=m.config
        self.info=dict(model_id=ENG.MODEL_ID,arch=[D.nl,D.h,c.num_attention_heads,c.num_key_value_heads],dtype=str(next(m.parameters()).dtype).replace("torch.",""),
            attn=getattr(c,"_attn_implementation",None),gpu=torch.cuda.get_device_name(0),gpu_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            torch=torch.__version__,transformers=transformers.__version__,gradio=gr.__version__,python=sys.version.split()[0],platform=platform.platform(),
            params=sum(p.numel()for p in m.parameters()),rope=str(getattr(c,"rope_parameters",None)),engine_file=ENGINE_FILE,engine_sha256=ENGINE_SHA256,
            init_seconds=init_seconds,startup_utc=STARTUP_UTC,
            sentinel_method="SHA-256 over 16 slices × 256 values of 7 weight tensors (TEST560 sentinel layout)",
            guard_method="all-parameter guard: SHA-256 over (name, shape, dtype, sum, |sum|, sum of squares) of every parameter tensor",
            forward_counter="recording-only forward pre-hook on the causal-LM module, attached only during an engine API call",
            observer="append observer: wraps the engine's own _init_incremental/_append_incremental during load_case() only; calls them unchanged; compares prior K/V rows bitwise")
        self.info["sentinel0"]=self.sentinel();self.info["guard0"]=self.guard();self.info["hooks0"]=hook_stats(m)[0]
    # ---- read-only views of engine state ----
    def config(self):
        D=self.D
        return dict(MODEL_ID=ENG.MODEL_ID,CUT=ENG.CUT,MAX_NEW=ENG.MAX_NEW,SEED=ENG.SEED,PANEL_SEED=ENG.PANEL_SEED,EXPECTED_PANEL_SHA=ENG.EXPECTED_PANEL_SHA,panel_sha=D.panel_sha,
                    prefix_tokens=D.p,n_cartridges=len(D.car),n_cases=len(D.panel),positions=list(ENG.POSITIONS),system=ENG.SYSTEM,
                    engine_sha256=hashlib.sha256(ENGINE_SRC.encode("utf-8")).hexdigest())
    def status(self):return self.D.status()
    def cases(self):return self.D.cases()
    def keys(self,case,position):return list(self.D._keys(int(case),str(position)))
    def car(self,ci):c=self.D.car[ci];return dict(id=ci,world=c["world"],type=c["type"],fact=c["fact"],gold=c["gold"],body_len=len(c["body"]))
    def target(self,case):return int(self.D.panel[case]["target"])
    def source_ids(self,ci):return list(self.D.prefix_ids)+list(self.D.car[ci]["body"])
    def query_ids(self,q):return list(self.D.tok(ENG.query_suffix(q),add_special_tokens=False).input_ids)
    def decode_each(self,ids):return[self.D.tok.decode([t])for t in ids]
    def panel_strings(self):return sorted({w for row in ENG.BASE for w in row})
    # ---- integrity ----
    def sentinel(self):
        m=self.D.model;L=m.model.layers
        FP=[L[0].self_attn.q_proj.weight,L[8].self_attn.o_proj.weight,L[16].mlp.down_proj.weight,L[24].self_attn.o_proj.weight,L[31].mlp.down_proj.weight,m.model.norm.weight,m.lm_head.weight]
        h=hashlib.sha256()
        for p in FP:
            a=p.detach().reshape(-1);n=a.numel()
            for j in range(16):o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
        return h.hexdigest()
    def guard(self):
        h=hashlib.sha256()
        with torch.inference_mode():
            for name,p in self.D.model.named_parameters():
                x=p.detach().float();vals=(float(x.sum().item()),float(x.abs().sum().item()),float((x*x).sum().item()))
                h.update(name.encode());h.update(str(tuple(p.shape)).encode());h.update(str(p.dtype).encode());h.update(np.asarray(vals,dtype=np.float64).tobytes());del x
        torch.cuda.empty_cache();return h.hexdigest()
    def frozen(self):
        m=self.D.model;tot,fo=hook_stats(m)
        return dict(training=bool(m.training),trainable_tensors=sum(int(p.requires_grad)for p in m.parameters()),
                    lora=bool(hasattr(m,"peft_config")or any("lora"in n.lower()for n,_ in m.named_modules())),optimizer=optimizer_present(),
                    hooks=tot,foreign_hooks=sum(fo.values()),foreign_modules=fo,sentinel=self.sentinel())
    # ---- measured engine calls ----
    def _recorder(self):
        rec=[]
        def _rec(mod,a,kw):
            ids=kw.get("input_ids",a[0]if a else None);emb=kw.get("inputs_embeds");pkv=kw.get("past_key_values");pl=None
            if pkv is not None:
                try:pl=int(pkv.get_seq_length())
                except Exception:pl=-1
            if ids is not None:
                n=int(ids.shape[-1]);rec.append(dict(kind="ids",n=n,ids=[int(t)for t in ids.reshape(-1).tolist()]if n<=256 else None,embeds_zero=None,has_past=pkv is not None,past_len=pl))
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
say("[4/5] Measurement adapter · weight sentinel · all-parameter weight guard …")
INSTR=Instrument(DEMO_ENGINE,ENGINE_INIT_SECONDS)
_cfg=INSTR.config();_st=INSTR.status();_fz=INSTR.frozen()
ENGINE_LOCK=[("engine source SHA-256 = supplied engine file",_cfg["engine_sha256"]==ENGINE_SHA256_EXPECTED),
 ("model = mistralai/Mistral-7B-Instruct-v0.3",_cfg["MODEL_ID"]==MODEL_ID),("architecture 32 layers / hidden 4096 / 32 Q / 8 KV heads",tuple(INSTR.info["arch"])==ARCH),
 ("consolidation cut = layer 6",_cfg["CUT"]==CUT_EXPECTED),("canonical instruction prefix = 27 tokens",_cfg["prefix_tokens"]==PREFIX_EXPECTED),
 ("locked TEST560 panel SHA-256",_cfg["panel_sha"]==EXPECTED_PANEL_SHA==_cfg["EXPECTED_PANEL_SHA"]),("96 cartridges · 24 cases · FIRST/MIDDLE/LAST",_cfg["n_cartridges"]==96 and _cfg["n_cases"]==N_CASES and tuple(_cfg["positions"])==POSITIONS),
 ("engine status: TEST560 reference 72/72",_st.get("reference_test")=="TEST560"and _st.get("reference_score")=="72/72"),
 ("weight sentinel = value at initialisation",_fz["sentinel"]==INSTR.info["sentinel0"]),("trainable tensors = 0",_fz["trainable_tensors"]==0),
 ("model in eval mode, no LoRA, no optimizer",(not _fz["training"])and not _fz["lora"]and not _fz["optimizer"]),("no foreign forward/backward hooks",_fz["foreign_hooks"]==0)]
_bad=[n for n,ok in ENGINE_LOCK if not ok]
if _bad:raise RuntimeError(f"ENGINE LOCK FAILED: {_bad}")
say(f"[5/5] Engine lock PASS · {len(ENGINE_LOCK)} checks · init {ENGINE_INIT_SECONDS:.1f}s · guard {INSTR.info['guard0'][:24]}… · sentinel {INSTR.info['sentinel0'][:24]}…")
say("PART 1 / 3 complete. Now run PART 2 / 3 in the next cell.");say("="*140)
MAM_DC6_PART1_OK=True
#<<ENGINE_END>>
