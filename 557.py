# TEST557 — AKBASCORE MAM · RESIDUAL-STATE vs K/V CAUSAL CHANNEL DISSECTION
# TEST556 BASELINE/PANEL/MASK/SCORING PRESERVED
# CRITICAL CUTS ONLY: L13 + L15
# SWITCH / STATE / KV / STATE+KV × FULL / TARGET / HISTORY
# FROZEN MISTRAL · NO TRAINING · NO RETRIEVAL · NO ROUTER · NO QSF
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,inspect,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

TEST="557";SEED=552552;PANEL_SEED=550550
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";DEVICE=torch.device("cuda");DTYPE=torch.bfloat16
MAX_NEW=16;CUTS=[13,15];SCOPES=["FULL","TARGET","HISTORY"];MODES=["STATE","KV","STATE_KV"]
FACT_TYPES=["CURRENT","NEAR","FORMER","ROLE"];QUERY_CYCLE=["CURRENT","FORMER","NEAR","ROLE"];POSITIONS=["FIRST","MIDDLE","LAST"]
EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
if not torch.cuda.is_available(): raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);T0=time.perf_counter()
def sha_obj(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s): return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE).split())
def hit(s,g): return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None

print("="*164)
print("TEST557 — AKBASCORE MAM · RESIDUAL-STATE vs K/V CAUSAL CHANNEL DISSECTION")
print("TEST556 CONTRACT PRESERVED · CUTS L13/L15 · SWITCH / STATE / KV / STATE+KV")
print("FROZEN MISTRAL · NO TRAINING · NO RETRIEVAL · NO ROUTER · NO QSF")
print("="*164)

print("\n[1/10] Frozen Mistral...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters(): p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128): raise RuntimeError(f"Architecture mismatch: {(NL,H,QH,KVH,HD)}")
print("MODEL:",MODEL_ID);print("GPU:",torch.cuda.get_device_name(0));print("TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__)
print("NL=%d H=%d QH=%d KVH=%d HD=%d DTYPE=%s"%(NL,H,QH,KVH,HD,DTYPE));print("ROPE:",getattr(cfg,"rope_parameters",None));print("TRAINABLE:",sum(int(p.requires_grad) for p in model.parameters()))
FP=[model.model.layers[0].self_attn.q_proj.weight,model.model.layers[8].self_attn.o_proj.weight,model.model.layers[16].mlp.down_proj.weight,model.model.layers[24].self_attn.o_proj.weight,model.model.layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
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
W=[]
for i,(s,current,near,former,role_target) in enumerate(BASE):
    facts=[("CURRENT",f"The current capital of {s} is {current}.",current),("NEAR",f"The largest city of {s} is {near}.",near),("FORMER",f"The former capital of {s} was {former}.",former),("ROLE",f"The current capital of {role_target} is {s}.",s)]
    queries=[("CURRENT",f"What is the current capital of {s}?",current),("FORMER",f"What was the former capital of {s}?",former),("NEAR",f"What is the largest city of {s}?",near),("ROLE",f"What is the current capital of {role_target}?",s)]
    W.append({"id":i,"facts":facts,"queries":queries})
SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def source_prefix(f): return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{f}\n\n"
def canonical_prefix(): return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"
def query_suffix(q): return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
PREFIX_IDS=tok(canonical_prefix(),add_special_tokens=False).input_ids;P=len(PREFIX_IDS)
CAR=[];KEY_TO_IDX={}
for wi,w in enumerate(W):
    for typ,fact,gold in w["facts"]:
        idx=len(CAR);ids=tok(source_prefix(fact),add_special_tokens=False).input_ids
        if ids[:P]!=PREFIX_IDS: raise RuntimeError(f"Canonical prefix mismatch cartridge {idx}")
        CAR.append({"idx":idx,"world":wi,"type":typ,"fact":fact,"gold":gold,"body":ids[P:]});KEY_TO_IDX[(wi,typ)]=idx
if len(CAR)!=96: raise RuntimeError("Cartridge count mismatch.")

print("\n[2/10] Panel provenance...")
rng=random.Random(PANEL_SEED);pool=list(range(96));rng.shuffle(pool);PANEL=[];cursor=0
for wi in range(24):
    typ=QUERY_CYCLE[wi%4];target=KEY_TO_IDX[(wi,typ)];others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]["world"]==wi or any(CAR[x]["world"]==CAR[ci]["world"] for x in others): continue
        others.append(ci)
    PANEL.append({"case":wi,"target":target,"others":others})
PANEL_SHA=sha_obj(PANEL);print("PANEL SHA:",PANEL_SHA)
if PANEL_SHA!=EXPECTED_PANEL_SHA: raise RuntimeError("Panel provenance FAIL.")
print("PANEL PROVENANCE: PASS")

def order_keys(z,slot):
    d=list(z["others"]);t=z["target"]
    return [t]+d if slot=="FIRST" else d[:2]+[t]+d[2:] if slot=="MIDDLE" else d+[t]
def cache_layers(pkv):
    if hasattr(pkv,"layers"):
        out=[]
        for layer in pkv.layers:
            k=getattr(layer,"keys",None);v=getattr(layer,"values",None)
            if k is None:k=getattr(layer,"key_cache",None)
            if v is None:v=getattr(layer,"value_cache",None)
            if k is None or v is None: raise RuntimeError("Cache API mismatch.")
            out.append((k,v))
        if out:return tuple(out)
    if hasattr(pkv,"key_cache"): return tuple(zip(pkv.key_cache,pkv.value_cache))
    raise RuntimeError("Cache API mismatch.")
def clone_layers(pkv): return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))
def build_cache(layers):
    c=DynamicCache()
    for li,(k,v) in enumerate(layers):c.update(k,v,li)
    return c
def cache_len(c):return int(c.get_seq_length())

@torch.no_grad()
def answer_layers(layers,q):
    cache=build_cache(layers);physical=cache_len(cache);qids=tok(query_suffix(q),add_special_tokens=False).input_ids
    ids=torch.tensor(qids,dtype=torch.long,device=DEVICE).reshape(1,-1);qlen=ids.shape[1]
    pos=torch.arange(physical,physical+qlen,dtype=torch.long,device=DEVICE).unsqueeze(0);att=torch.ones((1,physical+qlen),dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
    cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);eos=set() if tok.eos_token_id is None else {int(tok.eos_token_id)};gen=[]
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        physical=cache_len(cache);pos=torch.tensor([[physical]],dtype=torch.long,device=DEVICE);att=torch.ones((1,physical+1),dtype=torch.long,device=DEVICE)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
        cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()

def make_case(keys,target):
    ids=list(PREFIX_IDS);ranges={}
    for ci in keys:
        a=len(ids);ids.extend(CAR[ci]["body"]);ranges[ci]=(a,len(ids))
    n=len(ids);group=torch.full((n,),-1,dtype=torch.long,device=DEVICE)
    for j,ci in enumerate(keys):
        a,b=ranges[ci];group[a:b]=j
    i=torch.arange(n,device=DEVICE);causal=i[:,None]>=i[None,:]
    allowed=causal&((group[:,None]==-1)|(group[None,:]==-1)|(group[:,None]==group[None,:]))
    block=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE);block.masked_fill_(~allowed,torch.finfo(DTYPE).min)
    joint=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE);joint.masked_fill_(~causal,torch.finfo(DTYPE).min)
    a,b=ranges[target];tm=torch.zeros(n,dtype=torch.bool,device=DEVICE);tm[a:b]=True;hm=torch.zeros(n,dtype=torch.bool,device=DEVICE)
    for ci in keys:
        if ci!=target:
            x,y=ranges[ci];hm[x:y]=True
    fm=torch.zeros(n,dtype=torch.bool,device=DEVICE);fm[P:]=True
    return torch.tensor(ids,dtype=torch.long,device=DEVICE).reshape(1,-1),joint,block,{"FULL":fm,"TARGET":tm,"HISTORY":hm},ranges

# PASS CONTRACT
# SWITCH: BLOCK layers <= cut; JOINT layers > cut.
# STATE : SWITCH + JOINT decoder-layer output state transplant at cut.
# KV    : SWITCH hidden stream untouched; after write pass replace CUT-layer cache K/V
#         with the corresponding JOINT CUT-layer cache K/V on selected token positions.
# STATE_KV: STATE plus the same selected JOINT CUT-layer K/V transplant.
#
# IMPORTANT: KV transplant is applied to the completed write cache AFTER the forward pass.
# Therefore it isolates the effect of CUT-layer stored K/V during later QUESTION readout.
# It does NOT retroactively alter write-time computation in layers > cut.
# STATE, by contrast, can alter downstream write-time computation and therefore downstream caches.

@torch.no_grad()
def write_pass(ids,joint,block,mode="JOINT",cut=None,scope=None,reference_state=None,mask=None,capture=False):
    handles=[];states={}
    if mode in ("SWITCH","STATE","STATE_KV") and cut not in CUTS: raise RuntimeError("Invalid cut.")
    if mode in ("STATE","STATE_KV") and (scope not in SCOPES or reference_state is None or mask is None): raise RuntimeError("STATE transplant contract failure.")
    try:
        for li,layer in enumerate(model.model.layers):
            def prehook(module,args,kwargs,li=li):
                if mode=="JOINT":chosen=joint
                elif mode=="BLOCK":chosen=block
                elif mode in ("SWITCH","STATE","STATE_KV"):chosen=block if li<=cut else joint
                else:raise ValueError(mode)
                kwargs["attention_mask"]=chosen;return args,kwargs
            handles.append(layer.register_forward_pre_hook(prehook,with_kwargs=True))
            if capture:
                def record(module,args,output,li=li):
                    h=output[0] if isinstance(output,(tuple,list)) else output;states[li]=h.detach().clone()
                handles.append(layer.register_forward_hook(record))
            if mode in ("STATE","STATE_KV") and li==cut:
                def transplant(module,args,output,li=li):
                    h=output[0] if isinstance(output,(tuple,list)) else output;ref=reference_state[li]
                    if h.shape!=ref.shape or mask.numel()!=h.shape[1]:raise RuntimeError("STATE transplant shape mismatch.")
                    patched=torch.where(mask.reshape(1,-1,1),ref,h)
                    if isinstance(output,tuple):return (patched,)+output[1:]
                    if isinstance(output,list):return [patched]+output[1:]
                    return patched
                handles.append(layer.register_forward_hook(transplant))
        n=ids.shape[1];pos=torch.arange(n,dtype=torch.long,device=DEVICE).unsqueeze(0)
        out=model(input_ids=ids,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
        layers=clone_layers(out.past_key_values);del out
        return layers,states
    finally:
        for h in handles:h.remove()

def patch_cache_layer(base_layers,joint_layers,cut,mask):
    out=[]
    for li,((bk,bv),(jk,jv)) in enumerate(zip(base_layers,joint_layers)):
        if li!=cut:
            out.append((bk,bv));continue
        if bk.shape!=jk.shape or bv.shape!=jv.shape:raise RuntimeError(f"KV shape mismatch L{cut}")
        if mask.numel()!=bk.shape[-2]:raise RuntimeError(f"KV mask length mismatch L{cut}")
        m=mask.reshape(1,1,-1,1)
        pk=torch.where(m,jk,bk);pv=torch.where(m,jv,bv);out.append((pk,pv))
    return tuple(out)

print("\n[3/10] Mask contract...")
z=PANEL[0];keys=order_keys(z,"MIDDLE");ids,joint,block,masks,ranges=make_case(keys,z["target"]);n=ids.shape[1]
if joint.shape!=(1,1,n,n) or block.shape!=(1,1,n,n):raise RuntimeError("Mask shape mismatch.")
if torch.any(masks["TARGET"]&masks["HISTORY"]):raise RuntimeError("TARGET/HISTORY overlap.")
if not torch.equal(masks["FULL"],masks["TARGET"]|masks["HISTORY"]):raise RuntimeError("FULL union mismatch.")
if int(masks["FULL"][:P].sum())!=0:raise RuntimeError("Prefix scope leak.")
print(f"MASK CONTRACT: PASS | TOKENS={n} PREFIX={P} TARGET={int(masks['TARGET'].sum())} HISTORY={int(masks['HISTORY'].sum())} FULL={int(masks['FULL'].sum())}")
del ids,joint,block,masks,ranges

print("\n[4/10] Baseline replication...")
BASELINES=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"];q=next(x[1] for x in W[ci]["queries"] if x[0]==typ);gold=CAR[z["target"]]["gold"]
    for slot in POSITIONS:
        keys=order_keys(z,slot);ids,joint,block,masks,ranges=make_case(keys,z["target"])
        jl,_=write_pass(ids,joint,block,"JOINT");bl,_=write_pass(ids,joint,block,"BLOCK")
        ja=answer_layers(jl,q);ba=answer_layers(bl,q)
        BASELINES.append({"case":ci,"slot":slot,"joint_ok":bool(hit(ja,gold)),"block_ok":bool(hit(ba,gold))})
        del jl,bl,ids,joint,block,masks,ranges
BJ={s:sum(int(x["joint_ok"]) for x in BASELINES if x["slot"]==s) for s in POSITIONS}
BB={s:sum(int(x["block_ok"]) for x in BASELINES if x["slot"]==s) for s in POSITIONS}
print("JOINT:",BJ);print("BLOCK:",BB)
if BJ!={"FIRST":24,"MIDDLE":24,"LAST":24} or BB!={"FIRST":4,"MIDDLE":3,"LAST":6}:raise RuntimeError("TEST556 baseline replication FAIL.")
print("BASELINE REPLICATION: PASS")

print("\n[5/10] L13/L15 channel dissection...")
SWITCH=[];RESULTS=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"];q=next(x[1] for x in W[ci]["queries"] if x[0]==typ);gold=CAR[z["target"]]["gold"]
    for slot in POSITIONS:
        keys=order_keys(z,slot);ids,joint,block,masks,ranges=make_case(keys,z["target"])
        joint_layers,joint_states=write_pass(ids,joint,block,"JOINT",capture=True)
        for cut in CUTS:
            sw_layers,_=write_pass(ids,joint,block,"SWITCH",cut=cut);sw_ans=answer_layers(sw_layers,q)
            SWITCH.append({"case":ci,"slot":slot,"layer":cut,"gold":gold,"answer":sw_ans,"ok":bool(hit(sw_ans,gold))})
            for scope in SCOPES:
                state_layers,_=write_pass(ids,joint,block,"STATE",cut=cut,scope=scope,reference_state=joint_states,mask=masks[scope])
                state_ans=answer_layers(state_layers,q)
                RESULTS.append({"case":ci,"slot":slot,"layer":cut,"scope":scope,"mode":"STATE","gold":gold,"answer":state_ans,"ok":bool(hit(state_ans,gold))})
                kv_layers=patch_cache_layer(sw_layers,joint_layers,cut,masks[scope]);kv_ans=answer_layers(kv_layers,q)
                RESULTS.append({"case":ci,"slot":slot,"layer":cut,"scope":scope,"mode":"KV","gold":gold,"answer":kv_ans,"ok":bool(hit(kv_ans,gold))})
                state_kv_layers=patch_cache_layer(state_layers,joint_layers,cut,masks[scope]);state_kv_ans=answer_layers(state_kv_layers,q)
                RESULTS.append({"case":ci,"slot":slot,"layer":cut,"scope":scope,"mode":"STATE_KV","gold":gold,"answer":state_kv_ans,"ok":bool(hit(state_kv_ans,gold))})
                del state_layers,kv_layers,state_kv_layers
            del sw_layers
        del joint_layers,joint_states,ids,joint,block,masks,ranges
    print(f"CASE {ci:02d} complete")
    if (ci+1)%4==0:torch.cuda.empty_cache();gc.collect();print(f"PROGRESS {ci+1}/24 | elapsed={time.perf_counter()-T0:.1f}s")

print("\n[6/10] Integrity...")
EXP_SWITCH=24*3*len(CUTS);EXP_RESULTS=24*3*len(CUTS)*len(SCOPES)*len(MODES)
print("SWITCH:",len(SWITCH),"EXPECTED:",EXP_SWITCH);print("RESULTS:",len(RESULTS),"EXPECTED:",EXP_RESULTS)
if len(SWITCH)!=EXP_SWITCH or len(RESULTS)!=EXP_RESULTS:raise RuntimeError("Row count FAIL.")
for cut in CUTS:
    if sum(1 for r in SWITCH if r["layer"]==cut)!=72:raise RuntimeError(f"SWITCH L{cut} cardinality FAIL.")
    for scope in SCOPES:
        for mode in MODES:
            n=sum(1 for r in RESULTS if r["layer"]==cut and r["scope"]==scope and r["mode"]==mode)
            if n!=72:raise RuntimeError(f"{mode}/{scope}/L{cut}: {n} != 72")
print("RAW INTEGRITY: PASS")

print("\n[7/10] Channel scores...")
SUMMARY={"SWITCH":{}}
for cut in CUTS:SUMMARY["SWITCH"][cut]=sum(int(r["ok"]) for r in SWITCH if r["layer"]==cut)
for mode in MODES:
    SUMMARY[mode]={}
    for scope in SCOPES:
        SUMMARY[mode][scope]={}
        for cut in CUTS:SUMMARY[mode][scope][cut]=sum(int(r["ok"]) for r in RESULTS if r["mode"]==mode and r["scope"]==scope and r["layer"]==cut)
for cut in CUTS:
    print(f"\nL{cut:02d} SWITCH={SUMMARY['SWITCH'][cut]}/72")
    for scope in SCOPES:
        s=SUMMARY["STATE"][scope][cut];k=SUMMARY["KV"][scope][cut];sk=SUMMARY["STATE_KV"][scope][cut];sw=SUMMARY["SWITCH"][cut]
        print(f"{scope:8s} STATE={s:02d}/72 ΔS={s-sw:+03d} | KV={k:02d}/72 ΔK={k-sw:+03d} | STATE+KV={sk:02d}/72 ΔSK={sk-sw:+03d}")

print("\n[8/10] Case-level causal decomposition...")
DECOMP={}
for cut in CUTS:
    DECOMP[cut]={}
    sw={(r["case"],r["slot"]):r for r in SWITCH if r["layer"]==cut}
    for scope in SCOPES:
        maps={m:{(r["case"],r["slot"]):r for r in RESULTS if r["layer"]==cut and r["scope"]==scope and r["mode"]==m} for m in MODES}
        keys=set(sw)
        if any(set(maps[m])!=keys for m in MODES):raise RuntimeError("Case map mismatch.")
        state_only=sum(int((not sw[k]["ok"]) and maps["STATE"][k]["ok"] and not maps["KV"][k]["ok"]) for k in keys)
        kv_only=sum(int((not sw[k]["ok"]) and maps["KV"][k]["ok"] and not maps["STATE"][k]["ok"]) for k in keys)
        both=sum(int((not sw[k]["ok"]) and maps["STATE"][k]["ok"] and maps["KV"][k]["ok"]) for k in keys)
        neither=sum(int((not sw[k]["ok"]) and not maps["STATE"][k]["ok"] and not maps["KV"][k]["ok"]) for k in keys)
        sk_gain=sum(int((not sw[k]["ok"]) and maps["STATE_KV"][k]["ok"]) for k in keys)
        sk_over_state=sum(int((not maps["STATE"][k]["ok"]) and maps["STATE_KV"][k]["ok"]) for k in keys)
        sk_harm=sum(int(maps["STATE"][k]["ok"] and not maps["STATE_KV"][k]["ok"]) for k in keys)
        DECOMP[cut][scope]={"STATE_ONLY":state_only,"KV_ONLY":kv_only,"BOTH":both,"NEITHER":neither,"STATE_KV_GAIN":sk_gain,"STATE_KV_OVER_STATE":sk_over_state,"STATE_KV_HARM":sk_harm}
        print(f"L{cut:02d} {scope:8s} STATE_ONLY={state_only:02d} KV_ONLY={kv_only:02d} BOTH={both:02d} NEITHER={neither:02d} | STATE+KV_GAIN={sk_gain:02d} OVER_STATE={sk_over_state:02d} HARM={sk_harm:02d}")

print("\n[9/10] Mechanism decision...")
# Primary channel attribution uses net accuracy over identical SWITCH control.
CHANNEL={}
for cut in CUTS:
    CHANNEL[cut]={}
    sw=SUMMARY["SWITCH"][cut]
    for scope in SCOPES:
        ds=SUMMARY["STATE"][scope][cut]-sw;dk=SUMMARY["KV"][scope][cut]-sw;dsk=SUMMARY["STATE_KV"][scope][cut]-sw
        if ds>0 and dk<=0:label="STATE-DOMINANT"
        elif dk>0 and ds<=0:label="CUT-KV-DOMINANT"
        elif ds>0 and dk>0:
            if ds>dk:label="BOTH-ACTIVE / STATE-LARGER"
            elif dk>ds:label="BOTH-ACTIVE / KV-LARGER"
            else:label="BOTH-ACTIVE / TIED"
        else:label="NO POSITIVE CHANNEL EFFECT"
        CHANNEL[cut][scope]={"delta_state":ds,"delta_kv":dk,"delta_state_kv":dsk,"label":label}
        print(f"L{cut:02d} {scope:8s}: ΔSTATE={ds:+d} ΔKV={dk:+d} ΔSTATE+KV={dsk:+d} => {label}")

print("\n[10/10] Final research record...")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
LOCK={"test":TEST,"model":MODEL_ID,"panel_sha":PANEL_SHA,"seed":SEED,"panel_seed":PANEL_SEED,"cuts":CUTS,"scopes":SCOPES,"modes":MODES,"switch_contract":"BLOCK <= cut; JOINT > cut","state_contract":"TEST556 decoder-layer-output transplant","kv_contract":"post-write selected-token replacement of cut-layer cache K/V from JOINT reference","state_kv_contract":"STATE write plus selected-token cut-layer JOINT K/V replacement","baseline_contract":"TEST556 exact","training":False,"retrieval":False,"router":False,"qsf":False}
LOCK_SHA=sha_obj(LOCK)
FINAL={"test":TEST,"lock_sha":LOCK_SHA,"panel_sha":PANEL_SHA,"baseline_joint":BJ,"baseline_block":BB,"summary":SUMMARY,"decomposition":DECOMP,"channel":CHANNEL,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE}
RESULT_SHA=sha_obj(FINAL)
print("="*164)
print("TEST557 — FINAL RESEARCH RECORD")
print("="*164)
print("MODEL               :",MODEL_ID)
print("PANEL SHA           :",PANEL_SHA)
print("LOCK SHA            :",LOCK_SHA)
print("RESULT SHA          :",RESULT_SHA)
print("CUTS                :",CUTS)
print("JOINT BASELINE      :",BJ)
print("BLOCK BASELINE      :",BB)
for cut in CUTS:
    print(f"L{cut:02d} SWITCH          : {SUMMARY['SWITCH'][cut]}/72")
    for scope in SCOPES:
        print(f"L{cut:02d} {scope:8s} STATE={SUMMARY['STATE'][scope][cut]}/72 KV={SUMMARY['KV'][scope][cut]}/72 STATE+KV={SUMMARY['STATE_KV'][scope][cut]}/72 | {CHANNEL[cut][scope]['label']}")
print("WEIGHT SENTINEL     :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE           :",TRAINABLE)
print("TOTAL TIME          :",f"{time.perf_counter()-T0:.2f}s")
print("INTERPRETATION RULE : KV arm tests stored cut-layer K/V contribution during QUESTION readout; it does not recreate downstream write-time K/V.")
print("NEXT TEST           : TEST558 only after TEST557 interpretation")
print("="*164)
if not WEIGHT_OK:raise RuntimeError("WEIGHT SENTINEL FAILURE.")
if TRAINABLE!=0:raise RuntimeError("TRAINABLE PARAMETER FAILURE.")
