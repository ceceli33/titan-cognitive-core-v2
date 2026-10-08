# TEST558 — AKBASCORE MAM · INDEPENDENT-CARTRIDGE COMPOSITION REPAIR
# TEST557 WORKING BACKBONE PRESERVED
# INDEP vs JOINT · L13/L15 STATE-CACHE REPAIR · 5-FACT COMPOSITION
# FROZEN MISTRAL · SAME PANEL · SAME TOKENS/LOGICAL POSITIONS · NO TRAINING/RETRIEVAL/ROUTER/QSF
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,inspect,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="558";SEED=552552;PANEL_SEED=550550;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16
POSITIONS=["FIRST","MIDDLE","LAST"];QUERY_CYCLE=["CURRENT","FORMER","NEAR","ROLE"];EXPECTED_PANEL_SHA="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
ARMS=["INDEP","JOINT","STATE13","STATE15","KV15","STATE15_KV15","STATE13_STATE15_KV15"]
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);T0=time.perf_counter()
def sha_obj(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold().strip(),flags=re.UNICODE).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
print("="*160);print("TEST558 — AKBASCORE MAM · INDEPENDENT-CARTRIDGE COMPOSITION REPAIR");print("TEST557 BACKBONE PRESERVED · INDEP vs JOINT · L13/L15 STATE-CACHE REPAIR");print("FROZEN MISTRAL · NO TRAINING · NO RETRIEVAL · NO ROUTER · NO QSF");print("="*160)
print("\n[1/10] Frozen Mistral...")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,QH,KVH,HD)}")
print("MODEL:",MODEL_ID);print("GPU:",torch.cuda.get_device_name(0));print("TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__);print(f"NL={NL} H={H} QH={QH} KVH={KVH} HD={HD} DTYPE={DTYPE}");print("ROPE:",getattr(cfg,"rope_parameters",None));print("TRAINABLE:",sum(int(p.requires_grad) for p in model.parameters()))
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
def source_prefix(f):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{f}\n\n"
def canonical_prefix():return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
PREFIX_IDS=tok(canonical_prefix(),add_special_tokens=False).input_ids;P=len(PREFIX_IDS);CAR=[];KEY_TO_IDX={}
for wi,w in enumerate(W):
    for typ,fact,gold in w["facts"]:
        idx=len(CAR);ids=tok(source_prefix(fact),add_special_tokens=False).input_ids
        if ids[:P]!=PREFIX_IDS:raise RuntimeError(f"Canonical prefix mismatch {idx}")
        CAR.append({"idx":idx,"world":wi,"type":typ,"fact":fact,"gold":gold,"body":ids[P:]});KEY_TO_IDX[(wi,typ)]=idx
if len(CAR)!=96:raise RuntimeError("Cartridge count mismatch.")
print("\n[2/10] Panel provenance...")
rng=random.Random(PANEL_SEED);pool=list(range(96));rng.shuffle(pool);PANEL=[];cursor=0
for wi in range(24):
    typ=QUERY_CYCLE[wi%4];target=KEY_TO_IDX[(wi,typ)];others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]["world"]==wi or any(CAR[x]["world"]==CAR[ci]["world"] for x in others):continue
        others.append(ci)
    PANEL.append({"case":wi,"target":target,"others":others})
PANEL_SHA=sha_obj(PANEL);print("PANEL SHA:",PANEL_SHA)
if PANEL_SHA!=EXPECTED_PANEL_SHA:raise RuntimeError("Panel provenance FAIL.")
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
            if k is None or v is None:raise RuntimeError("Cache API mismatch.")
            out.append((k,v))
        if out:return tuple(out)
    if hasattr(pkv,"key_cache"):return tuple(zip(pkv.key_cache,pkv.value_cache))
    raise RuntimeError("Cache API mismatch.")
def clone_layers(pkv):return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))
def build_cache(layers):
    c=DynamicCache()
    for li,(k,v) in enumerate(layers):c.update(k,v,li)
    return c
def cache_len(c):return int(c.get_seq_length())
@torch.no_grad()
def answer_layers(layers,q):
    cache=build_cache(layers);physical=cache_len(cache);qids=tok(query_suffix(q),add_special_tokens=False).input_ids;ids=torch.tensor(qids,dtype=torch.long,device=DEVICE).reshape(1,-1);qlen=ids.shape[1]
    pos=torch.arange(physical,physical+qlen,dtype=torch.long,device=DEVICE).unsqueeze(0);att=torch.ones((1,physical+qlen),dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);eos=set() if tok.eos_token_id is None else {int(tok.eos_token_id)};gen=[]
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        physical=cache_len(cache);pos=torch.tensor([[physical]],dtype=torch.long,device=DEVICE);att=torch.ones((1,physical+1),dtype=torch.long,device=DEVICE)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=att,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True);cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    return tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()
def make_case(keys,target):
    ids=list(PREFIX_IDS);ranges={}
    for ci in keys:
        a=len(ids);ids.extend(CAR[ci]["body"]);ranges[ci]=(a,len(ids))
    n=len(ids);group=torch.full((n,),-1,dtype=torch.long,device=DEVICE)
    for j,ci in enumerate(keys):
        a,b=ranges[ci];group[a:b]=j
    i=torch.arange(n,device=DEVICE);causal=i[:,None]>=i[None,:];allowed=causal&((group[:,None]==-1)|(group[None,:]==-1)|(group[:,None]==group[None,:]))
    block=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE);block.masked_fill_(~allowed,torch.finfo(DTYPE).min);joint=torch.zeros((1,1,n,n),dtype=DTYPE,device=DEVICE);joint.masked_fill_(~causal,torch.finfo(DTYPE).min)
    fm=torch.zeros(n,dtype=torch.bool,device=DEVICE);fm[P:]=True
    return torch.tensor(ids,dtype=torch.long,device=DEVICE).reshape(1,-1),joint,block,fm,ranges
@torch.no_grad()
def write(ids,joint,block,kind="JOINT",patches=None,capture=False):
    # patches: {layer: reference_hidden}; FULL information-body state replacement after that decoder layer.
    patches={} if patches is None else patches;handles=[];states={}
    try:
        for li,layer in enumerate(model.model.layers):
            def pre(module,args,kwargs,li=li):
                kwargs["attention_mask"]=joint if kind=="JOINT" else block;return args,kwargs
            handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True))
            if capture:
                def rec(module,args,out,li=li):
                    h=out[0] if isinstance(out,(tuple,list)) else out;states[li]=h.detach().clone()
                handles.append(layer.register_forward_hook(rec))
            if li in patches:
                ref,mask=patches[li]
                def patch(module,args,out,ref=ref,mask=mask):
                    h=out[0] if isinstance(out,(tuple,list)) else out
                    if h.shape!=ref.shape or mask.numel()!=h.shape[1]:raise RuntimeError("State patch mismatch.")
                    x=torch.where(mask.reshape(1,-1,1),ref,h)
                    if isinstance(out,tuple):return (x,)+out[1:]
                    if isinstance(out,list):return [x]+out[1:]
                    return x
                handles.append(layer.register_forward_hook(patch))
        n=ids.shape[1];pos=torch.arange(n,dtype=torch.long,device=DEVICE).unsqueeze(0)
        out=model(input_ids=ids,attention_mask=torch.ones((1,n),dtype=torch.long,device=DEVICE),position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
        layers=clone_layers(out.past_key_values);del out;return layers,states
    finally:
        for h in handles:h.remove()
def patch_cache(base,joint,layer,mask):
    out=[]
    for li,((bk,bv),(jk,jv)) in enumerate(zip(base,joint)):
        if li!=layer:out.append((bk,bv));continue
        if bk.shape!=jk.shape or bv.shape!=jv.shape or mask.numel()!=bk.shape[-2]:raise RuntimeError("KV patch mismatch.")
        m=mask.reshape(1,1,-1,1);out.append((torch.where(m,jk,bk),torch.where(m,jv,bv)))
    return tuple(out)
print("\n[3/10] Mask contract...")
z=PANEL[0];keys=order_keys(z,"MIDDLE");ids,joint,block,full,ranges=make_case(keys,z["target"]);n=ids.shape[1]
if joint.shape!=(1,1,n,n) or block.shape!=(1,1,n,n) or int(full[:P].sum())!=0:raise RuntimeError("Mask contract FAIL.")
print(f"MASK CONTRACT: PASS | TOKENS={n} PREFIX={P} FULL={int(full.sum())}");del ids,joint,block,full,ranges
print("\n[4/10] TEST556/557 baseline replication...")
BASELINES=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"];q=next(x[1] for x in W[ci]["queries"] if x[0]==typ);gold=CAR[z["target"]]["gold"]
    for slot in POSITIONS:
        keys=order_keys(z,slot);ids,joint,block,full,_=make_case(keys,z["target"]);jl,_=write(ids,joint,block,"JOINT");il,_=write(ids,joint,block,"INDEP");ja=answer_layers(jl,q);ia=answer_layers(il,q)
        BASELINES.append({"case":ci,"slot":slot,"joint":bool(hit(ja,gold)),"indep":bool(hit(ia,gold))});del jl,il,ids,joint,block,full
BJ={s:sum(int(x["joint"]) for x in BASELINES if x["slot"]==s) for s in POSITIONS};BI={s:sum(int(x["indep"]) for x in BASELINES if x["slot"]==s) for s in POSITIONS}
print("JOINT:",BJ);print("INDEP:",BI)
if BJ!={"FIRST":24,"MIDDLE":24,"LAST":24} or BI!={"FIRST":4,"MIDDLE":3,"LAST":6}:raise RuntimeError("Baseline replication FAIL.")
print("BASELINE REPLICATION: PASS")
print("\n[5/10] Independent-composition repair...")
RESULTS=[]
for ci,z in enumerate(PANEL):
    typ=CAR[z["target"]]["type"];q=next(x[1] for x in W[ci]["queries"] if x[0]==typ);gold=CAR[z["target"]]["gold"]
    for slot in POSITIONS:
        keys=order_keys(z,slot);ids,joint,block,full,_=make_case(keys,z["target"])
        jl,js=write(ids,joint,block,"JOINT",capture=True);il,_=write(ids,joint,block,"INDEP")
        arms={"INDEP":il,"JOINT":jl}
        s13,_=write(ids,joint,block,"INDEP",patches={13:(js[13],full)});arms["STATE13"]=s13
        s15,_=write(ids,joint,block,"INDEP",patches={15:(js[15],full)});arms["STATE15"]=s15
        kv15=patch_cache(il,jl,15,full);arms["KV15"]=kv15
        s15kv=patch_cache(s15,jl,15,full);arms["STATE15_KV15"]=s15kv
        s13s15,_=write(ids,joint,block,"INDEP",patches={13:(js[13],full),15:(js[15],full)});s13s15kv=patch_cache(s13s15,jl,15,full);arms["STATE13_STATE15_KV15"]=s13s15kv
        for arm,layers in arms.items():
            ans=answer_layers(layers,q);RESULTS.append({"case":ci,"slot":slot,"arm":arm,"gold":gold,"answer":ans,"ok":bool(hit(ans,gold))})
        del jl,js,il,s13,s15,kv15,s15kv,s13s15,s13s15kv,arms,ids,joint,block,full
    print(f"CASE {ci:02d} complete")
    if (ci+1)%4==0:torch.cuda.empty_cache();gc.collect();print(f"PROGRESS {ci+1}/24 | elapsed={time.perf_counter()-T0:.1f}s")
print("\n[6/10] Integrity...")
EXP=24*3*len(ARMS);print("ROWS:",len(RESULTS),"EXPECTED:",EXP)
if len(RESULTS)!=EXP:raise RuntimeError("Row count FAIL.")
for arm in ARMS:
    rr=[r for r in RESULTS if r["arm"]==arm]
    if len(rr)!=72 or len({(r["case"],r["slot"]) for r in rr})!=72:raise RuntimeError(f"{arm} integrity FAIL.")
print("RAW INTEGRITY: PASS")
print("\n[7/10] Repair scores...")
S={}
for arm in ARMS:
    S[arm]={slot:sum(int(r["ok"]) for r in RESULTS if r["arm"]==arm and r["slot"]==slot) for slot in POSITIONS};S[arm]["TOTAL"]=sum(S[arm][s] for s in POSITIONS)
    print(f"{arm:24s} FIRST={S[arm]['FIRST']:02d}/24 MIDDLE={S[arm]['MIDDLE']:02d}/24 LAST={S[arm]['LAST']:02d}/24 TOTAL={S[arm]['TOTAL']:02d}/72")
print("\n[8/10] Gain/loss vs INDEP + distance to JOINT...")
BASEMAP={(r["case"],r["slot"]):r for r in RESULTS if r["arm"]=="INDEP"};JOINTMAP={(r["case"],r["slot"]):r for r in RESULTS if r["arm"]=="JOINT"};DIAG={}
for arm in ARMS:
    if arm in ("INDEP","JOINT"):continue
    M={(r["case"],r["slot"]):r for r in RESULTS if r["arm"]==arm}
    gain=sum(int((not BASEMAP[k]["ok"]) and M[k]["ok"]) for k in BASEMAP);loss=sum(int(BASEMAP[k]["ok"] and not M[k]["ok"]) for k in BASEMAP)
    match_joint=sum(int(M[k]["ok"]==JOINTMAP[k]["ok"]) for k in BASEMAP);ans_joint=sum(int(norm(M[k]["answer"])==norm(JOINTMAP[k]["answer"])) for k in BASEMAP)
    DIAG[arm]={"gain":gain,"loss":loss,"net":gain-loss,"ok_identity_joint":match_joint,"answer_identity_joint":ans_joint}
    print(f"{arm:24s} GAIN={gain:02d} LOSS={loss:02d} NET={gain-loss:+03d} | OK-ID-JOINT={match_joint:02d}/72 ANS-ID-JOINT={ans_joint:02d}/72")
print("\n[9/10] Engineering decision...")
best=max([a for a in ARMS if a not in ("INDEP","JOINT")],key=lambda a:S[a]["TOTAL"]);best_score=S[best]["TOTAL"];gap=72-best_score
if best_score==72:DECISION=f"FULL REPAIR — {best} REACHES JOINT 72/72"
elif best_score>=65:DECISION=f"STRONG PARTIAL REPAIR — {best}={best_score}/72, GAP TO JOINT={gap}"
elif best_score>S["INDEP"]["TOTAL"]:DECISION=f"PARTIAL REPAIR — {best}={best_score}/72, GAP TO JOINT={gap}"
else:DECISION="NO REPAIR ABOVE INDEPENDENT BASELINE"
print("BEST:",best,best_score,"/72");print("JOINT:",S["JOINT"]["TOTAL"],"/72");print("INDEP:",S["INDEP"]["TOTAL"],"/72");print("DECISION:",DECISION)
print("CAUTION: JOINT hidden states/KV are oracle reference interventions; this test asks causal sufficiency for composition repair, not yet deployable independent reconstruction.")
print("\n[10/10] Final research record...")
WEIGHT_OK=sentinel()==S0;TRAINABLE=sum(int(p.requires_grad) for p in model.parameters())
LOCK={"test":TEST,"model":MODEL_ID,"panel_sha":PANEL_SHA,"seed":SEED,"panel_seed":PANEL_SEED,"arms":ARMS,"repair_layers":[13,15],"scope":"FULL","baseline":"TEST556/557 exact BLOCK-style independent write","oracle_reference":"JOINT hidden states/cache","training":False,"retrieval":False,"router":False,"qsf":False}
LOCK_SHA=sha_obj(LOCK);FINAL={"test":TEST,"lock_sha":LOCK_SHA,"panel_sha":PANEL_SHA,"scores":S,"diag":DIAG,"best":best,"decision":DECISION,"weight_ok":WEIGHT_OK,"trainable":TRAINABLE};RESULT_SHA=sha_obj(FINAL)
print("="*160);print("TEST558 — FINAL RESEARCH RECORD");print("="*160)
print("MODEL                  :",MODEL_ID);print("PANEL SHA              :",PANEL_SHA);print("LOCK SHA               :",LOCK_SHA);print("RESULT SHA             :",RESULT_SHA)
for arm in ARMS:print(f"{arm:24s}: {S[arm]}")
print("BEST REPAIR            :",best,f"{best_score}/72");print("ENGINEERING DECISION   :",DECISION);print("WEIGHT SENTINEL        :","PASS" if WEIGHT_OK else "FAIL");print("TRAINABLE              :",TRAINABLE);print("TOTAL TIME             :",f"{time.perf_counter()-T0:.2f}s")
print("INTERPRETATION         : Oracle JOINT state/cache repair tests whether missing joint-write internal coordination is causally sufficient to repair independent composition.")
print("NEXT TEST              : TEST559 only after TEST558 interpretation");print("="*160)
if not WEIGHT_OK:raise RuntimeError("WEIGHT SENTINEL FAILURE.")
if TRAINABLE!=0:raise RuntimeError("TRAINABLE PARAMETER FAILURE.")
