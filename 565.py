
# TEST565 — AKBASCORE MAM · QWEN FULL-LAYER X-RAY + CAUSAL CUT VALIDATION
# TEST564 WORKING BACKBONE PRESERVED · 28-LAYER MEASUREMENT · NO TRAINING / RETRIEVAL / ROUTER
import os,sys,subprocess,importlib.util,time,random,re,hashlib,json,gc,math
for m in ("torch","transformers","accelerate"):
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",m])
import torch,numpy as np,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
TEST="565";MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SEED=552552;PANEL_SEED=550550;MAX_NEW=16;DEVICE=torch.device("cuda");DTYPE=torch.bfloat16
XRAY_CASES=[0,1,2,3];VALIDATION_CASES=list(range(8));SLOTS=["FIRST","MIDDLE","LAST"];TOP_CUTS=3;EPS=1e-8
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_grad_enabled(False);T0=time.perf_counter()
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",s.casefold()).split())
def hit(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
def cos(a,b):
    x=a.float().reshape(-1);y=b.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(x.unsqueeze(0),y.unsqueeze(0)).item())
def rel(a,b):return float((a.float()-b.float()).norm().item()/(b.float().norm().item()+EPS))
def ratio(a,b):return float(a.float().norm().item()/(b.float().norm().item()+EPS))
print("="*132);print("TEST565 — AKBASCORE MAM · QWEN FULL-LAYER X-RAY");print("TEST564 BACKBONE PRESERVED · 28-LAYER DIAGNOSTICS → CAUSAL VALIDATION");print("="*132)
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=DTYPE).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=H//QH
assert (NL,H,QH,KVH,HD)==(28,3584,28,4,128),(NL,H,QH,KVH,HD)
print("GPU:",torch.cuda.get_device_name(0),"TORCH:",torch.__version__,"TRANSFORMERS:",transformers.__version__);print("ARCH:",NL,H,QH,KVH,HD)
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
    for typ,f,gold in facts:KEY[(wi,typ)]=len(CAR);CAR.append(dict(world=wi,type=typ,fact=f,gold=gold))
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
PANEL_SHA=sha(PANEL);assert PANEL_SHA=="69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45"
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
        return dict(h=state["h"],kv=clone(out.past_key_values)[:cut+1],pos=pos,body=n-P)
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
        return cp["kv"]+upper[cut+1:]
    finally:h.remove()
@torch.no_grad()
def append(memory,ci,cut,integrity=True):
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
        if integrity:
            for (ok,ov),(nk,nv) in zip(memory,result):
                if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError("APPEND-ONLY BITWISE FAIL")
        if any(k.shape[-2]!=oldn+q for k,v in result):raise RuntimeError("Cache length mismatch")
        return result
    finally:h.remove()
@torch.no_grad()
def answer(memory,q,return_margin=False,gold=None):
    cache=cache_build(memory);n=memory[0][0].shape[-2];ids=tok(suffix(q),add_special_tokens=False).input_ids
    x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,n+len(ids),device=DEVICE)
    out=model(input_ids=x,past_key_values=cache,attention_mask=torch.ones((1,n+len(ids)),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
    cache=out.past_key_values;logits=out.logits[:,-1,:].float();next_id=logits.argmax(-1,keepdim=True);margin=None
    if gold is not None:
        gids=tok(" "+gold,add_special_tokens=False).input_ids
        if gids:
            gid=gids[0];best_other=logits.clone();best_other[0,gid]=-float("inf")
            margin=float((logits[0,gid]-best_other.max()).item())
    gen=[]
    for _ in range(MAX_NEW):
        t=int(next_id.item())
        if t in (tok.eos_token_id,tok.convert_tokens_to_ids("<|im_end|>")):break
        gen.append(t);n=cache.get_seq_length();pos=torch.tensor([n],device=DEVICE)
        out=model(input_ids=next_id,past_key_values=cache,attention_mask=torch.ones((1,n+1),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        cache=out.past_key_values;next_id=out.logits[:,-1,:].argmax(-1,keepdim=True)
    result=tok.decode(gen,skip_special_tokens=True).strip()
    return (result,margin) if return_margin else result
@torch.no_grad()
def full_reference(ci):
    ids=PIDS+CAR[ci]["body"];n=len(ids);x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,device=DEVICE);hs={}
    hooks=[]
    try:
        for li,layer in enumerate(model.model.layers):
            def cap(module,args,out,li=li):hs[li]=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
            hooks.append(layer.register_forward_hook(cap))
        out=model(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        return hs,clone(out.past_key_values),pos
    finally:
        for h in hooks:h.remove()
@torch.no_grad()
def full_joint(keys):
    ids=list(PIDS)
    for ci in keys:ids+=CAR[ci]["body"]
    n=len(ids);x=torch.tensor([ids],device=DEVICE);pos=torch.arange(n,device=DEVICE);hs={};hooks=[]
    try:
        for li,layer in enumerate(model.model.layers):
            def cap(module,args,out,li=li):hs[li]=(out[0] if isinstance(out,(tuple,list)) else out).detach().clone()
            hooks.append(layer.register_forward_hook(cap))
        out=model(input_ids=x,attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
        return hs,clone(out.past_key_values),pos
    finally:
        for h in hooks:h.remove()
def xray_case(wi,slot):
    z=PANEL[wi];keys=ordered(z,slot);target=z["target"];start=P+sum(len(CAR[ci]["body"]) for ci in keys[:keys.index(target)]);end=start+len(CAR[target]["body"])
    local_h,local_kv,local_pos=full_reference(target);joint_h,joint_kv,joint_pos=full_joint(keys)
    rows=[]
    for li in range(NL):
        a=local_h[li][:,P:,:];b=joint_h[li][:,start:end,:]
        lk,lv=local_kv[li];jk,jv=joint_kv[li];old=local_pos[P:];new=joint_pos[start:end]
        rk=rephase(lk[:,:,P:,:],old,new);jks=jk[:,:,start:end,:];jvs=jv[:,:,start:end,:];lvs=lv[:,:,P:,:]
        rows.append(dict(layer=li,h_cos=cos(a,b),h_rel=rel(a,b),h_ratio=ratio(a,b),k_cos=cos(rk,jks),k_rel=rel(rk,jks),v_cos=cos(lvs,jvs),v_rel=rel(lvs,jvs)))
    del local_h,local_kv,joint_h,joint_kv
    return rows
print("\n[1/4] FULL-LAYER X-RAY — 28 LAYERS")
XR=[]
for wi in XRAY_CASES:
    for slot in SLOTS:
        rows=xray_case(wi,slot);XR.append(dict(case=wi,slot=slot,rows=rows))
        print(f"XRAY CASE={wi:02d} SLOT={slot:6s} | L0 Hcos={rows[0]['h_cos']:.5f} L6 Hcos={rows[6]['h_cos']:.5f} L14 Hcos={rows[14]['h_cos']:.5f} L27 Hcos={rows[27]['h_cos']:.5f}",flush=True)
print("\n[2/4] LAYER STATISTICS")
STATS=[]
for li in range(NL):
    rr=[x["rows"][li] for x in XR]
    d=dict(layer=li)
    for k in ("h_cos","h_rel","h_ratio","k_cos","k_rel","v_cos","v_rel"):d[k]=float(np.mean([r[k] for r in rr]))
    STATS.append(d)
for d in STATS:
    print(f"L{d['layer']:02d} | Hcos={d['h_cos']:.6f} Hrel={d['h_rel']:.6f} Hnorm={d['h_ratio']:.4f} | Kcos={d['k_cos']:.6f} Krel={d['k_rel']:.5f} | Vcos={d['v_cos']:.6f} Vrel={d['v_rel']:.5f}")
# The diagnostic score is a candidate prioritizer, not a claim of causal optimality.
# Favor a low H-relative distortion, stable K/V, and sufficient remaining upper layers.
CANDIDATES=list(range(1,NL-2))
def priority(d):
    depth=(NL-1-d["layer"])/(NL-1)
    return (d["h_rel"]+0.25*d["k_rel"]+0.25*d["v_rel"])/(0.25+0.75*depth)
RANKED=sorted((d for d in STATS if d["layer"] in CANDIDATES),key=lambda d:(priority(d),d["layer"]))
SELECTED=[d["layer"] for d in RANKED[:TOP_CUTS]]
if 6 not in SELECTED:SELECTED.append(6)
SELECTED=sorted(set(SELECTED))
print("\nXRAY PRIORITY (lower is better; heuristic only):")
for d in RANKED[:10]:print(f"L{d['layer']:02d} priority={priority(d):.6f} Hrel={d['h_rel']:.6f} Krel={d['k_rel']:.6f} Vrel={d['v_rel']:.6f}")
print("CAUSAL CUT CANDIDATES:",SELECTED)
print("\n[3/4] CAUSAL CONSOLIDATION VALIDATION")
RESULTS=[]
for cut in SELECTED:
    t=time.perf_counter();rows=[];error=None
    try:
        for wi in VALIDATION_CASES:
            z=PANEL[wi];typ=CAR[z["target"]]["type"];q=W[wi][typ];gold=CAR[z["target"]]["gold"]
            for slot in SLOTS:
                keys=ordered(z,slot);memory=init(keys[0],cut)
                for ci in keys[1:]:memory=append(memory,ci,cut)
                a,margin=answer(memory,q,True,gold);ok=hit(a,gold)
                rows.append(dict(case=wi,slot=slot,answer=a,gold=gold,ok=ok,margin=margin))
                print(f"CUT={cut:02d} CASE={wi:02d} {slot:6s} {'PASS' if ok else 'FAIL'} | margin={margin:+.3f} | {a!r}",flush=True)
                del memory
    except Exception as e:error=repr(e);print(f"CUT={cut} ERROR:",error,flush=True)
    score=sum(int(r["ok"]) for r in rows);margins=[r["margin"] for r in rows if r["margin"] is not None]
    RESULTS.append(dict(cut=cut,score=score,total=len(rows),complete=len(rows)==len(VALIDATION_CASES)*3 and error is None,mean_margin=float(np.mean(margins)) if margins else None,error=error,seconds=round(time.perf_counter()-t,2)))
    print(f"CUT={cut:02d} SCORE={score}/24 MEAN_FIRST_TOKEN_MARGIN={RESULTS[-1]['mean_margin']} TIME={RESULTS[-1]['seconds']}s",flush=True)
    torch.cuda.empty_cache();gc.collect()
print("\n[4/4] FINAL X-RAY RESEARCH RECORD")
valid=[r for r in RESULTS if r["complete"]]
best=sorted(valid,key=lambda r:(-r["score"],-(r["mean_margin"] if r["mean_margin"] is not None else -1e9),r["cut"]))[0] if valid else None
print("="*132);print("TEST565 — FINAL");print("="*132)
print("MODEL:",MODEL_ID,"ARCH:",(NL,H,QH,KVH,HD));print("PANEL SHA:",PANEL_SHA);print("XRAY CASES:",XRAY_CASES,"XRAY ROWS:",len(XR)*NL)
print("TOP XRAY LAYERS:",[(d["layer"],round(priority(d),6)) for d in RANKED[:10]])
for r in RESULTS:print(f"CUT={r['cut']:02d} SCORE={r['score']:02d}/24 MARGIN={r['mean_margin']} COMPLETE={r['complete']} ERROR={r['error']}")
print("SELECTED CUT:",best["cut"] if best else "NONE")
print("WEIGHT SENTINEL:","PASS" if sentinel()==S0 else "FAIL");print("TRAINABLE:",sum(int(p.requires_grad) for p in model.parameters()))
print("TOTAL SECONDS:",round(time.perf_counter()-T0,2))
print("DECISION:","CANDIDATE READY FOR INDEPENDENT 72-QUESTION VALIDATION" if best and best["score"]==24 else "FURTHER CAUSAL CALIBRATION REQUIRED")
print("="*132)
assert sentinel()==S0,"WEIGHT SENTINEL FAIL"
