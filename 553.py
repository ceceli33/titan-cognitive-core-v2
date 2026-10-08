# TEST552R-OWN — AKBASCORE MAM · OWNERSHIP GATE + TEST552 RoPE REPAIR
# TEST548 SINGLE-CARTRIDGE CONTRACT PRESERVED
# PART-A: N=1 OWNERSHIP — CORRECT / SAME-SUBJECT-WRONG-RELATION / FOREIGN-SUBJECT
#         TEXT + NUMERICAL PKV
# PART-B: MODEL-NATIVE rotary_emb RoPE VALIDATION + REPAIRED COT-SHR
# NO RETRIEVAL · NO ROUTER · NO SCORER · NO TOP-K · NO THRESHOLD · NO TRAINING
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,math
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache=None

TEST="552R-OWN";SEED=552552;PANEL_SEED=550550
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT548="68e1594d7095cf7828286572a2ae6250bfcd5a7e3612e6db9c15a429bc4a8a87"
PARENT551="69a6c6275298ee028ad9399dca869128989778fd34c5ad450ec9dc0b6c49c08b"
PARENT552="603d858510fb438defbb5427c391264374387f33bc306c18d58e4e1161622f6e"
PANEL552="0b5cccba3fef19842d277ad1c98edbad29fe0fe26814493331f0889203c9f5b9"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16;K=5
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*186)
print("TEST552R-OWN — AKBASCORE MAM · OWNERSHIP GATE + TEST552 RoPE REPAIR")
print("TEST548 SINGLE-CARTRIDGE CONTRACT · TEXT↔PKV OWNERSHIP · MODEL-NATIVE rotary_emb · REPAIRED COT-SHR")
print("NO RETRIEVAL · NO ROUTER · NO SCORER · NO TOP-K · NO THRESHOLD · NO TRAINING")
print("="*186);T0=time.perf_counter()

print("\n[1/11] Donmuş Mistral...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")
if DynamicCache is None:raise RuntimeError("DynamicCache import edilemedi.")
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | Torch {torch.__version__} | Transformers {transformers.__version__}")
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | {DTYPE} | SDPA | trainable=0")
rope_cfg=getattr(cfg,"rope_parameters",None)
print("      config.rope_theta      :",getattr(cfg,"rope_theta",None))
print("      config.rope_parameters :",rope_cfg)

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
("Zorvan","Melket","Dravel","Oakhaven","Pelnor"),("Kelvar","Nareth","Solven","Branik","Tarsen"),
("Tarev","Luneth","Varos","Cedran","Mireth"),("Belnor","Arven","Dorel","Kesmar","Falven"),
("Ravik","Selora","Terven","Maldor","Nerik"),("Nemor","Calven","Istral","Pareth","Dovren"),
("Darsen","Velora","Keldin","Orvek","Sarnel"),("Feron","Talven","Merith","Sovran","Belvik"),
("Larev","Nerith","Calder","Veyron","Tormek"),("Torven","Elsar","Marvek","Dorin","Kaleth"),
("Selnor","Kareth","Valen","Ordan","Mervek"),("Mirev","Taldor","Neris","Kelmar","Sorvik"),
("Varen","Solith","Deran","Malvek","Cordan"),("Kelor","Ardin","Velmar","Toren","Narell"),
("Narev","Belith","Corven","Sareth","Dorvik"),("Dervan","Mirel","Talvek","Orsen","Kelron"),
("Calnor","Verith","Naldor","Seren","Parvek"),("Parel","Dorven","Kelith","Maros","Tervik"),
("Sorven","Tarell","Vindor","Nelmar","Calrek"),("Barel","Corith","Laven","Derik","Solmar"),
("Ralen","Mervor","Talith","Kesven","Noreth"),("Norel","Valdor","Serith","Calvenor","Darvek"),
("Tervan","Orel","Mardin","Velos","Karven"),("Karev","Solen","Dareth","Mirven","Talrek")]
W=[]
for i,(s,current,near,former,role_target) in enumerate(BASE):
    facts=[("CURRENT",f"The current capital of {s} is {current}.",current),
           ("NEAR",f"The largest city of {s} is {near}.",near),
           ("FORMER",f"The former capital of {s} was {former}.",former),
           ("ROLE",f"The current capital of {role_target} is {s}.",s)]
    queries=[("CURRENT",f"What is the current capital of {s}?",current),
             ("FORMER",f"What was the former capital of {s}?",former),
             ("NEAR",f"What is the largest city of {s}?",near),
             ("ROLE",f"What is the current capital of {role_target}?",s)]
    W.append({"id":i,"subject":s,"facts":facts,"queries":queries})

SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def source_prefix(fact):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{fact}\n\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
def norm(s):
    s=s.casefold().strip();s=re.sub(r"[^\w\s-]"," ",s,flags=re.UNICODE);return " ".join(s.split())
def hit(out,gold):return re.search(r"(?<!\w)"+re.escape(norm(gold))+r"(?!\w)",norm(out)) is not None
ALL_VALUES=set()
for w in W:
    for _,_,g in w["facts"]:ALL_VALUES.add(norm(g))

LOCK={"test":TEST,"parent548":PARENT548,"parent551":PARENT551,"parent552":PARENT552,
      "model":MODEL_ID,"seed":SEED,"panel_seed":PANEL_SEED,
      "ownership":{"correct":1,"same_subject_wrong_relation":3,"foreign_subject":4,"text":True,"pkv":True},
      "rope":"MODEL_NATIVE_ROTARY_EMB","cot_shr_repair":True,"k":K,
      "retrieval":False,"router":False,"scorer":False,"topk":False,"threshold":False,
      "training":False,"lora":False,"optimizer":False}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      LOCK SHA:",LOCK_SHA)

def cache_layers(pkv):
    if hasattr(pkv,"layers"):
        out=[]
        for layer in pkv.layers:
            k=getattr(layer,"keys",None);v=getattr(layer,"values",None)
            if k is None:k=getattr(layer,"key_cache",None)
            if v is None:v=getattr(layer,"value_cache",None)
            if k is None or v is None:raise RuntimeError(f"DynamicCache layer API tanınmadı: {type(layer)}")
            out.append((k,v))
        if out:return tuple(out)
    if hasattr(pkv,"key_cache") and hasattr(pkv,"value_cache"):return tuple(zip(pkv.key_cache,pkv.value_cache))
    if hasattr(pkv,"to_legacy_cache"):
        p=pkv.to_legacy_cache()
        if isinstance(p,(tuple,list)):return tuple(p)
    if isinstance(pkv,(tuple,list)):return tuple(pkv)
    raise RuntimeError(f"Desteklenmeyen cache tipi: {type(pkv)}")

def cache_len(pkv):
    if hasattr(pkv,"get_seq_length"):
        try:return int(pkv.get_seq_length())
        except TypeError:return int(pkv.get_seq_length(0))
    return int(cache_layers(pkv)[0][0].shape[-2])

def cache_audit(pkv,expected=None):
    L=cache_layers(pkv)
    if len(L)!=NL:raise RuntimeError(f"PKV layer sayısı {len(L)} != {NL}")
    lens=[]
    for li,(k,v) in enumerate(L):
        if k.ndim!=4 or v.ndim!=4 or tuple(k.shape)!=tuple(v.shape):raise RuntimeError(f"L{li} K/V shape hatası: K={tuple(k.shape)} V={tuple(v.shape)}")
        if k.shape[0]!=1 or k.shape[1]!=KVH or k.shape[-1]!=HD:raise RuntimeError(f"L{li} shape {tuple(k.shape)}")
        lens.append(int(k.shape[-2]))
    if len(set(lens))!=1:raise RuntimeError("Katman cache uzunlukları farklı.")
    n=lens[0]
    if expected is not None and n!=expected:raise RuntimeError(f"Cache/token uzunluk uyuşmazlığı {n}!={expected}")
    return n

def clone_layers(pkv):return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))
def new_cache_from_layers(layers):
    c=DynamicCache()
    for li,(k,v) in enumerate(layers):
        try:c.update(k,v,li)
        except TypeError:c.update(k,v,li,{})
    cache_audit(c,int(layers[0][0].shape[-2]));return c
def raw_bytes(x):
    x=x.detach().contiguous().cpu()
    if x.dtype==torch.bfloat16:x=x.view(torch.uint16)
    return x.numpy().tobytes()
def tensor_sha(layers):
    h=hashlib.sha256()
    for k,v in layers:h.update(raw_bytes(k));h.update(raw_bytes(v))
    return h.hexdigest()

@torch.no_grad()
def forge(fact):
    x=tok(source_prefix(fact),return_tensors="pt",add_special_tokens=False).to(DEVICE);n=int(x.input_ids.shape[1])
    pos=torch.arange(n,dtype=torch.long,device=DEVICE).unsqueeze(0)
    out=model(**x,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
    layers=clone_layers(out.past_key_values);cache_audit(out.past_key_values,n)
    del out,x
    return layers,n

@torch.no_grad()
def continue_from_cache(cache,nxt):
    gen=[];eos=tok.eos_token_id
    eos=set() if eos is None else ({int(eos)} if not isinstance(eos,(list,tuple,set)) else set(map(int,eos)))
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        past=cache_len(cache);att=torch.ones((1,past+1),dtype=torch.long,device=DEVICE);cp=torch.tensor([past],dtype=torch.long,device=DEVICE)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=att,cache_position=cp,position_ids=cp.unsqueeze(0),use_cache=True,return_dict=True)
        cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);del out
    return "" if not gen else tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()

@torch.no_grad()
def text_answer(fact,q):
    s=source_prefix(fact)+query_suffix(q);x=tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE);n=int(x.input_ids.shape[1])
    pos=torch.arange(n,dtype=torch.long,device=DEVICE).unsqueeze(0)
    out=model(**x,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
    nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);ans=continue_from_cache(out.past_key_values,nxt)
    del out,x
    return ans

@torch.no_grad()
def pkv_answer(layers,q):
    pkv=new_cache_from_layers(tuple((k.clone(),v.clone()) for k,v in layers));past=cache_len(pkv)
    x=tok(query_suffix(q),return_tensors="pt",add_special_tokens=False).to(DEVICE);qlen=int(x.input_ids.shape[1])
    att=torch.ones((1,past+qlen),dtype=torch.long,device=DEVICE);cp=torch.arange(past,past+qlen,dtype=torch.long,device=DEVICE)
    out=model(input_ids=x.input_ids,past_key_values=pkv,attention_mask=att,cache_position=cp,position_ids=cp.unsqueeze(0),use_cache=True,return_dict=True)
    nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);ans=continue_from_cache(out.past_key_values,nxt)
    del out,x,pkv
    return ans

def rotate_half(x):
    x1=x[...,:x.shape[-1]//2];x2=x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

@torch.no_grad()
def rope_cos_sin(position_ids,dtype=torch.float32):
    rotary=model.model.rotary_emb
    pos=position_ids.reshape(1,-1).to(device=DEVICE,dtype=torch.long)
    dummy=torch.zeros((1,pos.shape[1],H),device=DEVICE,dtype=DTYPE)
    try:cos,sin=rotary(dummy,pos)
    except TypeError:cos,sin=rotary(dummy,position_ids=pos)
    if cos.ndim==2:cos=cos.unsqueeze(0);sin=sin.unsqueeze(0)
    if cos.ndim==3:cos=cos.unsqueeze(1);sin=sin.unsqueeze(1)
    return cos.to(dtype=dtype),sin.to(dtype=dtype)

@torch.no_grad()
def rope_rebase_k(k,old_start,new_start):
    L=int(k.shape[-2]);old=torch.arange(old_start,old_start+L,dtype=torch.long,device=DEVICE);new=torch.arange(new_start,new_start+L,dtype=torch.long,device=DEVICE)
    co,so=rope_cos_sin(old);cn,sn=rope_cos_sin(new);kf=k.float()
    raw=kf*co-rotate_half(kf)*so
    return (raw*cn+rotate_half(raw)*sn).to(k.dtype)

@torch.no_grad()
def direct_forge_at_offset(fact,offset):
    x=tok(source_prefix(fact),return_tensors="pt",add_special_tokens=False).to(DEVICE);n=int(x.input_ids.shape[1])
    pos=torch.arange(offset,offset+n,dtype=torch.long,device=DEVICE).unsqueeze(0)
    out=model(**x,position_ids=pos,cache_position=torch.arange(n,dtype=torch.long,device=DEVICE),use_cache=True,return_dict=True)
    layers=clone_layers(out.past_key_values);del out,x
    return layers,n

print("\n[2/11] TEST548 sözleşmesiyle 96 bağımsız cartridge forge...")
CAR=[];idx=0
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/24 {w['subject']}")
    qmap={x[0]:x for x in w["queries"]}
    for typ,fact,gold in w["facts"]:
        layers,n=forge(fact)
        CAR.append({"idx":idx,"world":wi,"type":typ,"fact":fact,"gold":gold,"query":qmap[typ][1],"layers":layers,"n":n,"sha":tensor_sha(layers)})
        idx+=1
print("      Cartridges=96 | min/mean/max=",min(c["n"] for c in CAR),f"{np.mean([c['n'] for c in CAR]):.2f}",max(c["n"] for c in CAR))

print("\n[3/11] G-R — model-native rotary_emb rebase doğrulaması...")
GR=[]
probe_car=CAR[0]
for d in [1,2,7,31]:
    direct,n=direct_forge_at_offset(probe_car["fact"],d);base=probe_car["layers"]
    kd=[];vd=[]
    for li in range(NL):
        kr=rope_rebase_k(base[li][0],0,d)
        kd.append(float((kr.float()-direct[li][0].float()).abs().max().item()))
        vd.append(float((base[li][1].float()-direct[li][1].float()).abs().max().item()))
    mk=max(kd);mv=max(vd);GR.append({"delta":d,"kmax":mk,"vmax":mv})
    print(f"      δ={d:2d} | K maxΔ={mk:.8f} | V maxΔ={mv:.8f}")
GR_OK=all(x["kmax"]<=0.5 and x["vmax"]<=0.5 for x in GR)
print("      G-R:", "PASS" if GR_OK else "FAIL")

print("\n[4/11] PART-A — N=1 OWNERSHIP panel hazırlanıyor...")
# Her dünya için bir sorgu tipi seçilir: W00 CURRENT, W01 FORMER, W02 NEAR, W03 ROLE, sonra döngü.
# Her hedef: correct + aynı dünyanın diğer 3 ilişkisi + deterministik 4 yabancı dünya kartuşu.
OWN_PANEL=[];rng=random.Random(SEED)
for wi,w in enumerate(W):
    typ=["CURRENT","FORMER","NEAR","ROLE"][wi%4]
    target=next(c for c in CAR if c["world"]==wi and c["type"]==typ)
    same=[c for c in CAR if c["world"]==wi and c["idx"]!=target["idx"]]
    foreign_worlds=[x for x in range(24) if x!=wi];rng.shuffle(foreign_worlds);foreign=[]
    for fw in foreign_worlds[:4]:
        fcands=[c for c in CAR if c["world"]==fw]
        foreign.append(fcands[(wi+fw)%4])
    OWN_PANEL.append({"world":wi,"target":target["idx"],"same":[c["idx"] for c in same],"foreign":[c["idx"] for c in foreign]})
OWN_SHA=hashlib.sha256(json.dumps(OWN_PANEL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Cases=24 | per case: 1 correct + 3 same-subject/wrong-relation + 4 foreign-subject")
print("      OWNERSHIP PANEL SHA:",OWN_SHA)

def classify_mismatch(out,car):
    no=norm(out);foreign_value=hit(out,car["gold"])
    if foreign_value:return "FOREIGN_VALUE"
    refusal_terms=["not provided","not given","cannot determine","cannot be determined","insufficient","unknown","not specified","no information","does not provide","information provided does not"]
    if any(x in no for x in refusal_terms):return "REFUSAL"
    return "OTHER"

print("\n[5/11] PART-A — TEXT + PKV ownership çalışıyor...")
OWNRES=[]
for pi,z in enumerate(OWN_PANEL):
    target=CAR[z["target"]];q=target["query"]
    print(f"\n      CASE {pi:02d} W{target['world']:02d} {target['type']:8s} query={q!r} gold={target['gold']}")
    groups=[("CORRECT",[z["target"]]),("SAME_REL_MISMATCH",z["same"]),("FOREIGN_SUBJECT",z["foreign"])]
    for group,inds in groups:
        for ci in inds:
            c=CAR[ci]
            ta=text_answer(c["fact"],q);pa=pkv_answer(c["layers"],q)
            if group=="CORRECT":
                tc="CORRECT" if hit(ta,target["gold"]) else "WRONG";pc="CORRECT" if hit(pa,target["gold"]) else "WRONG"
            else:
                tc=classify_mismatch(ta,c);pc=classify_mismatch(pa,c)
            OWNRES.append({"case":pi,"group":group,"target":z["target"],"car":ci,"text":ta,"pkv":pa,"text_class":tc,"pkv_class":pc})
            print(f"      {group:17s} C{ci:02d} {c['type']:8s} | TEXT={tc:13s} {ta!r} | PKV={pc:13s} {pa!r}")

print("\n[6/11] Ownership agregasyonu...")
def own_summary(modality):
    out={}
    ck=modality+"_class"
    for g in ["CORRECT","SAME_REL_MISMATCH","FOREIGN_SUBJECT"]:
        rows=[x for x in OWNRES if x["group"]==g];cnt={}
        for x in rows:cnt[x[ck]]=cnt.get(x[ck],0)+1
        out[g]={"n":len(rows),"counts":cnt}
    return out
OS_TEXT=own_summary("text");OS_PKV=own_summary("pkv")
for name,S in [("TEXT",OS_TEXT),("PKV",OS_PKV)]:
    print(f"\n      {name}")
    for g,v in S.items():print(f"      {g:17s} n={v['n']:3d} | {v['counts']}")
pkv_mismatch=[x for x in OWNRES if x["group"]!="CORRECT"]
pkv_foreign=sum(x["pkv_class"]=="FOREIGN_VALUE" for x in pkv_mismatch)
pkv_refusal=sum(x["pkv_class"]=="REFUSAL" for x in pkv_mismatch)
nm=len(pkv_mismatch);foreign_rate=pkv_foreign/nm if nm else 0.;refusal_rate=pkv_refusal/nm if nm else 0.
same_rows=[x for x in OWNRES if x["group"]=="SAME_REL_MISMATCH"];foreign_rows=[x for x in OWNRES if x["group"]=="FOREIGN_SUBJECT"]
same_foreign=sum(x["pkv_class"]=="FOREIGN_VALUE" for x in same_rows)/len(same_rows)
same_refusal=sum(x["pkv_class"]=="REFUSAL" for x in same_rows)/len(same_rows)
diff_foreign=sum(x["pkv_class"]=="FOREIGN_VALUE" for x in foreign_rows)/len(foreign_rows)
diff_refusal=sum(x["pkv_class"]=="REFUSAL" for x in foreign_rows)/len(foreign_rows)
print(f"\n      PKV mismatch ALL foreign-value={pkv_foreign}/{nm}={foreign_rate:.6f} | refusal={pkv_refusal}/{nm}={refusal_rate:.6f}")
print(f"      SAME SUBJECT / wrong relation foreign-value={same_foreign:.6f} | refusal={same_refusal:.6f}")
print(f"      FOREIGN SUBJECT foreign-value={diff_foreign:.6f} | refusal={diff_refusal:.6f}")

print("\n[7/11] PART-B — TEST552 panelini aynı tohumla yeniden kur...")
# TEST552 panel convention: 24 targets, one per world, cycling relation; four distractors from other worlds.
# This block is sealed by expected PANEL552 SHA. If reconstruction differs, repaired COT-SHR is NOT interpreted.
rng2=random.Random(PANEL_SEED);pool=list(range(96));rng2.shuffle(pool)
PANEL=[];cursor=0
for wi in range(24):
    typ=["CURRENT","FORMER","NEAR","ROLE"][wi%4]
    target=next(c["idx"] for c in CAR if c["world"]==wi and c["type"]==typ)
    others=[]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]["world"]==wi or any(CAR[x]["world"]==CAR[ci]["world"] for x in others):continue
        others.append(ci)
    PANEL.append({"case":wi,"target":target,"others":others})
PANEL_SHA=hashlib.sha256(json.dumps(PANEL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Reconstructed PANEL SHA:",PANEL_SHA)
print("      Expected TEST552 SHA    :",PANEL552)
PANEL_MATCH=PANEL_SHA==PANEL552
print("      PANEL MATCH             :","PASS" if PANEL_MATCH else "FAIL — COT-SHR result descriptive only")

# Canonical shared prefix = common 27-token prefix found in TEST552.
prefix_text=f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n"
prefix_ids=tok(prefix_text,return_tensors="pt",add_special_tokens=False).input_ids[0].to(DEVICE)
P=int(prefix_ids.numel())
print("      Canonical prefix tokens :",P)

@torch.no_grad()
def forge_prefix():
    ids=prefix_ids.unsqueeze(0);pos=torch.arange(P,dtype=torch.long,device=DEVICE).unsqueeze(0)
    out=model(input_ids=ids,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
    L=clone_layers(out.past_key_values);del out
    return L
PREFIX=forge_prefix()

@torch.no_grad()
def forge_body_from_prefix(fact):
    full=tok(source_prefix(fact),return_tensors="pt",add_special_tokens=False).input_ids[0].to(DEVICE)
    if int(full.numel())<P or not torch.equal(full[:P],prefix_ids):raise RuntimeError("Canonical prefix token eşleşmesi bozuldu.")
    body=full[P:];L=int(body.numel());pkv=new_cache_from_layers(tuple((k.clone(),v.clone()) for k,v in PREFIX))
    cp=torch.arange(P,P+L,dtype=torch.long,device=DEVICE);att=torch.ones((1,P+L),dtype=torch.long,device=DEVICE)
    out=model(input_ids=body.unsqueeze(0),past_key_values=pkv,attention_mask=att,cache_position=cp,position_ids=cp.unsqueeze(0),use_cache=True,return_dict=True)
    allL=clone_layers(out.past_key_values);del out,pkv
    bodyL=tuple((k[:,:,P:,:].detach().clone(),v[:,:,P:,:].detach().clone()) for k,v in allL)
    return bodyL,L

print("\n[8/11] Repaired COT-SHR için shared-prefix body bankaları...")
BODY=[]
for i,c in enumerate(CAR):
    b,n=forge_body_from_prefix(c["fact"]);BODY.append({"idx":i,"layers":b,"n":n})
print("      Bodies=96 | min/mean/max=",min(x["n"] for x in BODY),f"{np.mean([x['n'] for x in BODY]):.2f}",max(x["n"] for x in BODY))

@torch.no_grad()
def cot_shr_layers(indices):
    lens=[BODY[i]["n"] for i in indices];Lstar=max(lens);per=[[] for _ in range(NL)]
    # prefix once at physical front; already at positions 0..P-1
    for li in range(NL):per[li].append((PREFIX[li][0].clone(),PREFIX[li][1].clone()))
    # all bodies get co-terminal virtual positions: P+Lstar-L_i ... P+Lstar-1
    for ci in indices:
        b=BODY[ci];new_start=P+Lstar-b["n"]
        for li,(k,v) in enumerate(b["layers"]):
            # body was forged at original virtual positions P..P+L_i-1
            kr=rope_rebase_k(k,P,new_start)
            per[li].append((kr,v.clone()))
    layers=tuple((torch.cat([x[0] for x in per[li]],dim=-2),torch.cat([x[1] for x in per[li]],dim=-2)) for li in range(NL))
    physical=int(layers[0][0].shape[-2]);virtual_q=P+Lstar
    return layers,physical,virtual_q

@torch.no_grad()
def answer_virtual_cache(layers,virtual_q,q):
    pkv=new_cache_from_layers(tuple((k.clone(),v.clone()) for k,v in layers));physical=cache_len(pkv)
    x=tok(query_suffix(q),return_tensors="pt",add_special_tokens=False).to(DEVICE);qlen=int(x.input_ids.shape[1])
    # physical cache index and RoPE position are deliberately separated.
    cp=torch.arange(physical,physical+qlen,dtype=torch.long,device=DEVICE)
    pos=torch.arange(virtual_q,virtual_q+qlen,dtype=torch.long,device=DEVICE).unsqueeze(0)
    att=torch.ones((1,physical+qlen),dtype=torch.long,device=DEVICE)
    out=model(input_ids=x.input_ids,past_key_values=pkv,attention_mask=att,cache_position=cp,position_ids=pos,use_cache=True,return_dict=True)
    nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    # generation must continue in virtual RoPE coordinates while physical cache grows.
    gen=[];eos=tok.eos_token_id
    eos=set() if eos is None else ({int(eos)} if not isinstance(eos,(list,tuple,set)) else set(map(int,eos)))
    cache=out.past_key_values;next_virtual=virtual_q+qlen
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        phys=cache_len(cache);att2=torch.ones((1,phys+1),dtype=torch.long,device=DEVICE)
        cp2=torch.tensor([phys],dtype=torch.long,device=DEVICE);pos2=torch.tensor([[next_virtual]],dtype=torch.long,device=DEVICE)
        step=model(input_ids=nxt,past_key_values=cache,attention_mask=att2,cache_position=cp2,position_ids=pos2,use_cache=True,return_dict=True)
        cache=step.past_key_values;nxt=step.logits[:,-1,:].argmax(-1,keepdim=True);next_virtual+=1;del step
    ans="" if not gen else tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()
    del out,x,pkv
    return ans

print("\n[9/11] Repaired COT-SHR — FIRST/MIDDLE/LAST...")
COTRES=[]
for z in PANEL:
    target=CAR[z["target"]];o=z["others"]
    orders={"FIRST":[z["target"],o[0],o[1],o[2],o[3]],
            "MIDDLE":[o[0],o[1],z["target"],o[2],o[3]],
            "LAST":[o[0],o[1],o[2],o[3],z["target"]]}
    row={"case":z["case"],"target":z["target"],"gold":target["gold"]}
    outs={}
    for name,inds in orders.items():
        layers,physical,vq=cot_shr_layers(inds);ans=answer_virtual_cache(layers,vq,target["query"]);outs[name]=ans
        row[name]=ans;row[name+"_ok"]=hit(ans,target["gold"])
    COTRES.append(row)
    print(f"      {z['case']:02d} W{target['world']:02d} {target['type']:8s} gold={target['gold']:10s} | F={outs['FIRST']!r} | M={outs['MIDDLE']!r} | L={outs['LAST']!r}")
CF=sum(x["FIRST_ok"] for x in COTRES);CM=sum(x["MIDDLE_ok"] for x in COTRES);CL=sum(x["LAST_ok"] for x in COTRES)
CSAME=sum(norm(x["FIRST"])==norm(x["MIDDLE"])==norm(x["LAST"]) for x in COTRES)
print(f"\n      REPAIRED COT-SHR FIRST ={CF:02d}/24")
print(f"      REPAIRED COT-SHR MIDDLE={CM:02d}/24")
print(f"      REPAIRED COT-SHR LAST  ={CL:02d}/24")
print(f"      SAME F=M=L             ={CSAME:02d}/24")

print("\n[10/11] Mekanizma kararı...")
correct_pkvs=[x for x in OWNRES if x["group"]=="CORRECT"]
correct_ok=sum(x["pkv_class"]=="CORRECT" for x in correct_pkvs)
if foreign_rate>=0.50:
    OWN_DECISION="M-BLIND"
    NEXT="QSF_PRECONDITION_FAIL__DECISION_C"
elif refusal_rate>=0.80:
    OWN_DECISION="M-AWARE"
    NEXT="QSF_PRECONDITION_PASS__PROCEED_TEST553"
else:
    OWN_DECISION="PARTIAL"
    NEXT="QSF_PARTIAL_PRECONDITION__REPORT_SPLIT_AND_TEST_CAUTIOUSLY"
if PANEL_MATCH and min(CF,CM,CL)>=22:
    NEXT="REPAIRED_CSUA_SURVIVES__QSF_NOT_YET_NEEDED"
print(f"      Correct PKV reference       : {correct_ok}/24")
print(f"      Mismatch foreign-value rate : {foreign_rate:.6f}")
print(f"      Mismatch refusal rate       : {refusal_rate:.6f}")
print(f"      Ownership mechanism         : {OWN_DECISION}")
print(f"      Repaired COT-SHR min acc    : {min(CF,CM,CL)}/24")
print(f"      Next                        : {NEXT}")

print("\n[11/11] Anti-drift / sentinel / mühür...")
seal_ok=all(tensor_sha(c["layers"])==c["sha"] for c in CAR)
S1=sentinel();WOK=S0==S1;trainable=sum(int(p.requires_grad) for p in model.parameters())
print("      Source text replay at query : OWNERSHIP TEXT control only; PKV arm YOK")
print("      Candidate search/retrieval  : YOK")
print("      Cartridge ID as address     : YOK")
print("      Cosine / ANN / top-k        : YOK")
print("      Router / classifier         : YOK")
print("      Relevance threshold         : YOK")
print("      Learned/fitted scorer       : YOK")
print("      Training / LoRA / optimizer : YOK")
print("      Model-native rotary_emb      : EVET")
print("      Original cartridge seal     :","PASS" if seal_ok else "FAIL")
print("      Weight sentinel             :","PASS" if WOK else "FAIL")
print("      Trainable tensors           :",trainable)

FINAL={"test":TEST,"parent548":PARENT548,"parent551":PARENT551,"parent552":PARENT552,
       "lock_sha":LOCK_SHA,"ownership_panel_sha":OWN_SHA,"test552_expected_panel_sha":PANEL552,
       "reconstructed_panel_sha":PANEL_SHA,"panel_match":PANEL_MATCH,"gr":GR,"gr_ok":GR_OK,
       "ownership":{"correct_pkv":correct_ok,"mismatch_n":nm,"foreign_value":pkv_foreign,
                    "foreign_rate":foreign_rate,"refusal":pkv_refusal,"refusal_rate":refusal_rate,
                    "same_subject_wrong_relation":{"foreign_rate":same_foreign,"refusal_rate":same_refusal},
                    "foreign_subject":{"foreign_rate":diff_foreign,"refusal_rate":diff_refusal},
                    "decision":OWN_DECISION},
       "cot_shr_repaired":{"first":CF,"middle":CM,"last":CL,"same":CSAME},
       "cartridge_seal":seal_ok,"weight_sentinel":WOK,"trainable":trainable,"next":NEXT,
       "retrieval":False,"router":False,"scorer":False,"topk":False,"threshold":False,"training":False}
RESULT_SHA=hashlib.sha256(json.dumps(FINAL,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*186)
print("TEST552R-OWN SONUÇ — OWNERSHIP GATE + TEST552 RoPE REPAIR")
print("="*186)
print("MODEL                              :",MODEL_ID)
print("G-R MODEL-NATIVE RoPE             :","PASS" if GR_OK else "FAIL")
print("CORRECT PKV REFERENCE             :",f"{correct_ok}/24")
print("PKV MISMATCH N                    :",nm)
print("PKV FOREIGN-VALUE                 :",f"{pkv_foreign}/{nm} = {foreign_rate:.6f}")
print("PKV REFUSAL                       :",f"{pkv_refusal}/{nm} = {refusal_rate:.6f}")
print("SAME-SUBJECT WRONG-REL FOREIGN    :",f"{same_foreign:.6f}")
print("SAME-SUBJECT WRONG-REL REFUSAL    :",f"{same_refusal:.6f}")
print("FOREIGN-SUBJECT FOREIGN           :",f"{diff_foreign:.6f}")
print("FOREIGN-SUBJECT REFUSAL           :",f"{diff_refusal:.6f}")
print("OWNERSHIP DECISION                :",OWN_DECISION)
print("TEST552 PANEL MATCH               :","PASS" if PANEL_MATCH else "FAIL")
print("REPAIRED COT-SHR FIRST            :",f"{CF}/24")
print("REPAIRED COT-SHR MIDDLE           :",f"{CM}/24")
print("REPAIRED COT-SHR LAST             :",f"{CL}/24")
print("REPAIRED COT-SHR SAME F=M=L       :",f"{CSAME}/24")
print("CARTRIDGE SEAL                    :","PASS" if seal_ok else "FAIL")
print("WEIGHT SENTINEL                   :","PASS" if WOK else "FAIL")
print("TEST548 RESULT SHA                :",PARENT548)
print("TEST551 RESULT SHA                :",PARENT551)
print("TEST552 RESULT SHA                :",PARENT552)
print("TEST552R-OWN ÖN KİLİT SHA         :",LOCK_SHA)
print("OWNERSHIP PANEL SHA               :",OWN_SHA)
print("TEST552 EXPECTED PANEL SHA        :",PANEL552)
print("RECONSTRUCTED PANEL SHA           :",PANEL_SHA)
print("TEST552R-OWN RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                       :",f"{time.perf_counter()-T0:.2f}s")
print("SONRAKİ ADIM                      :",NEXT)
if not GR_OK:
    print("KARAR                              : ROTARY_VALIDATION_FAILURE — SONUÇ YORUMLANMAZ")
elif not WOK or not seal_ok:
    print("KARAR                              : PHYSICAL_VALIDITY_FAILURE")
elif OWN_DECISION=="M-BLIND":
    print("KARAR                              : OWNERSHIP_GATE_FAIL — QSF ÖNKOŞULU YOK")
elif OWN_DECISION=="M-AWARE":
    print("KARAR                              : OWNERSHIP_GATE_PASS — TEST553 QSF GEREKÇELİ")
else:
    print("KARAR                              : OWNERSHIP_PARTIAL — AYRIŞTIRILMIŞ SINIR KOŞULU")
print("="*186)
