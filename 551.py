# TEST551 — AKBASCORE MAM · LAST-CARTRIDGE CAPTURE X-RAY
# PAIRED FIRST↔LAST · T-BLOCK + P-RAW · LAYERWISE QK ATTENTION / V→O TRANSPORT / RESIDUAL TRAJECTORY
# PURPOSE: DOES CORRECT INDEPENDENT MEMORY BIND EARLY AND GET LOST, OR NEVER WIN?
# SDPA UNCHANGED · NO FIX · NO STEERING · NO RETRIEVAL · NO ROUTER · NO SCORER · NO TRAINING
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,math
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache=None

TEST="551";SEED=551551;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT548="68e1594d7095cf7828286572a2ae6250bfcd5a7e3612e6db9c15a429bc4a8a87"
PARENT549="f92d5ea0e4e29224ea0e72d163d4c855c18969be14622d151bf54f5fae674186"
PARENT550="64de77d2668a88793762b1ed1114e6183f6d921a3729553b9e8c37270bfb26c1"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16;K=5;NTRIAL=12
XRAY_LAYERS=list(range(32));FOCUS=list(range(7,20))
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*178)
print("TEST551 — AKBASCORE MAM · LAST-CARTRIDGE CAPTURE X-RAY")
print("PAIRED FIRST↔LAST · T-BLOCK + P-RAW · LAYERWISE QK ATTENTION / V→O TRANSPORT / RESIDUAL TRAJECTORY")
print("QUESTION: CORRECT MEMORY BINDS EARLY THEN GETS LOST, OR NEVER WINS?")
print("SDPA UNCHANGED · NO FIX · NO STEERING · NO RETRIEVAL · NO ROUTER · NO SCORER · NO TRAINING")
print("="*178);T0=time.perf_counter()

print("\n[1/10] Donmuş Mistral...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")
if QH%KVH!=0:raise RuntimeError("QH/KVH grup yapısı geçersiz.")
GROUP=QH//KVH
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | Torch {torch.__version__} | Transformers {transformers.__version__}")
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} GROUP={GROUP} | {DTYPE} | SDPA | trainable=0")

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
    facts=[("CURRENT",f"The current capital of {s} is {current}.",current),("NEAR",f"The largest city of {s} is {near}.",near),("FORMER",f"The former capital of {s} was {former}.",former),("ROLE",f"The current capital of {role_target} is {s}.",s)]
    queries=[("CURRENT",f"What is the current capital of {s}?",current),("FORMER",f"What was the former capital of {s}?",former),("NEAR",f"What is the largest city of {s}?",near),("ROLE",f"What is the current capital of {role_target}?",s)]
    W.append({"id":i,"subject":s,"facts":facts,"queries":queries})

SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def norm(s):
    s=s.casefold().strip();s=re.sub(r"[^\w\s-]"," ",s,flags=re.UNICODE);return " ".join(s.split())
def hit(out,gold):return re.search(r"(?<!\w)"+re.escape(norm(gold))+r"(?!\w)",norm(out)) is not None
def source_prefix(fact):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{fact}\n\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"

LOCK={"test":TEST,"parent548":PARENT548,"parent549":PARENT549,"parent550":PARENT550,"model":MODEL_ID,"seed":SEED,"k":K,"trials":NTRIAL,"paired_first_last":True,"arms":["T-BLOCK","P-RAW"],"xray":["PHYSICAL_CACHE_QK_ATTENTION","PHYSICAL_CACHE_VO_TRANSPORT","RESIDUAL_TRAJECTORY"],"sdpa_unchanged":True,"fix":False,"retrieval":False,"router":False,"scorer":False,"steering":False,"training":False,"lora":False,"optimizer":False}
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
    if len(set(lens))!=1:raise RuntimeError("Katman cache uzunlukları farklı")
    n=lens[0]
    if expected is not None and n!=expected:raise RuntimeError(f"Cache/token uzunluk uyuşmazlığı {n}!={expected}")
    return n

def new_cache_from_layers(layers):
    if DynamicCache is None:raise RuntimeError("DynamicCache import edilemedi.")
    c=DynamicCache()
    for li,(k,v) in enumerate(layers):
        try:c.update(k,v,li)
        except TypeError:c.update(k,v,li,{})
    cache_audit(c,int(layers[0][0].shape[-2]));return c

def clone_layers(pkv):return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))
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
    pkv=out.past_key_values;cache_audit(pkv,n);layers=clone_layers(pkv);del out,x,pkv
    return layers,n

def rotate_half(x):
    x1=x[...,:x.shape[-1]//2];x2=x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

@torch.no_grad()
def rope_cos_sin(position_ids,dtype=torch.float32):
    rotary=model.model.rotary_emb;pos=position_ids.reshape(1,-1).to(device=DEVICE,dtype=torch.long)
    dummy=torch.zeros((1,pos.shape[1],H),device=DEVICE,dtype=DTYPE)
    try:cos,sin=rotary(dummy,pos)
    except TypeError:cos,sin=rotary(dummy,position_ids=pos)
    if cos.ndim==2:cos=cos.unsqueeze(0);sin=sin.unsqueeze(0)
    if cos.ndim==3:cos=cos.unsqueeze(1);sin=sin.unsqueeze(1)
    return cos.to(dtype=dtype),sin.to(dtype=dtype)

@torch.no_grad()
def rope_rebase_k(k,old_start,new_start):
    L=int(k.shape[-2])
    old=torch.arange(old_start,old_start+L,dtype=torch.long,device=DEVICE)
    new=torch.arange(new_start,new_start+L,dtype=torch.long,device=DEVICE)
    co,so=rope_cos_sin(old);cn,sn=rope_cos_sin(new);kf=k.float()
    raw=kf*co-rotate_half(kf)*so
    return (raw*cn+rotate_half(raw)*sn).to(k.dtype)

@torch.no_grad()
def rebase_layers(layers,new_start):
    return tuple((rope_rebase_k(k,0,new_start),v.detach().clone()) for k,v in layers)

def compose_population(cars):
    off=0;per=[[] for _ in range(NL)];ranges=[]
    for c in cars:
        rb=rebase_layers(c["layers"],off);ranges.append((off,off+c["n"]))
        for l,(k,v) in enumerate(rb):per[l].append((k,v))
        off+=c["n"]
    layers=tuple((torch.cat([x[0] for x in per[l]],dim=-2),torch.cat([x[1] for x in per[l]],dim=-2)) for l in range(NL))
    return layers,ranges,off

@torch.no_grad()
def continue_from_cache(cache,nxt):
    gen=[];eos=tok.eos_token_id
    eos=set() if eos is None else ({int(eos)} if not isinstance(eos,(list,tuple,set)) else set(map(int,eos)))
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        past=cache_len(cache);att=torch.ones((1,past+1),dtype=torch.long,device=DEVICE);cp=torch.tensor([past],dtype=torch.long,device=DEVICE)
        step=model(input_ids=nxt,past_key_values=cache,attention_mask=att,cache_position=cp,position_ids=cp.unsqueeze(0),use_cache=True,return_dict=True)
        cache=step.past_key_values;nxt=step.logits[:,-1,:].argmax(-1,keepdim=True);del step
    return "" if not gen else tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()

CAP={}
CAPTURE_ENABLED=False
def make_hook(li):
    def hook(module,args):
        if CAPTURE_ENABLED and len(args) and args[0] is not None:CAP[li]=args[0].detach().clone()
    return hook
HOOKS=[model.model.layers[i].register_forward_pre_hook(make_hook(i)) for i in XRAY_LAYERS]
def clear_cap():CAP.clear()
def snapshot_cap():
    missing=[li for li in XRAY_LAYERS if li not in CAP]
    if missing:raise RuntimeError(f"Hook capture eksik katmanlar: {missing}")
    return {li:CAP[li].detach().clone() for li in XRAY_LAYERS}

@torch.no_grad()
def q_from_hidden(li,h,absolute_pos):
    if h.ndim!=3 or h.shape[0]!=1 or h.shape[-1]!=H:raise RuntimeError(f"L{li} hidden shape geçersiz: {tuple(h.shape)}")
    layer=model.model.layers[li];hn=layer.input_layernorm(h)
    q=layer.self_attn.q_proj(hn[:,-1:,:]).view(1,1,QH,HD).transpose(1,2).float()
    pos=torch.tensor([absolute_pos],dtype=torch.long,device=DEVICE)
    cos,sin=rope_cos_sin(pos)
    if cos.shape[-2]!=1:cos=cos[...,-1:,:];sin=sin[...,-1:,:]
    return q*cos+rotate_half(q)*sin

def repeat_kv(x):
    if x.ndim!=4 or x.shape[1]!=KVH or x.shape[-1]!=HD:raise RuntimeError(f"KV shape geçersiz: {tuple(x.shape)}")
    return x.repeat_interleave(GROUP,dim=1)

@torch.no_grad()
def metrics_from_physical_cache(li,q,k,v,ranges,target_slot):
    if q.shape!=(1,QH,1,HD):raise RuntimeError(f"L{li} Q shape geçersiz: {tuple(q.shape)}")
    if k.ndim!=4 or v.ndim!=4 or k.shape!=v.shape:raise RuntimeError(f"L{li} cache K/V shape geçersiz")
    total=int(k.shape[-2])
    for a,b in ranges:
        if not (0<=a<b<=total):raise RuntimeError(f"L{li} range geçersiz: {(a,b)} total={total}")
    Kq=repeat_kv(k.float());Vq=repeat_kv(v.float())
    scores=torch.matmul(q.float(),Kq.transpose(-2,-1))/math.sqrt(HD)
    p=torch.softmax(scores,dim=-1)
    masses=[];ovnorm=[];oproj=model.model.layers[li].self_attn.o_proj
    for a,b in ranges:
        pb=p[...,a:b]
        vb=Vq[:,:,a:b,:]
        if pb.shape[-1]!=vb.shape[-2]:raise RuntimeError(f"L{li} block matmul shape uyuşmazlığı: p={tuple(pb.shape)} V={tuple(vb.shape)}")
        masses.append(float(pb.sum(dim=-1).mean().item()))
        ctx=torch.matmul(pb,vb)
        ctx=ctx.transpose(1,2).contiguous().view(1,1,H).to(DTYPE)
        ovnorm.append(float(oproj(ctx).float().norm().item()))
    tm=masses[target_slot];to=ovnorm[target_slot]
    distract_m=[x for i,x in enumerate(masses) if i!=target_slot]
    distract_o=[x for i,x in enumerate(ovnorm) if i!=target_slot]
    dm=max(distract_m);do=max(distract_o)
    return tm,dm,tm/(dm+1e-12),to,do,to/(do+1e-12)

@torch.no_grad()
def run_block_xray(cars,target_slot,q):
    global CAPTURE_ENABLED
    clear_cap()
    blocks=[tok(source_prefix(c["fact"]),return_tensors="pt",add_special_tokens=False).input_ids[0].to(DEVICE) for c in cars]
    qids=tok(query_suffix(q),return_tensors="pt",add_special_tokens=False).input_ids[0].to(DEVICE)
    lens=[int(x.numel()) for x in blocks];past=sum(lens);qlen=int(qids.numel());total=past+qlen
    ids=torch.cat(blocks+[qids],dim=0).unsqueeze(0)
    ranges=[];off=0
    for L in lens:ranges.append((off,off+L));off+=L
    neg=torch.finfo(DTYPE).min
    mask=torch.full((1,1,total,total),neg,dtype=DTYPE,device=DEVICE);off=0
    for L in lens:
        tri=torch.tril(torch.ones((L,L),dtype=torch.bool,device=DEVICE))
        block=torch.zeros((L,L),dtype=DTYPE,device=DEVICE).masked_fill(~tri,neg)
        mask[0,0,off:off+L,off:off+L]=block;off+=L
    for r in range(qlen):
        row=past+r;mask[0,0,row,:past]=0;mask[0,0,row,past:past+r+1]=0
    pos=torch.arange(total,dtype=torch.long,device=DEVICE).unsqueeze(0)
    CAPTURE_ENABLED=True
    out=model(input_ids=ids,attention_mask=mask,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
    CAPTURE_ENABLED=False
    cap=snapshot_cap()
    physical=clone_layers(out.past_key_values)
    cache_audit(out.past_key_values,total)
    if any(int(k.shape[-2])!=total for k,v in physical):raise RuntimeError("T-BLOCK physical cache uzunluğu geçersiz.")
    logits=out.logits[:,-1,:].float();nxt=logits.argmax(-1,keepdim=True)
    M=[]
    for li in XRAY_LAYERS:
        h=cap[li];qv=q_from_hidden(li,h,total-1);k,v=physical[li]
        tm,dm,ratio,to,do,oratio=metrics_from_physical_cache(li,qv,k,v,ranges,target_slot)
        resid=float(h[:,-1,:].float().norm().item())
        M.append({"l":li,"tm":tm,"dm":dm,"r":ratio,"to":to,"do":do,"ro":oratio,"resid":resid})
    clear_cap()
    CAPTURE_ENABLED=False
    ans=continue_from_cache(out.past_key_values,nxt)
    del ids,mask,pos,out,blocks,qids,physical,cap
    return ans,logits.detach().cpu(),M

@torch.no_grad()
def run_praw_xray(cars,target_slot,q):
    global CAPTURE_ENABLED
    raw,ranges,past=compose_population(cars);sha=tensor_sha(raw);clear_cap()
    pkv=new_cache_from_layers(tuple((k.clone(),v.clone()) for k,v in raw))
    x=tok(query_suffix(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    ids=x.input_ids;qlen=int(ids.shape[1]);total=past+qlen
    att=torch.ones((1,total),dtype=torch.long,device=DEVICE)
    cp=torch.arange(past,total,dtype=torch.long,device=DEVICE)
    CAPTURE_ENABLED=True
    out=model(input_ids=ids,past_key_values=pkv,attention_mask=att,cache_position=cp,position_ids=cp.unsqueeze(0),use_cache=True,return_dict=True)
    CAPTURE_ENABLED=False
    cap=snapshot_cap()
    physical=clone_layers(out.past_key_values)
    cache_audit(out.past_key_values,total)
    if any(int(k.shape[-2])!=total for k,v in physical):raise RuntimeError("P-RAW physical cache uzunluğu geçersiz.")
    logits=out.logits[:,-1,:].float();nxt=logits.argmax(-1,keepdim=True)
    M=[]
    for li in XRAY_LAYERS:
        h=cap[li];qv=q_from_hidden(li,h,total-1);k,v=physical[li]
        tm,dm,ratio,to,do,oratio=metrics_from_physical_cache(li,qv,k,v,ranges,target_slot)
        resid=float(h[:,-1,:].float().norm().item())
        M.append({"l":li,"tm":tm,"dm":dm,"r":ratio,"to":to,"do":do,"ro":oratio,"resid":resid})
    if tensor_sha(raw)!=sha:raise RuntimeError("P-RAW population X-ray sırasında değişti.")
    clear_cap()
    CAPTURE_ENABLED=False
    ans=continue_from_cache(out.past_key_values,nxt)
    del raw,pkv,x,out,physical,cap
    return ans,logits.detach().cpu(),M

print("\n[2/10] Protokol...")
print("      Aynı target + aynı 4 distractor iki kez çalışır: TARGET FIRST ve TARGET LAST.")
print("      T-BLOCK ve P-RAW aynı paired düzeni görür.")
print("      SDPA değiştirilmez; X-ray gerçek forward'ın fiziksel cache K/V'si + query Q projectionı üzerinden hesaplanır.")
print("      Generation forward'ları X-ray hook snapshot'ını değiştiremez.")
print("      Ölçüm karar vermez, cache seçmez, maskeyi query'ye göre değiştirmez.")
print("      Ölçümler: target QK mass / strongest distractor, target V→O / strongest distractor, query residual norm.")
print("      Ana soru: doğru target erken/mid katmanda kazanıp sonra mı kayboluyor, yoksa hiç kazanamıyor mu?")

print("\n[3/10] TEST550 uyumlu 96 bağımsız native PKV forge...")
CAR=[];idx=0
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/24 {w['subject']}")
    qmap={x[0]:x for x in w["queries"]}
    for typ,fact,gold in w["facts"]:
        layers,n=forge(fact)
        CAR.append({"idx":idx,"world":wi,"type":typ,"fact":fact,"gold":gold,"query":qmap[typ][1],"layers":layers,"n":n,"sha":tensor_sha(layers)})
        idx+=1
print("      Cartridges:",len(CAR),"| Length min/mean/max:",min(c["n"] for c in CAR),f"{np.mean([c['n'] for c in CAR]):.2f}",max(c["n"] for c in CAR))

print("\n[4/10] Paired k=5 panel...")
rng=random.Random(SEED);pool=list(range(96));rng.shuffle(pool);PAN=[];cursor=0
for t in range(NTRIAL):
    target=pool[cursor%96];cursor+=1;others=[];tc=CAR[target]
    while len(others)<4:
        ci=pool[cursor%96];cursor+=1
        if ci==target or CAR[ci]["world"]==tc["world"] or any(CAR[x]["world"]==CAR[ci]["world"] for x in others):continue
        others.append(ci)
    first=[target]+others;last=others+[target]
    PAN.append({"trial":t,"target":target,"others":others,"first":first,"last":last})
PANEL_SHA=hashlib.sha256(json.dumps(PAN,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Trials:",len(PAN),"| PANEL SHA:",PANEL_SHA)
for z in PAN[:4]:print(f"      T{z['trial']:02d} target=C{z['target']:02d} FIRST={z['first']} LAST={z['last']}")

print("\n[5/10] Paired FIRST↔LAST X-ray...")
RES=[]
for z in PAN:
    target=CAR[z["target"]]
    print(f"\n      TRIAL {z['trial']+1:02d}/{NTRIAL} C{z['target']:02d} {target['type']:8s} gold={target['gold']}")
    for order in ["FIRST","LAST"]:
        inds=z[order.lower()];cars=[CAR[i] for i in inds];slot=inds.index(z["target"])
        ab,lb,mb=run_block_xray(cars,slot,target["query"])
        ap,lp,mp=run_praw_xray(cars,slot,target["query"])
        ob=hit(ab,target["gold"]);op=hit(ap,target["gold"])
        RES.append({"trial":z["trial"],"order":order,"target":z["target"],"block_ok":ob,"praw_ok":op,"block":ab,"praw":ap,"mb":mb,"mp":mp})
        print(f"      {order:5s} | T-BLOCK={'PASS' if ob else 'FAIL':4s} {ab!r} | P-RAW={'PASS' if op else 'FAIL':4s} {ap!r}")
        for li in [0,4,7,10,12,14,16,19,23,27,31]:
            b=mb[li];p=mp[li]
            print(f"             L{li:02d} B QK={b['r']:.3f} VO={b['ro']:.3f} | P QK={p['r']:.3f} VO={p['ro']:.3f}")

print("\n[6/10] Katman agregasyonu...")
AGG={}
for arm,key in [("T-BLOCK","mb"),("P-RAW","mp")]:
    AGG[arm]={}
    print(f"\n      {arm}")
    print("      L | FIRST_QK LAST_QK | FIRST_VO LAST_VO | ΔQK(L-F) ΔVO(L-F)")
    for li in XRAY_LAYERS:
        F=[x[key][li] for x in RES if x["order"]=="FIRST"];L=[x[key][li] for x in RES if x["order"]=="LAST"]
        fq=float(np.mean([x["r"] for x in F]));lq=float(np.mean([x["r"] for x in L]))
        fv=float(np.mean([x["ro"] for x in F]));lv=float(np.mean([x["ro"] for x in L]))
        fr=float(np.mean([x["resid"] for x in F]));lr=float(np.mean([x["resid"] for x in L]))
        AGG[arm][str(li)]={"first_qk":fq,"last_qk":lq,"first_vo":fv,"last_vo":lv,"dq":lq-fq,"dvo":lv-fv,"first_resid":fr,"last_resid":lr}
        print(f"      {li:02d} | {fq:8.3f} {lq:7.3f} | {fv:8.3f} {lv:7.3f} | {lq-fq:+8.3f} {lv-fv:+8.3f}")

print("\n[7/10] Native-binding trajectory diagnosis...")
def trajectory_diag(arm):
    a=AGG[arm]
    early=max(a[str(l)]["first_qk"] for l in range(0,7))
    mid=max(a[str(l)]["first_qk"] for l in FOCUS)
    late=float(np.mean([a[str(l)]["first_qk"] for l in range(24,32)]))
    mid_vo=max(a[str(l)]["first_vo"] for l in FOCUS)
    late_vo=float(np.mean([a[str(l)]["first_vo"] for l in range(24,32)]))
    if mid>1.0 and late<mid*.80:return "FIRST_BINDS_MID_THEN_DECAYS",early,mid,late,mid_vo,late_vo
    if mid<=1.0:return "FIRST_NEVER_BEATS_STRONGEST_DISTRACTOR",early,mid,late,mid_vo,late_vo
    return "FIRST_BINDING_PERSISTS_OR_MIXED",early,mid,late,mid_vo,late_vo
DIAG={}
for arm in ["T-BLOCK","P-RAW"]:
    d,e,m,l,mv,lv=trajectory_diag(arm);DIAG[arm]=d
    print(f"      {arm:7s}: {d}")
    print(f"               FIRST QK early_max={e:.4f} mid_max={m:.4f} late_mean={l:.4f} | VO mid_max={mv:.4f} late_mean={lv:.4f}")
print("      Diagnosis yalnız röntgen özeti; yeni routing/selection kuralı değildir.")

print("\n[8/10] Behavioral paired effect...")
BEH={}
for arm,okey in [("T-BLOCK","block_ok"),("P-RAW","praw_ok")]:
    F=[x for x in RES if x["order"]=="FIRST"];L=[x for x in RES if x["order"]=="LAST"]
    fc=sum(x[okey] for x in F);lc=sum(x[okey] for x in L);pairs=[]
    for t in range(NTRIAL):
        f=next(x for x in F if x["trial"]==t);l=next(x for x in L if x["trial"]==t)
        pairs.append((int(f[okey]),int(l[okey])))
    gain=sum(a==0 and b==1 for a,b in pairs);loss=sum(a==1 and b==0 for a,b in pairs)
    BEH[arm]={"first":fc,"last":lc,"n":NTRIAL,"first_to_last_gain":gain,"first_to_last_loss":loss}
    print(f"      {arm:7s}: FIRST={fc}/{NTRIAL} LAST={lc}/{NTRIAL} | F→L gain={gain} loss={loss}")

print("\n[9/10] Anti-drift / physical audit...")
seal_ok=all(tensor_sha(c["layers"])==c["sha"] for c in CAR)
print("      Original cartridge SHA unchanged      :","PASS" if seal_ok else "FAIL")
print("      SDPA inference implementation          : UNCHANGED")
print("      Paired target/distractors              : SAME; ONLY ORDER CHANGES")
print("      X-ray K/V source                       : ACTUAL FORWARD PHYSICAL CACHE")
print("      Generation overwrites X-ray snapshot   : YOK")
print("      X-ray changes model computation        : YOK")
print("      BOS/template surgery                   : YOK")
print("      Steering / intervention                : YOK")
print("      Query-dependent cartridge selection    : YOK")
print("      External score / cosine / ANN          : YOK")
print("      Router / classifier                    : YOK")
print("      Training / LoRA / optimizer            : YOK")

print("\n[10/10] Sentinel / mühür...")
for h in HOOKS:h.remove()
S1=sentinel();WOK=S0==S1;trainable=sum(int(p.requires_grad) for p in model.parameters())
FINAL={"test":TEST,"parent548":PARENT548,"parent549":PARENT549,"parent550":PARENT550,"lock_sha":LOCK_SHA,"panel_sha":PANEL_SHA,"behavior":BEH,"diagnosis":DIAG,"cartridge_seal":seal_ok,"weight_sentinel":WOK,"trainable":trainable,"sdpa_unchanged":True,"physical_cache_xray":True,"retrieval":False,"router":False,"scorer":False,"steering":False,"training":False}
RESULT_SHA=hashlib.sha256(json.dumps(FINAL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Cartridge seal            :","PASS" if seal_ok else "FAIL")
print("      Weight sentinel            :","PASS" if WOK else "FAIL")
print("      Trainable tensors          :",trainable)

print("\n"+"="*178)
print("TEST551 SONUÇ — LAST-CARTRIDGE CAPTURE X-RAY")
print("="*178)
print("MODEL                         :",MODEL_ID)
for arm in ["T-BLOCK","P-RAW"]:
    b=BEH[arm]
    print(f"{arm} FIRST                   : {b['first']}/{b['n']}")
    print(f"{arm} LAST                    : {b['last']}/{b['n']}")
    print(f"{arm} FIRST→LAST GAINS        : {b['first_to_last_gain']}")
    print(f"{arm} FIRST→LAST LOSSES       : {b['first_to_last_loss']}")
    print(f"{arm} TRAJECTORY              : {DIAG[arm]}")
print("FOCUS BAND                    : L7–L19")
print("XRAY K/V                      : ACTUAL PHYSICAL CACHE")
print("SDPA                          : UNCHANGED")
print("CARTRIDGE SEAL                :","PASS" if seal_ok else "FAIL")
print("WEIGHT SENTINEL               :","PASS" if WOK else "FAIL")
print("TEST548 RESULT SHA            :",PARENT548)
print("TEST549 RESULT SHA            :",PARENT549)
print("TEST550 RESULT SHA            :",PARENT550)
print("TEST551 ÖN KİLİT SHA          :",LOCK_SHA)
print("TEST551 RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                         : TEST551_LAST_CAPTURE_PHYSICS_MAPPED" if WOK and seal_ok else "KARAR                         : TEST551_PHYSICAL_VALIDITY_FAILURE")
print("="*178)
