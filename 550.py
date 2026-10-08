# TEST550 — AKBASCORE MAM · INDEPENDENT CARTRIDGE COMPOSITION DIAGNOSIS
# T-JOINT vs T-BLOCK vs P-RAW · k=2 / k=5
# PURPOSE: TEST549 FAILURE = INDEPENDENT-WRITE LIMIT OR NUMERICAL PKV COMPOSITION ARTIFACT?
# NO FIX · NO BOS REMOVAL · NO TEMPLATE REMOVAL · NO RETRIEVAL · NO ROUTER · NO SCORER
# NO QUERY-DEPENDENT CARTRIDGE OP · NO TRAINING · NO LoRA · NO OPTIMIZER
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache=None

TEST="550";SEED=550550;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT548="68e1594d7095cf7828286572a2ae6250bfcd5a7e3612e6db9c15a429bc4a8a87"
PARENT549="f92d5ea0e4e29224ea0e72d163d4c855c18969be14622d151bf54f5fae674186"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16;K_LIST=[2,5]
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*178)
print("TEST550 — AKBASCORE MAM · INDEPENDENT CARTRIDGE COMPOSITION DIAGNOSIS")
print("T-JOINT vs T-BLOCK vs P-RAW · k=2 / k=5")
print("TEST549 FAILURE: INDEPENDENT-WRITE LIMIT OR NUMERICAL PKV COMPOSITION ARTIFACT?")
print("NO FIX · NO RETRIEVAL · NO ROUTER · NO SCORER · NO BOS/TEMPLATE SURGERY · NO TRAINING")
print("="*178);T0=time.perf_counter()

print("\n[1/9] Donmuş Mistral...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | Torch {torch.__version__} | Transformers {transformers.__version__}")
print(f"      {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | {DTYPE} | trainable=0")

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
("Tervan","Orel","Mardin","Velos","Karven"),("Karev","Solen","Dareth","Mirven","Talrek")
]
W=[]
for i,(s,current,near,former,role_target) in enumerate(BASE):
    facts=[
        ("CURRENT",f"The current capital of {s} is {current}.",current),
        ("NEAR",f"The largest city of {s} is {near}.",near),
        ("FORMER",f"The former capital of {s} was {former}.",former),
        ("ROLE",f"The current capital of {role_target} is {s}.",s)]
    queries=[
        ("CURRENT",f"What is the current capital of {s}?",current),
        ("FORMER",f"What was the former capital of {s}?",former),
        ("NEAR",f"What is the largest city of {s}?",near),
        ("ROLE",f"What is the current capital of {role_target}?",s)]
    W.append({"id":i,"subject":s,"facts":facts,"queries":queries})

SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def norm(s):
    s=s.casefold().strip();s=re.sub(r"[^\w\s-]"," ",s,flags=re.UNICODE);return " ".join(s.split())
def hit(out,gold):return re.search(r"(?<!\w)"+re.escape(norm(gold))+r"(?!\w)",norm(out)) is not None
def source_prefix(fact):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{fact}\n\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
def joint_text(facts,q):
    info="\n".join(facts)
    return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{info}\n\nQUESTION:\n{q}\n\nANSWER: [/INST]"

LOCK={"test":TEST,"parent548":PARENT548,"parent549":PARENT549,"model":MODEL_ID,"seed":SEED,"k_list":K_LIST,"arms":["T-JOINT","T-BLOCK","P-RAW"],"diagnostic_only":True,"independent_write":True,"bos_removal":False,"template_removal":False,"retrieval":False,"router":False,"scorer":False,"query_dependent_cartridge_op":False,"training":False,"lora":False,"optimizer":False}
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
        if k.ndim!=4 or v.ndim!=4 or tuple(k.shape)!=tuple(v.shape):raise RuntimeError(f"L{li} K/V shape hatası")
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
    cache_audit(c,int(layers[0][0].shape[-2]))
    return c

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
def forge(fact,position_offset=0):
    x=tok(source_prefix(fact),return_tensors="pt",add_special_tokens=False).to(DEVICE);n=int(x.input_ids.shape[1])
    pos=torch.arange(position_offset,position_offset+n,dtype=torch.long,device=DEVICE).unsqueeze(0)
    out=model(**x,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
    pkv=out.past_key_values;cache_audit(pkv,n);layers=clone_layers(pkv)
    del out,x,pkv
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
    old=torch.arange(old_start,old_start+L,device=DEVICE,dtype=torch.long)
    new=torch.arange(new_start,new_start+L,device=DEVICE,dtype=torch.long)
    cos_old,sin_old=rope_cos_sin(old,torch.float32);cos_new,sin_new=rope_cos_sin(new,torch.float32);ko=k.float()
    raw=ko*cos_old-rotate_half(ko)*sin_old;knew=raw*cos_new+rotate_half(raw)*sin_new
    return knew.to(dtype=k.dtype)

@torch.no_grad()
def rebase_layers(layers,old_start,new_start):
    return tuple((rope_rebase_k(k,old_start,new_start),v.detach().clone()) for k,v in layers)

def compose_population(cars):
    offset=0;per=[[] for _ in range(NL)];ranges=[]
    for c in cars:
        rb=rebase_layers(c["layers"],0,offset);ranges.append((offset,offset+c["n"]))
        for l,(k,v) in enumerate(rb):per[l].append((k,v))
        offset+=c["n"]
    layers=tuple((torch.cat([x[0] for x in per[l]],dim=-2),torch.cat([x[1] for x in per[l]],dim=-2)) for l in range(NL))
    return layers,ranges,offset

@torch.no_grad()
def continue_from_cache(cache,nxt):
    gen=[]
    eos=tok.eos_token_id
    eos=set() if eos is None else ({int(eos)} if not isinstance(eos,(list,tuple,set)) else set(map(int,eos)))
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        past=cache_len(cache)
        att=torch.ones((1,past+1),dtype=torch.long,device=DEVICE)
        cp=torch.tensor([past],device=DEVICE,dtype=torch.long)
        step=model(input_ids=nxt,past_key_values=cache,attention_mask=att,cache_position=cp,position_ids=cp.unsqueeze(0),use_cache=True,return_dict=True)
        cache=step.past_key_values;nxt=step.logits[:,-1,:].argmax(-1,keepdim=True)
        del step
    return "" if not gen else tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()

@torch.no_grad()
def answer_from_layers(layers,q,return_logits=False):
    pkv=new_cache_from_layers(tuple((k.clone(),v.clone()) for k,v in layers))
    x=tok(query_suffix(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    ids=x.input_ids;past=cache_len(pkv);qlen=int(ids.shape[1])
    att=torch.ones((1,past+qlen),dtype=torch.long,device=DEVICE)
    cp=torch.arange(past,past+qlen,dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids,past_key_values=pkv,attention_mask=att,cache_position=cp,position_ids=cp.unsqueeze(0),use_cache=True,return_dict=True)
    logits=out.logits[:,-1,:].float();nxt=logits.argmax(-1,keepdim=True)
    cache=out.past_key_values
    ans=continue_from_cache(cache,nxt)
    ret=(ans,logits.detach().cpu()) if return_logits else ans
    del pkv,x,out,cache
    return ret

@torch.no_grad()
def answer_joint(facts,q):
    text=joint_text(facts,q)
    x=tok(text,return_tensors="pt",add_special_tokens=False).to(DEVICE);n=int(x.input_ids.shape[1])
    y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    ans=tok.decode(y[0,n:],skip_special_tokens=True).strip()
    del x,y
    return ans

def block_tokens(fact):
    return tok(source_prefix(fact),return_tensors="pt",add_special_tokens=False).input_ids[0]

@torch.no_grad()
def answer_block(cars,q,return_logits=False):
    blocks=[block_tokens(c["fact"]).to(DEVICE) for c in cars]
    qids=tok(query_suffix(q),return_tensors="pt",add_special_tokens=False).input_ids[0].to(DEVICE)
    lens=[int(x.numel()) for x in blocks];past=sum(lens);qlen=int(qids.numel());total=past+qlen
    ids=torch.cat(blocks+[qids],dim=0).unsqueeze(0)

    # Static block-diagonal causal mask.
    # Her source bloğu yalnız kendi içindeki geçmişi görür.
    # Query bütün tamamlanmış source bloklarını ve kendi causal geçmişini görür.
    neg=torch.finfo(DTYPE).min
    mask=torch.full((1,1,total,total),neg,dtype=DTYPE,device=DEVICE)
    off=0
    for L in lens:
        tri=torch.tril(torch.ones((L,L),dtype=torch.bool,device=DEVICE))
        block=torch.zeros((L,L),dtype=DTYPE,device=DEVICE).masked_fill(~tri,neg)
        mask[0,0,off:off+L,off:off+L]=block
        off+=L
    for r in range(qlen):
        row=past+r
        mask[0,0,row,:past]=0
        mask[0,0,row,past:past+r+1]=0

    pos=torch.arange(total,dtype=torch.long,device=DEVICE).unsqueeze(0)
    out=model(input_ids=ids,attention_mask=mask,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
    logits=out.logits[:,-1,:].float();nxt=logits.argmax(-1,keepdim=True)
    cache=out.past_key_values
    ans=continue_from_cache(cache,nxt)
    ret=(ans,logits.detach().cpu()) if return_logits else ans
    del ids,mask,pos,out,cache,blocks,qids
    return ret

def logit_diag(a,b):
    d=(a-b).abs()
    return float(d.max().item()),float(d.mean().item()),int(a.argmax(-1).item()==b.argmax(-1).item())

print("\n[2/9] Protokol...")
print("      T-JOINT : normal ortak causal text context — doğal tavan.")
print("      T-BLOCK : her source bloğu bağımsız causal ada; query bütün tamamlanmış blokları görür.")
print("      P-RAW   : TEST549 bağımsız PKV forge + RoPE rebase + concat AYNEN.")
print("      TEST550 hiçbir şeyi düzeltmez: BOS/template silme veya yeniden yazma YOK.")
print("      Amaç yalnız TEST549 çöküşünün hangi fiziksel sınırdan geldiğini ayırmaktır.")

print("\n[3/9] TEST549 uyumlu 96 bağımsız PKV forge...")
CAR=[];idx=0
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/24 {w['subject']}")
    qmap={x[0]:x for x in w["queries"]}
    for typ,fact,gold in w["facts"]:
        layers,n=forge(fact,0);q=qmap[typ][1]
        CAR.append({"idx":idx,"world":wi,"type":typ,"fact":fact,"gold":gold,"query":q,"layers":layers,"n":n,"sha":tensor_sha(layers)})
        idx+=1
print("      Cartridges:",len(CAR))
print("      Length min/mean/max:",min(c["n"] for c in CAR),f"{np.mean([c['n'] for c in CAR]):.2f}",max(c["n"] for c in CAR))

print("\n[4/9] Önceden sabitlenmiş k=2 / k=5 paneller...")
rng=random.Random(SEED);pool=list(range(len(CAR)));rng.shuffle(pool)
PANELS={2:[],5:[]};used=0
for k,trials in [(2,24),(5,24)]:
    for t in range(trials):
        target=pool[used%len(pool)];used+=1;tc=CAR[target];others=[];scan=0
        while len(others)<k-1:
            ci=pool[(used+scan)%len(pool)];scan+=1
            if ci==target or CAR[ci]["world"]==tc["world"] or any(CAR[x]["world"]==CAR[ci]["world"] for x in others):continue
            others.append(ci)
        used+=scan
        inds=[target]+others if t%2==0 else others+[target]
        PANELS[k].append({"trial":t,"target":target,"indices":inds,"target_pos":inds.index(target)})
    sha=hashlib.sha256(json.dumps([x["indices"] for x in PANELS[k]],separators=(",",":")).encode()).hexdigest()
    print(f"      k={k}: trials={len(PANELS[k])} PANEL_SHA={sha}")
    print("           first 4:",[x["indices"] for x in PANELS[k][:4]])

print("\n[5/9] T-JOINT / T-BLOCK / P-RAW...")
RES=[]
for k in K_LIST:
    print(f"\n      {'='*120}\n      k={k}\n      {'='*120}")
    for z in PANELS[k]:
        cars=[CAR[i] for i in z["indices"]];target=CAR[z["target"]];facts=[c["fact"] for c in cars]
        aj=answer_joint(facts,target["query"])
        ab,lb=answer_block(cars,target["query"],True)
        raw,ranges,total=compose_population(cars);sealed=tensor_sha(raw)
        ap,lp=answer_from_layers(raw,target["query"],True)
        if tensor_sha(raw)!=sealed:raise RuntimeError("P-RAW seal query sırasında değişti.")
        oj=hit(aj,target["gold"]);ob=hit(ab,target["gold"]);op=hit(ap,target["gold"])
        lmax,lmean,top1same=logit_diag(lb,lp)
        row={"k":k,"trial":z["trial"],"target":z["target"],"target_pos":z["target_pos"],"indices":z["indices"],"gold":target["gold"],"joint":aj,"block":ab,"praw":ap,"joint_ok":oj,"block_ok":ob,"praw_ok":op,"logit_max":lmax,"logit_mean":lmean,"top1_same":top1same,"praw_sha":sealed}
        RES.append(row)
        where="FIRST" if z["target_pos"]==0 else ("LAST" if z["target_pos"]==k-1 else f"POS{z['target_pos']}")
        print(f"      {z['trial']+1:02d}/24 target=C{z['target']:02d} {where:5s} gold={target['gold']:10s} | J={'P' if oj else 'F'} B={'P' if ob else 'F'} P={'P' if op else 'F'} | B↔P top1={top1same} Δmax={lmax:.5f} Δmean={lmean:.5f}")
        print(f"             J={aj!r}")
        print(f"             B={ab!r}")
        print(f"             P={ap!r}")
        del raw

print("\n[6/9] Sonuç kırılımı...")
SUMMARY={}
for k in K_LIST:
    rr=[x for x in RES if x["k"]==k];n=len(rr)
    J=sum(x["joint_ok"] for x in rr);B=sum(x["block_ok"] for x in rr);P=sum(x["praw_ok"] for x in rr);TS=sum(x["top1_same"] for x in rr)
    first=[x for x in rr if x["target_pos"]==0];last=[x for x in rr if x["target_pos"]==k-1]
    BF=sum(x["block_ok"] for x in first);PF=sum(x["praw_ok"] for x in first);BL=sum(x["block_ok"] for x in last);PL=sum(x["praw_ok"] for x in last)
    lm=float(np.mean([x["logit_mean"] for x in rr]));lx=max(x["logit_max"] for x in rr)
    SUMMARY[str(k)]={"n":n,"joint":J,"block":B,"praw":P,"top1_same":TS,"block_first":BF,"praw_first":PF,"first_n":len(first),"block_last":BL,"praw_last":PL,"last_n":len(last),"logit_mean":lm,"logit_max":lx}
    print(f"      k={k} T-JOINT : {J:02d}/{n} = {J/n:.6f}")
    print(f"      k={k} T-BLOCK : {B:02d}/{n} = {B/n:.6f}")
    print(f"      k={k} P-RAW   : {P:02d}/{n} = {P/n:.6f}")
    print(f"      k={k} B↔P top1 same: {TS:02d}/{n} = {TS/n:.6f}")
    print(f"      k={k} FIRST target | T-BLOCK={BF}/{len(first)} P-RAW={PF}/{len(first)}")
    print(f"      k={k} LAST  target | T-BLOCK={BL}/{len(last)} P-RAW={PL}/{len(last)}")
    print(f"      k={k} B↔P logits | mean|Δ|={lm:.8f} max|Δ|={lx:.8f}")

print("\n[7/9] Teşhis matrisi...")
def diagnose(k):
    s=SUMMARY[str(k)];j=s["joint"]/s["n"];b=s["block"]/s["n"];p=s["praw"]/s["n"]
    if j>=.90 and b>=.90 and p<.75:return "COMPOSITION_ARTIFACT_CANDIDATE"
    if j>=.90 and b<.75:return "INDEPENDENT_WRITE_LIMIT_CANDIDATE"
    if j>=.90 and b>=.90 and p>=.90:return "SMALL_K_COMPOSITION_WORKS"
    return "MIXED_OR_INCONCLUSIVE"
for k in K_LIST:print(f"      k={k}: {diagnose(k)}")
print("      Bu etiketler mekanizma kanıtı değil; bir sonraki fiziksel dalı seçmek için önceden tanımlı teşhistir.")

print("\n[8/9] Anti-drift / seal audit...")
seal_ok=all(tensor_sha(c["layers"])==c["sha"] for c in CAR)
print("      Original cartridge SHA unchanged      :","PASS" if seal_ok else "FAIL")
print("      P-RAW = TEST549 composition law        : EVET")
print("      BOS removal                            : YOK")
print("      Template removal                       : YOK")
print("      Cartridge rewriting                    : YOK")
print("      Query-dependent cartridge selection    : YOK")
print("      External score / cosine / ANN          : YOK")
print("      Router / classifier                    : YOK")
print("      RAG / external search                  : YOK")
print("      Training / LoRA / optimizer            : YOK")
print("      T-JOINT/T-BLOCK                        : DIAGNOSTIC CONTROLS ONLY")

print("\n[9/9] Sentinel / mühür...")
S1=sentinel();WOK=S0==S1;trainable=sum(int(p.requires_grad) for p in model.parameters())
FINAL={"test":TEST,"parent548":PARENT548,"parent549":PARENT549,"lock_sha":LOCK_SHA,"summary":SUMMARY,"diagnosis":{str(k):diagnose(k) for k in K_LIST},"cartridge_seal":seal_ok,"weight_sentinel":WOK,"trainable":trainable,"bos_removal":False,"template_removal":False,"retrieval":False,"router":False,"scorer":False,"training":False}
RESULT_SHA=hashlib.sha256(json.dumps(FINAL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Cartridge seal            :","PASS" if seal_ok else "FAIL")
print("      Weight sentinel            :","PASS" if WOK else "FAIL")
print("      Trainable tensors          :",trainable)

print("\n"+"="*178)
print("TEST550 SONUÇ — INDEPENDENT CARTRIDGE COMPOSITION DIAGNOSIS")
print("="*178)
print("MODEL                         :",MODEL_ID)
for k in K_LIST:
    s=SUMMARY[str(k)]
    print(f"k={k} T-JOINT                  : {s['joint']}/{s['n']} = {s['joint']/s['n']:.6f}")
    print(f"k={k} T-BLOCK                  : {s['block']}/{s['n']} = {s['block']/s['n']:.6f}")
    print(f"k={k} P-RAW                    : {s['praw']}/{s['n']} = {s['praw']/s['n']:.6f}")
    print(f"k={k} B↔P TOP1 SAME            : {s['top1_same']}/{s['n']} = {s['top1_same']/s['n']:.6f}")
    print(f"k={k} B↔P MEAN LOGIT Δ         : {s['logit_mean']:.8f}")
    print(f"k={k} B↔P MAX LOGIT Δ          : {s['logit_max']:.8f}")
    print(f"k={k} DIAGNOSIS                : {diagnose(k)}")
print("CARTRIDGE SEAL                :","PASS" if seal_ok else "FAIL")
print("WEIGHT SENTINEL               :","PASS" if WOK else "FAIL")
print("TEST548 RESULT SHA            :",PARENT548)
print("TEST549 RESULT SHA            :",PARENT549)
print("TEST550 ÖN KİLİT SHA          :",LOCK_SHA)
print("TEST550 RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                         : TEST550_COMPOSITION_CAUSE_DIAGNOSED" if WOK and seal_ok else "KARAR                         : TEST550_PHYSICAL_VALIDITY_FAILURE")
print("="*178)
