# TEST549 — AKBASCORE MAM · VARAN 1 · MULTI-CARTRIDGE NATIVE BINDING
# GATE-A: MODEL-NATIVE RoPE REBASE PHYSICAL VALIDATION
# GATE-B: INDEPENDENT NUMERICAL PKV POPULATION → FROZEN MISTRAL NATIVE BINDING
# QUERY-BEFORE COMPOSITION YOK · QUERY-DEPENDENT CACHE OP YOK · RETRIEVAL YOK · ROUTER YOK · SCORER YOK
# SOURCE REPLAY YOK · RAG YOK · TRAINING YOK · LoRA YOK · OPTIMIZER YOK
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache=None

TEST="549";SEED=549549;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT548="68e1594d7095cf7828286572a2ae6250bfcd5a7e3612e6db9c15a429bc4a8a87"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16
N_LIST=[2,8,32,96];REBASE_DELTAS=[7,31,79]
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*178)
print("TEST549 — AKBASCORE MAM · VARAN 1 · MULTI-CARTRIDGE NATIVE BINDING")
print("GATE-A RoPE REBASE VALIDATION → GATE-B SEALED MULTI-PKV POPULATION → QUERY")
print("NO RETRIEVAL · NO ROUTER · NO SCORER · NO QUERY-DEPENDENT CACHE/MASK · NO SOURCE REPLAY · NO TRAINING")
print("="*178);T0=time.perf_counter()

print("\n[1/10] Donmuş Mistral...")
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

LOCK={"test":TEST,"parent548":PARENT548,"model":MODEL_ID,"seed":SEED,"worlds":24,"cartridges":96,"n_list":N_LIST,"rebase_deltas":REBASE_DELTAS,"independent_write":True,"query_before_compose":False,"sealed_population":True,"native_pkv":True,"retrieval":False,"router":False,"scorer":False,"query_dependent_mask":False,"query_dependent_cache_op":False,"rag":False,"training":False,"lora":False,"optimizer":False}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      LOCK SHA:",LOCK_SHA)

SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def norm(s):
    s=s.casefold().strip();s=re.sub(r"[^\w\s-]"," ",s,flags=re.UNICODE);return " ".join(s.split())
def hit(out,gold):return re.search(r"(?<!\w)"+re.escape(norm(gold))+r"(?!\w)",norm(out)) is not None
def source_prefix(fact):return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{fact}\n\n"
def query_suffix(q):return f"QUESTION:\n{q}\n\nANSWER: [/INST]"

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
    if hasattr(pkv,"key_cache") and hasattr(pkv,"value_cache"):
        return tuple(zip(pkv.key_cache,pkv.value_cache))
    if hasattr(pkv,"to_legacy_cache"):
        p=pkv.to_legacy_cache()
        if isinstance(p,(tuple,list)):return tuple(p)
    if isinstance(pkv,(tuple,list)):return tuple(pkv)
    raise RuntimeError(f"Desteklenmeyen cache tipi: {type(pkv)}")

def cache_len(pkv):
    if hasattr(pkv,"get_seq_length"):
        try:return int(pkv.get_seq_length())
        except TypeError:return int(pkv.get_seq_length(0))
    p=cache_layers(pkv);return int(p[0][0].shape[-2])

def cache_audit(pkv,expected=None):
    L=cache_layers(pkv)
    if len(L)!=NL:raise RuntimeError(f"PKV layer sayısı {len(L)} != {NL}")
    lens=[]
    for li,(k,v) in enumerate(L):
        if k.ndim!=4 or v.ndim!=4:raise RuntimeError(f"L{li} K/V rank hatası")
        if tuple(k.shape)!=tuple(v.shape):raise RuntimeError(f"L{li} K/V shape farklı")
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

def clone_layers(pkv):
    return tuple((k.detach().clone(),v.detach().clone()) for k,v in cache_layers(pkv))

def raw_bytes(x):
    x=x.detach().contiguous().cpu()
    if x.dtype==torch.bfloat16:x=x.view(torch.uint16)
    return x.numpy().tobytes()

def tensor_sha(layers):
    h=hashlib.sha256()
    for k,v in layers:
        h.update(raw_bytes(k));h.update(raw_bytes(v))
    return h.hexdigest()

@torch.no_grad()
def forge(fact,position_offset=0):
    text=source_prefix(fact)
    x=tok(text,return_tensors="pt",add_special_tokens=False).to(DEVICE)
    n=int(x.input_ids.shape[1])
    pos=torch.arange(position_offset,position_offset+n,dtype=torch.long,device=DEVICE).unsqueeze(0)
    out=model(**x,position_ids=pos,cache_position=pos.squeeze(0),use_cache=True,return_dict=True)
    pkv=out.past_key_values;cache_audit(pkv,n)
    layers=clone_layers(pkv)
    del out,x,pkv
    return layers,n

def rotate_half(x):
    x1=x[...,:x.shape[-1]//2];x2=x[...,x.shape[-1]//2:]
    return torch.cat((-x2,x1),dim=-1)

@torch.no_grad()
def rope_cos_sin(position_ids,dtype=torch.float32):
    rotary=model.model.rotary_emb
    pos=position_ids.reshape(1,-1).to(device=DEVICE,dtype=torch.long)
    dummy=torch.zeros((1,pos.shape[1],H),device=DEVICE,dtype=DTYPE)
    try:
        cos,sin=rotary(dummy,pos)
    except TypeError:
        cos,sin=rotary(dummy,position_ids=pos)
    if cos.ndim==2:
        cos=cos.unsqueeze(0)
        sin=sin.unsqueeze(0)
    if cos.ndim==3:
        cos=cos.unsqueeze(1)
        sin=sin.unsqueeze(1)
    return cos.to(dtype=dtype),sin.to(dtype=dtype)

@torch.no_grad()
def rope_rebase_k(k,old_start,new_start):
    L=int(k.shape[-2])
    old=torch.arange(old_start,old_start+L,device=DEVICE,dtype=torch.long)
    new=torch.arange(new_start,new_start+L,device=DEVICE,dtype=torch.long)
    cos_old,sin_old=rope_cos_sin(old,torch.float32)
    cos_new,sin_new=rope_cos_sin(new,torch.float32)
    ko=k.float()
    raw=ko*cos_old-rotate_half(ko)*sin_old
    knew=raw*cos_new+rotate_half(raw)*sin_new
    return knew.to(dtype=k.dtype)

@torch.no_grad()
def rebase_layers(layers,old_start,new_start):
    return tuple((rope_rebase_k(k,old_start,new_start),v.detach().clone()) for k,v in layers)

@torch.no_grad()
def answer_from_layers(layers,q):
    pkv=new_cache_from_layers(tuple((k.clone(),v.clone()) for k,v in layers))
    x=tok(query_suffix(q),return_tensors="pt",add_special_tokens=False).to(DEVICE)
    ids=x.input_ids;past=cache_len(pkv);qlen=int(ids.shape[1])
    att=torch.ones((1,past+qlen),dtype=torch.long,device=DEVICE)
    cp=torch.arange(past,past+qlen,dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids,past_key_values=pkv,attention_mask=att,cache_position=cp,position_ids=cp.unsqueeze(0),use_cache=True,return_dict=True)
    cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True);gen=[]
    eos=tok.eos_token_id
    eos=set() if eos is None else ({int(eos)} if not isinstance(eos,(list,tuple,set)) else set(map(int,eos)))
    for _ in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        past=cache_len(cache)
        att=torch.ones((1,past+1),dtype=torch.long,device=DEVICE)
        cp=torch.tensor([past],device=DEVICE,dtype=torch.long)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=att,cache_position=cp,position_ids=cp.unsqueeze(0),use_cache=True,return_dict=True)
        cache=out.past_key_values;nxt=out.logits[:,-1,:].argmax(-1,keepdim=True)
    ans="" if not gen else tok.decode(torch.cat(gen,dim=1)[0],skip_special_tokens=True).strip()
    del pkv,x,out,cache
    return ans

print("\n[2/10] Protokol...")
print("      GATE-A önce RoPE rebasing'i gerçek Mistral PKV üzerinde doğrular.")
print("      GATE-A geçmeden multi-cartridge native-binding sonucu yorumlanmaz.")
print("      GATE-B'de bütün cartridge population QUERY'DEN ÖNCE kurulur ve SHA ile mühürlenir.")
print("      Query geldikten sonra cartridge seçimi/sıralaması/döndürmesi/ağırlığı/maskesi YOK.")
print("      Model bütün numerical population'a kendi doğal attention/OV/residual/MLP zinciriyle erişir.")

print("\n[3/10] TEST548 uyumlu 96 bağımsız native PKV forge...")
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

print("\n[4/10] GATE-A — model-native RoPE rebase fizik doğrulaması...")
GA=[];probe_ids=[];seen=set()
for i,c in enumerate(CAR):
    key=(c["n"],c["type"])
    if key not in seen:seen.add(key);probe_ids.append(i)
    if len(probe_ids)>=8:break
for ci in probe_ids:
    c=CAR[ci]
    for delta in REBASE_DELTAS:
        direct,_=forge(c["fact"],delta)
        rebased=rebase_layers(c["layers"],0,delta)
        kmax=vmax=0.0
        for (kr,vr),(kd,vd) in zip(rebased,direct):
            kmax=max(kmax,float((kr.float()-kd.float()).abs().max().item()))
            vmax=max(vmax,float((vr.float()-vd.float()).abs().max().item()))
        a=answer_from_layers(rebased,c["query"]);b=answer_from_layers(direct,c["query"]);same=norm(a)==norm(b)
        GA.append({"ci":ci,"delta":delta,"kmax":kmax,"vmax":vmax,"same":same,"a":a,"b":b})
        print(f"      C{ci:02d} {c['type']:8s} L={c['n']:02d} δ={delta:03d} | Kmax={kmax:.8f} Vmax={vmax:.8f} | answer_same={same}")
GA_SAME=sum(x["same"] for x in GA);GA_K=max(x["kmax"] for x in GA);GA_V=max(x["vmax"] for x in GA)
print(f"      GATE-A answer equivalence: {GA_SAME}/{len(GA)}")
print(f"      GATE-A global Kmax={GA_K:.8f} Vmax={GA_V:.8f}")
GATE_A=(GA_SAME==len(GA))
if not GATE_A:
    print("\n"+"!"*178)
    print("GATE-A FAIL — RoPE rebase fizik eşdeğerliği davranışsal olarak doğrulanmadı.")
    print("MULTI-CARTRIDGE BINDING HAKKINDA SONUÇ ÇIKARILMAYACAK. TEST BURADA DURDU.")
    print("!"*178)
    S1=sentinel()
    FINAL={"test":TEST,"parent548":PARENT548,"lock_sha":LOCK_SHA,"gate_a":False,"gate_a_same":GA_SAME,"gate_a_total":len(GA),"gate_a_kmax":GA_K,"gate_a_vmax":GA_V,"weight_sentinel":S0==S1}
    RESULT_SHA=hashlib.sha256(json.dumps(FINAL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    print("WEIGHT SENTINEL :","PASS" if S0==S1 else "FAIL")
    print("RESULT SHA      :",RESULT_SHA)
    print("TOPLAM SÜRE     :",f"{time.perf_counter()-T0:.2f}s")
    raise SystemExit

print("\n[5/10] GATE-B population plan — QUERY görülmeden...")
ORDER=list(range(len(CAR)));rng=random.Random(SEED);rng.shuffle(ORDER)
print("      Fixed order SHA:",hashlib.sha256(json.dumps(ORDER,separators=(",",":")).encode()).hexdigest())
print("      Order first 16 :",ORDER[:16])

def compose_population(indices):
    offset=0;per_layer=[[] for _ in range(NL)];ranges={}
    for ci in indices:
        c=CAR[ci];rb=rebase_layers(c["layers"],0,offset)
        ranges[ci]=(offset,offset+c["n"])
        for l,(k,v) in enumerate(rb):per_layer[l].append((k,v))
        offset+=c["n"]
    layers=[]
    for l in range(NL):
        K=torch.cat([x[0] for x in per_layer[l]],dim=-2)
        V=torch.cat([x[1] for x in per_layer[l]],dim=-2)
        layers.append((K,V))
    return tuple(layers),ranges,offset

POPS={}
for N in N_LIST:
    inds=ORDER[:N];layers,ranges,total=compose_population(inds);sha=tensor_sha(layers)
    POPS[N]={"indices":inds,"layers":layers,"ranges":ranges,"tokens":total,"sha":sha}
    print(f"      N={N:02d} tokens={total:04d} SEALED_SHA={sha}")
print("      Populations sealed. Bundan sonra query'ye göre cache işlemi YOK.")

print("\n[6/10] Native binding — sealed numerical populations...")
RB=[]
for N in N_LIST:
    pop=POPS[N];before=tensor_sha(pop["layers"])
    if before!=pop["sha"]:raise RuntimeError("Population seal query öncesi bozuldu.")
    print(f"\n      N={N:02d} | cartridges={len(pop['indices'])} | tokens={pop['tokens']}")
    okn=0
    for j,ci in enumerate(pop["indices"]):
        c=CAR[ci];ans=answer_from_layers(pop["layers"],c["query"]);ok=hit(ans,c["gold"]);okn+=int(ok)
        RB.append({"N":N,"ci":ci,"world":c["world"],"type":c["type"],"gold":c["gold"],"answer":ans,"ok":ok})
        print(f"      {j+1:02d}/{N:02d} C{ci:02d} W{c['world']:02d} {c['type']:8s} gold={c['gold']:10s} -> {ans!r} {'PASS' if ok else 'FAIL'}")
    after=tensor_sha(pop["layers"])
    if after!=pop["sha"]:raise RuntimeError("SEALED population query sırasında değişti.")
    print(f"      N={N:02d} ACC={okn}/{N}={okn/N:.6f} | CACHE_SHA_UNCHANGED=PASS")

print("\n[7/10] Sonuç kırılımı...")
SUMMARY={}
for N in N_LIST:
    rr=[x for x in RB if x["N"]==N];correct=sum(x["ok"] for x in rr);a=correct/len(rr)
    SUMMARY[str(N)]={"correct":correct,"total":len(rr),"acc":a}
    print(f"      N={N:02d}: {correct:02d}/{len(rr):02d} = {a:.6f}")
rr96=[x for x in RB if x["N"]==96]
for typ in ["CURRENT","FORMER","NEAR","ROLE"]:
    z=[x for x in rr96 if x["type"]==typ];correct=sum(x["ok"] for x in z)
    print(f"      N=96 {typ:8s}: {correct:02d}/{len(z):02d} = {correct/len(z):.6f}")

print("\n[8/10] No-memory kontrolü...")
@torch.no_grad()
def no_memory_answer(q):
    text=f"<s>[INST] {SYSTEM}\n\nQUESTION:\n{q}\n\nANSWER: [/INST]"
    x=tok(text,return_tensors="pt",add_special_tokens=False).to(DEVICE);n=int(x.input_ids.shape[1])
    y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    ans=tok.decode(y[0,n:],skip_special_tokens=True).strip();del x,y
    return ans
NM=[]
for ci in ORDER[:16]:
    c=CAR[ci];ans=no_memory_answer(c["query"]);ok=hit(ans,c["gold"]);NM.append(ok)
    print(f"      C{ci:02d} {c['type']:8s} gold={c['gold']:10s} -> {ans!r} {'HIT' if ok else 'MISS'}")
print(f"      NO-MEMORY={sum(NM)}/{len(NM)}")

print("\n[9/10] Anti-retrieval / seal audit...")
seal_ok=True
for N in N_LIST:seal_ok=seal_ok and tensor_sha(POPS[N]["layers"])==POPS[N]["sha"]
print("      Population composed before query      : EVET")
print("      Population SHA unchanged              :","PASS" if seal_ok else "FAIL")
print("      Query-dependent cartridge selection   : YOK")
print("      Query-dependent cache transform       : YOK")
print("      Query-dependent cartridge ordering    : YOK")
print("      Query-dependent mask                  : YOK")
print("      External score / cosine / ANN         : YOK")
print("      Router / classifier / learned gate    : YOK")
print("      Source text replay at query           : YOK")
print("      Candidate source forward at query     : YOK")
print("      RAG / external file search            : YOK")
print("      Training / LoRA / optimizer           : YOK")

print("\n[10/10] Sentinel / mühür...")
S1=sentinel();WOK=S0==S1;ACC96=SUMMARY["96"]["acc"]
FINAL={"test":TEST,"parent548":PARENT548,"lock_sha":LOCK_SHA,"gate_a":GATE_A,"gate_a_same":GA_SAME,"gate_a_total":len(GA),"gate_a_kmax":GA_K,"gate_a_vmax":GA_V,"summary":SUMMARY,"acc96":ACC96,"no_memory_hits":sum(NM),"no_memory_total":len(NM),"population_seal":seal_ok,"weight_sentinel":WOK,"trainable":sum(int(p.requires_grad) for p in model.parameters()),"retrieval":False,"router":False,"scorer":False,"query_dependent_cache_op":False,"query_dependent_mask":False,"rag":False,"training":False}
RESULT_SHA=hashlib.sha256(json.dumps(FINAL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      GATE-A physical validation :","PASS" if GATE_A else "FAIL")
print("      Population seal            :","PASS" if seal_ok else "FAIL")
print("      Weight sentinel            :","PASS" if WOK else "FAIL")
print("      Trainable tensors          :",sum(int(p.requires_grad) for p in model.parameters()))
print("\n"+"="*178)
print("TEST549 SONUÇ — MULTI-CARTRIDGE NUMERICAL NATIVE BINDING")
print("="*178)
print("MODEL                         :",MODEL_ID)
print("CARTRIDGES FORGED             :",len(CAR))
print("GATE-A REBASE ANSWER SAME     :",f"{GA_SAME}/{len(GA)}")
print("GATE-A K MAX ERROR            :",f"{GA_K:.8f}")
print("GATE-A V MAX ERROR            :",f"{GA_V:.8f}")
for N in N_LIST:
    s=SUMMARY[str(N)]
    print(f"N={N:<3d} NATIVE BINDING ACC       : {s['correct']}/{s['total']} = {s['acc']:.6f}")
print("NO-MEMORY CONTROL             :",f"{sum(NM)}/{len(NM)}")
print("POPULATION SEAL               :","PASS" if seal_ok else "FAIL")
print("WEIGHT SENTINEL               :","PASS" if WOK else "FAIL")
print("TEST548 RESULT SHA            :",PARENT548)
print("TEST549 ÖN KİLİT SHA          :",LOCK_SHA)
print("TEST549 RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
if GATE_A and seal_ok and WOK:
    print("KARAR                         : TEST549_MULTI_CARTRIDGE_NATIVE_BINDING_MEASURED")
else:
    print("KARAR                         : TEST549_PHYSICAL_VALIDITY_FAILURE")
print("="*178)
