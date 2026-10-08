# TEST548 — AKBASCORE MAM · VARAN 1 · TEXT→NUMERICAL PKV SUBSTITUTION BRIDGE
# SOURCE FACT READ ONCE → FROZEN NATIVE PKV → SOURCE TEXT REMOVED → QUERY OVER NUMERICAL PKV
# TEXT BASELINE vs PKV CARTRIDGE · CURRENT / FORMER / NEAR / ROLE
# EXTERNAL FILE YOK · RAG YOK · RETRIEVAL YOK · ROUTER YOK · SCORER YOK · TRAINING YOK
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="548";SEED=548548;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT547="b3de5026c86098a4f3a147a2819eb14e10349bf8e5e2f355aae306ee0f2eedaf"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
print("="*174)
print("TEST548 — AKBASCORE MAM · VARAN 1 · TEXT→NUMERICAL PKV SUBSTITUTION BRIDGE")
print("SOURCE READ ONCE → NATIVE PKV → SOURCE TEXT REMOVED → QUERY OVER NUMERICAL CARTRIDGE")
print("EXTERNAL FILE YOK · RAG YOK · RETRIEVAL YOK · ROUTER YOK · SCORER YOK · TRAINING YOK")
print("="*174);T0=time.perf_counter()

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
        ("ROLE",f"The current capital of {role_target} is {s}.",s)
    ]
    queries=[
        ("CURRENT",f"What is the current capital of {s}?",current),
        ("FORMER",f"What was the former capital of {s}?",former),
        ("NEAR",f"What is the largest city of {s}?",near),
        ("ROLE",f"What is the current capital of {role_target}?",s)
    ]
    W.append({"id":i,"subject":s,"facts":facts,"queries":queries})

LOCK={"test":TEST,"parent547":PARENT547,"model":MODEL_ID,"seed":SEED,"worlds":24,"relations":["CURRENT","FORMER","NEAR","ROLE"],"source_read_once":True,"source_absent_at_query":True,"native_pkv":True,"text_baseline":True,"retrieval":False,"rag":False,"external_file":False,"router":False,"scorer":False,"threshold":False,"training":False,"lora":False,"optimizer":False}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      LOCK SHA:",LOCK_SHA)

SYSTEM="Answer the question using only the information provided. Give only the requested name and nothing else."
def norm(s):
    s=s.casefold().strip();s=re.sub(r"[^\w\s-]"," ",s,flags=re.UNICODE);return " ".join(s.split())
def hit(out,gold):return re.search(r"(?<!\w)"+re.escape(norm(gold))+r"(?!\w)",norm(out)) is not None

# Transformers v4/v5 DynamicCache compatibility.
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
        return tuple((k,v) for k,v in zip(pkv.key_cache,pkv.value_cache))
    if hasattr(pkv,"to_legacy_cache"):
        p=pkv.to_legacy_cache()
        if isinstance(p,(tuple,list)):return tuple(p)
    if isinstance(pkv,(tuple,list)):return tuple(pkv)
    raise RuntimeError(f"Desteklenmeyen cache tipi: {type(pkv)}")

def cache_len(pkv):
    if hasattr(pkv,"get_seq_length"):
        try:return int(pkv.get_seq_length())
        except TypeError:return int(pkv.get_seq_length(0))
    p=cache_layers(pkv)
    if not p:raise RuntimeError("Boş PKV cache.")
    return int(p[0][0].shape[-2])

def cache_audit(pkv,expected=None):
    layers=cache_layers(pkv)
    if len(layers)!=NL:raise RuntimeError(f"PKV layer sayısı yanlış: {len(layers)} != {NL}")
    lengths=[]
    for li,(k,v) in enumerate(layers):
        if k is None or v is None:raise RuntimeError(f"L{li}: boş K/V.")
        if k.ndim!=4 or v.ndim!=4:raise RuntimeError(f"L{li}: K/V rank yanlış: K{k.shape} V{v.shape}")
        if tuple(k.shape)!=tuple(v.shape):raise RuntimeError(f"L{li}: K/V shape farklı: K{k.shape} V{v.shape}")
        if k.shape[0]!=1 or k.shape[1]!=KVH or k.shape[-1]!=HD:
            raise RuntimeError(f"L{li}: beklenmeyen PKV shape: {tuple(k.shape)}")
        lengths.append(int(k.shape[-2]))
    if len(set(lengths))!=1:raise RuntimeError(f"Katman PKV uzunlukları farklı: {sorted(set(lengths))}")
    n=lengths[0]
    if expected is not None and n!=expected:raise RuntimeError(f"PKV uzunluğu {n}, source token uzunluğu {expected}.")
    if cache_len(pkv)!=n:raise RuntimeError(f"Cache API uzunluğu {cache_len(pkv)} != tensor uzunluğu {n}.")
    return n,[tuple(z.shape) for z in layers[0]]

# Source text is read only while forging the cartridge.
# Query-time model input contains only query_suffix(q); source text/token IDs are not replayed.
def source_prefix(fact):
    return f"<s>[INST] {SYSTEM}\n\nINFORMATION:\n{fact}\n\n"
def query_suffix(q):
    return f"QUESTION:\n{q}\n\nANSWER: [/INST]"
def full_text_prompt(fact,q):
    return source_prefix(fact)+query_suffix(q)

@torch.no_grad()
def forge(fact):
    text=source_prefix(fact)
    x=tok(text,return_tensors="pt",add_special_tokens=False).to(DEVICE)
    out=model(**x,use_cache=True,return_dict=True)
    pkv=out.past_key_values
    n=int(x.input_ids.shape[1])
    cn,shape=cache_audit(pkv,n)
    if cn!=n:raise RuntimeError("Forge cache/token uzunluk uyuşmazlığı.")
    del out,x
    return pkv,n,shape

@torch.no_grad()
def text_answer(fact,q):
    p=full_text_prompt(fact,q)
    x=tok(p,return_tensors="pt",add_special_tokens=False).to(DEVICE)
    n=int(x.input_ids.shape[1])
    y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    ans=tok.decode(y[0,n:],skip_special_tokens=True).strip()
    del x,y
    return ans

@torch.no_grad()
def pkv_answer(pkv,q):
    # IMPORTANT: DynamicCache is mutable. This function consumes/grows this cartridge once.
    # TEST548 uses every cartridge exactly once, so no clone/replay is required.
    suffix=query_suffix(q)
    x=tok(suffix,return_tensors="pt",add_special_tokens=False).to(DEVICE)
    ids=x.input_ids
    past=cache_len(pkv);qlen=int(ids.shape[1])
    if qlen<1:raise RuntimeError("Boş query suffix.")
    att=torch.ones((1,past+qlen),dtype=torch.long,device=DEVICE)
    cache_position=torch.arange(past,past+qlen,dtype=torch.long,device=DEVICE)
    out=model(input_ids=ids,past_key_values=pkv,attention_mask=att,cache_position=cache_position,use_cache=True,return_dict=True)
    cache=out.past_key_values
    expected=past+qlen
    if cache_len(cache)!=expected:raise RuntimeError(f"Query sonrası cache uzunluğu yanlış: {cache_len(cache)} != {expected}")
    nxt=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    gen=[]
    eos_ids=tok.eos_token_id
    if eos_ids is None:eos=set()
    elif isinstance(eos_ids,(list,tuple,set)):eos=set(int(z) for z in eos_ids)
    else:eos={int(eos_ids)}
    for step in range(MAX_NEW):
        gen.append(nxt)
        if int(nxt.item()) in eos:break
        past=cache_len(cache)
        att=torch.ones((1,past+1),dtype=torch.long,device=DEVICE)
        cache_position=torch.tensor([past],dtype=torch.long,device=DEVICE)
        out=model(input_ids=nxt,past_key_values=cache,attention_mask=att,cache_position=cache_position,use_cache=True,return_dict=True)
        cache=out.past_key_values
        if cache_len(cache)!=past+1:raise RuntimeError(f"Generation cache büyümedi: {cache_len(cache)} != {past+1}")
        nxt=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    g=torch.cat(gen,dim=1)
    ans=tok.decode(g[0],skip_special_tokens=True).strip()
    del x,out,g
    return ans

print("\n[2/9] Protokol...")
print("      Her fact ayrı olarak yalnızca bir kez okunacak.")
print("      Fact → frozen Mistral → native numerical PKV cartridge.")
print("      Query aşamasında source fact text/tokenları inputta YOK.")
print("      Aynı fact/query TEXT baseline ile karşılaştırılacak.")
print("      Retrieval yok: her deneyde hangi tek cartridge'in test edildiği protokol tarafından sabit.")
print("      Bu test cartridge discovery değil; text→numerical substitution fiziksel köprü testidir.")

print("\n[3/9] 96 native PKV cartridge forge...")
CAR={}
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/24 {w['subject']}")
    for typ,fact,gold in w["facts"]:
        pkv,n,shape=forge(fact)
        CAR[(wi,typ)]={"pkv":pkv,"source_tokens":n,"shape":shape,"fact_sha":hashlib.sha256(fact.encode()).hexdigest()}
print(f"      Cartridges={len(CAR)}")
lens=[v["source_tokens"] for v in CAR.values()]
print(f"      Source prefix token length min/mean/max = {min(lens)}/{np.mean(lens):.2f}/{max(lens)}")
sample=next(iter(CAR.values()))
print("      Cache type:",type(sample["pkv"]).__name__)
print("      Sample layer0 K/V shapes:",sample["shape"])

print("\n[4/9] TEXT baseline...")
RES=[]
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/24 {w['subject']}")
    fmap={x[0]:x for x in w["facts"]}
    for typ,q,gold in w["queries"]:
        fact=fmap[typ][1]
        ans=text_answer(fact,q)
        RES.append({"world":wi,"type":typ,"mode":"TEXT","gold":gold,"answer":ans,"ok":hit(ans,gold)})

print("\n[5/9] NUMERICAL PKV substitution...")
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/24 {w['subject']}")
    for typ,q,gold in w["queries"]:
        car=CAR[(wi,typ)]
        before=cache_len(car["pkv"])
        if before!=car["source_tokens"]:raise RuntimeError(f"W{wi:02d} {typ}: cartridge query öncesi değişmiş: {before} != {car['source_tokens']}")
        ans=pkv_answer(car["pkv"],q)
        RES.append({"world":wi,"type":typ,"mode":"PKV","gold":gold,"answer":ans,"ok":hit(ans,gold)})

print("\n[6/9] Davranış...")
def rows(mode=None,typ=None):
    return [r for r in RES if (mode is None or r["mode"]==mode) and (typ is None or r["type"]==typ)]
def acc(rr):return sum(x["ok"] for x in rr)/len(rr) if rr else float("nan")
SUMMARY={}
for typ in ["CURRENT","FORMER","NEAR","ROLE"]:
    a=rows("TEXT",typ);b=rows("PKV",typ)
    SUMMARY[typ]={"text":acc(a),"pkv":acc(b)}
    print(f"      {typ:8s} TEXT {sum(x['ok'] for x in a):02d}/{len(a):02d}={acc(a):.4f} | PKV {sum(x['ok'] for x in b):02d}/{len(b):02d}={acc(b):.4f} | Δ={acc(b)-acc(a):+.4f}")
A=rows("TEXT");B=rows("PKV")
print(f"\n      ALL      TEXT {sum(x['ok'] for x in A):02d}/{len(A):02d}={acc(A):.4f} | PKV {sum(x['ok'] for x in B):02d}/{len(B):02d}={acc(B):.4f} | Δ={acc(B)-acc(A):+.4f}")

print("\n[7/9] Paired substitution audit...")
T={(r["world"],r["type"]):r for r in RES if r["mode"]=="TEXT"}
P={(r["world"],r["type"]):r for r in RES if r["mode"]=="PKV"}
both=text_only=pkv_only=neither=same=0;EX=[]
for k in T:
    a,b=T[k],P[k]
    if a["ok"] and b["ok"]:both+=1
    elif a["ok"] and not b["ok"]:text_only+=1
    elif not a["ok"] and b["ok"]:pkv_only+=1
    else:neither+=1
    same+=int(norm(a["answer"])==norm(b["answer"]))
    if a["ok"]!=b["ok"] and len(EX)<24:EX.append((k,a["gold"],a["answer"],b["answer"],a["ok"],b["ok"]))
print(f"      BOTH CORRECT : {both}/{len(T)}")
print(f"      TEXT ONLY    : {text_only}/{len(T)}")
print(f"      PKV ONLY     : {pkv_only}/{len(T)}")
print(f"      NEITHER      : {neither}/{len(T)}")
print(f"      SAME OUTPUT  : {same}/{len(T)}")
if EX:
    print("\n      İlk substitution farkları:")
    for k,g,a,b,oa,ob in EX:print(f"      W{k[0]:02d} {k[1]:8s} gold={g:10s} TEXT={a!r} ({oa}) PKV={b!r} ({ob})")

print("\n[8/9] Source-absence / cartridge audit...")
leaks=0
for wi,w in enumerate(W):
    fmap={x[0]:x for x in w["facts"]}
    for typ,q,gold in w["queries"]:
        suffix=query_suffix(q);fact=fmap[typ][1]
        if fact in suffix:leaks+=1
print("      Source full fact present in query suffix :",leaks)
print("      Cartridge payload                        : model-native PKV only")
print("      External text/file retrieval             : YOK")
print("      Candidate bank/search                    : YOK")
print("      Query-time source forward                : YOK")
print("      Query-time source token replay           : YOK")

print("\n[9/9] Sentinel / mühür...")
S1=sentinel();OK=S0==S1
FINAL={"test":TEST,"parent547":PARENT547,"lock_sha":LOCK_SHA,"summary":SUMMARY,"text_acc":acc(A),"pkv_acc":acc(B),"both":both,"text_only":text_only,"pkv_only":pkv_only,"neither":neither,"same_output":same,"source_fact_leaks":leaks,"weight_sentinel":OK,"trainable":0,"source_absent_at_query":True,"native_pkv":True,"retrieval":False,"rag":False,"router":False,"scorer":False,"training":False}
RESULT_SHA=hashlib.sha256(json.dumps(FINAL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Weight sentinel             :","PASS" if OK else "FAIL")
print("      Trainable tensors           :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Source absent at query      : EVET")
print("      Native numerical PKV        : EVET")
print("      Query-time source replay    : YOK")
print("      External file / RAG         : YOK")
print("      Retrieval / candidate search: YOK")
print("      Router / scorer / threshold : YOK")
print("      Training/LoRA/optimizer     : YOK")
print("\n"+"="*174)
print("TEST548 SONUÇ — TEXT→NUMERICAL PKV SUBSTITUTION BRIDGE")
print("="*174)
print("MODEL                         :",MODEL_ID)
print("CARTRIDGES                    :",len(CAR))
print("PAIRED CASES                  :",len(T))
print("TEXT ACC                      :",f"{acc(A):.6f}")
print("PKV ACC                       :",f"{acc(B):.6f}")
print("BOTH CORRECT                  :",f"{both}/{len(T)}")
print("TEXT ONLY                     :",f"{text_only}/{len(T)}")
print("PKV ONLY                      :",f"{pkv_only}/{len(T)}")
print("NEITHER                       :",f"{neither}/{len(T)}")
print("SAME OUTPUT                   :",f"{same}/{len(T)}")
print("SOURCE FACT LEAKS             :",leaks)
print("WEIGHT SENTINEL               :","PASS" if OK else "FAIL")
print("TEST547 RESULT SHA            :",PARENT547)
print("TEST548 ÖN KİLİT SHA          :",LOCK_SHA)
print("TEST548 RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                         : TEST548_NUMERICAL_PKV_SUBSTITUTION_BRIDGE_TAMAMLANDI")
print("="*174)
