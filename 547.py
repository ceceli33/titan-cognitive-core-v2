# TEST547 — AKBASCORE MAM · VARAN 1 · NATIVE BINDING X-RAY
# SAME-FORWARD CONTEXT→QUERY ATTENTION · VALUE TRANSPORT · RESIDUAL FLOW
# TEST546A PANELİ · CURRENT / FORMER / NEAR / ROLE · FORWARD / REVERSE
# RETRIEVAL YOK · SEÇİM YOK · ROUTER YOK · SCORER YOK · EŞİK YOK · TRAINING YOK
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re,math
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="547";SEED=547547;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT546="d2601e1e9710872d76fb84dcee43b5ec095ceaec77ee573a8268f559ea43e798"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
print("="*170)
print("TEST547 — AKBASCORE MAM · VARAN 1 · NATIVE BINDING X-RAY")
print("SAME-FORWARD CONTEXT→QUERY ATTENTION · VALUE TRANSPORT · RESIDUAL FLOW")
print("TEST546A PANELİ · RETRIEVAL YOK · SEÇİM YOK · ROUTER YOK · SCORER YOK · TRAINING YOK")
print("="*170);T0=time.perf_counter()
print("\n[1/9] Donmuş Mistral...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="eager",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | trainable=0")
print("      Attention implementation: EAGER (gerçek attention weights X-ray için)")
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
    cs,cc=BASE[(i+1)%len(BASE)][0],BASE[(i+1)%len(BASE)][1]
    facts=[
        ("CURRENT",f"The current capital of {s} is {current}."),
        ("NEAR",f"The largest city of {s} is {near}."),
        ("FORMER",f"The former capital of {s} was {former}."),
        ("ROLE",f"The current capital of {role_target} is {s}."),
        ("CROSS",f"The current capital of {cs} is {cc}.")
    ]
    queries=[
        ("CURRENT",f"What is the current capital of {s}?",current),
        ("FORMER",f"What was the former capital of {s}?",former),
        ("NEAR",f"What is the largest city of {s}?",near),
        ("ROLE",f"What is the current capital of {role_target}?",s)
    ]
    W.append({"id":i,"subject":s,"facts":facts,"queries":queries})
LOCK={"test":TEST,"parent546":PARENT546,"model":MODEL_ID,"seed":SEED,"worlds":24,"facts":5,"queries":["CURRENT","FORMER","NEAR","ROLE"],"orders":["FORWARD","REVERSE"],"same_forward":True,"actual_attention":True,"value_transport":True,"residual_flow":True,"retrieval":False,"selection":False,"router":False,"scorer":False,"threshold":False,"training":False,"gold_selection":False}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      LOCK SHA:",LOCK_SHA)

SYSTEM="Answer the question using only the facts in the provided context. Give only the requested name and nothing else."
def build(w,q,reverse=False):
    facts=list(reversed(w["facts"])) if reverse else list(w["facts"])
    prefix="CONTEXT:\n"
    pieces=[prefix];spans={};pos=len(prefix)
    for label,text in facts:
        start=pos;pieces.append(text+"\n");pos+=len(text)+1;spans[label]=(start,pos-1)
    qprefix="\nQUESTION:\n";pieces.append(qprefix);pos+=len(qprefix);qstart=pos;pieces.append(q);pos+=len(q);qend=pos
    suffix="\n\nANSWER:";pieces.append(suffix)
    user="".join(pieces)
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":user}]
    full=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    # Robustly locate each literal fact and query in final rendered chat text, then map chars→tokens.
    enc=tok(full,return_tensors="pt",add_special_tokens=False,return_offsets_mapping=True)
    offs=enc.pop("offset_mapping")[0].tolist()
    def char_to_tokens(a,b):
        ids=[j for j,(x,y) in enumerate(offs) if y>a and x<b and y>x]
        if not ids:raise RuntimeError(f"Span token bulunamadı: {a}:{b}")
        return ids
    tspans={}
    for label,text in facts:
        a=full.find(text)
        if a<0:raise RuntimeError("Fact final prompt içinde bulunamadı.")
        tspans[label]=char_to_tokens(a,a+len(text))
    qa=full.rfind(q)
    if qa<0:raise RuntimeError("Query final prompt içinde bulunamadı.")
    qtokens=char_to_tokens(qa,qa+len(q))
    return full,{k:torch.tensor(v,dtype=torch.long) for k,v in tspans.items()},torch.tensor(qtokens,dtype=torch.long)

def norm(s):
    s=s.casefold().strip();s=re.sub(r"[^\w\s-]"," ",s,flags=re.UNICODE);return " ".join(s.split())
def contains_gold(out,gold):return re.search(r"(?<!\w)"+re.escape(norm(gold))+r"(?!\w)",norm(out)) is not None

# Hooks capture residual input/output and actual V projections. Attention probabilities come from output_attentions.
CACHE={}
def make_pre(li):
    def hook(mod,args):
        CACHE.setdefault(li,{})["rin"]=args[0].detach()
    return hook
def make_post(li):
    def hook(mod,args,out):
        y=out[0] if isinstance(out,(tuple,list)) else out
        CACHE.setdefault(li,{})["rout"]=y.detach()
    return hook
PRE=[];POST=[]
for li,layer in enumerate(model.model.layers):
    PRE.append(layer.register_forward_pre_hook(make_pre(li)))
    POST.append(layer.register_forward_hook(make_post(li)))

def expand_kv(x):
    # [..., KVH, T, HD] -> [..., QH, T, HD]
    return x.repeat_interleave(QH//KVH,dim=-3)

@torch.no_grad()
def xray(w,q,gold,reverse):
    global CACHE;CACHE={}
    prompt,spans,qtok=build(w,q,reverse)
    x=tok(prompt,return_tensors="pt",add_special_tokens=False).to(DEVICE)
    out=model(**x,output_attentions=True,use_cache=False,return_dict=True)
    atts=out.attentions
    if atts is None or len(atts)!=NL:raise RuntimeError("Attention weights dönmedi.")
    T=x.input_ids.shape[1]
    qidx=qtok.to(DEVICE);qidx=qidx[qidx<T]
    if qidx.numel()==0:raise RuntimeError("Query token span boş.")
    labels=["CURRENT","NEAR","FORMER","ROLE","CROSS"]
    MASS=np.zeros((NL,QH,len(labels)),np.float32)
    DENS=np.zeros_like(MASS);OVN=np.zeros_like(MASS)
    RFLOW=np.zeros(NL,np.float32)
    for li in range(NL):
        A=atts[li][0].float() # [H,T,T]
        aq=A[:,qidx,:].mean(dim=1) # [H,T], actual query-token attention
        rin=CACHE[li]["rin"][0].float();rout=CACHE[li]["rout"][0].float()
        RFLOW[li]=float((rout[qidx]-rin[qidx]).norm(dim=-1).mean().cpu())
        layer=model.model.layers[li];hs=CACHE[li]["rin"]
        hn=layer.input_layernorm(hs)
        v=layer.self_attn.v_proj(hn).view(1,T,KVH,HD).transpose(1,2)[0].float() # [KVH,T,D]
        v=expand_kv(v) # [QH,T,D]
        Wo=layer.self_attn.o_proj.weight.detach().float().view(H,QH,HD) # output_dim,QH,HD
        for fi,label in enumerate(labels):
            idx=spans[label].to(DEVICE);idx=idx[idx<T]
            MASS[li,:,fi]=aq[:,idx].sum(dim=-1).cpu().numpy()
            DENS[li,:,fi]=aq[:,idx].mean(dim=-1).cpu().numpy()
            # Actual attention-weighted V payload from this fact span for each query head,
            # projected through that head's corresponding O-projection slice.
            av=A[:,qidx,:][:,:,idx].mean(dim=1) # [QH,S]
            payload=torch.einsum("hs,hsd->hd",av,v[:,idx,:]) # [QH,D]
            op=torch.einsum("ohd,hd->ho",Wo,payload) # [QH,Hout]
            OVN[li,:,fi]=op.norm(dim=-1).cpu().numpy()
    # Free generation separately, same prompt/context; behavior only.
    y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    ans=tok.decode(y[0,T:],skip_special_tokens=True).strip()
    ok=contains_gold(ans,gold)
    del out,atts,y,x
    CACHE={}
    return {"mass":MASS,"dens":DENS,"ovn":OVN,"rflow":RFLOW,"answer":ans,"ok":ok}

print("\n[2/9] Same-forward panel...")
print("      24 worlds × 4 relations × 2 orders = 192 native-binding forwards")
print("      Query ve beş fact AYNI Transformer forward'ında.")
print("      Gold hiçbir X-ray hesabında kullanılmıyor; yalnız davranış auditinde.")

print("\n[3/9] Native-binding X-ray...")
R=[]
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/24 {w['subject']}")
    for reverse in [False,True]:
        for typ,q,gold in w["queries"]:
            z=xray(w,q,gold,reverse)
            R.append({"world":wi,"subject":w["subject"],"order":"REVERSE" if reverse else "FORWARD","type":typ,"gold":gold,**z})

for h in PRE:h.remove()
for h in POST:h.remove()

print("\n[4/9] Davranış audit...")
for order in ["FORWARD","REVERSE"]:
    print(f"\n      {order}")
    for typ in ["CURRENT","FORMER","NEAR","ROLE"]:
        rr=[r for r in R if r["order"]==order and r["type"]==typ]
        print(f"      {typ:8s} {sum(x['ok'] for x in rr):02d}/{len(rr):02d}")
print("\n      Not: contains-gold audit; TEST546A exact-name skorunun yerine geçmez.")

LAB=["CURRENT","NEAR","FORMER","ROLE","CROSS"];LI={x:i for i,x in enumerate(LAB)}
def intended(typ):return typ
def aggregate(metric,rows):
    # Distributed: first average heads, then layers, then examples. No winner/head/layer selection.
    vals=[]
    for r in rows:
        a=r[metric]
        if metric in ("mass","dens","ovn"):vals.append(a.mean(axis=(0,1)))
    return np.mean(vals,axis=0) if vals else np.full(len(LAB),np.nan)
print("\n[5/9] Gerçek context→query attention / transport...")
SUMMARY={}
for order in ["FORWARD","REVERSE"]:
    print(f"\n      {order}")
    for typ in ["CURRENT","FORMER","NEAR","ROLE"]:
        rr=[r for r in R if r["order"]==order and r["type"]==typ]
        m=aggregate("mass",rr);d=aggregate("dens",rr);o=aggregate("ovn",rr);ti=LI[intended(typ)]
        other=[j for j in range(5) if j!=ti]
        md=float(m[ti]-m[other].mean());dd=float(d[ti]-d[other].mean());od=float(o[ti]-o[other].mean())
        SUMMARY[f"{order}_{typ}"]={"mass_target_other":md,"density_target_other":dd,"ov_target_other":od}
        print(f"      {typ:8s} ATT_MASS Δ={md:+.8f} ATT_DENS Δ={dd:+.8f} OV_TRANSPORT Δ={od:+.8f}")

print("\n[6/9] Başarılı ↔ başarısız doğal boundary...")
# ROLE is especially useful because 546A showed order sensitivity.
role_ok=[r for r in R if r["type"]=="ROLE" and r["ok"]]
role_bad=[r for r in R if r["type"]=="ROLE" and not r["ok"]]
def target_delta(r,metric):
    a=r[metric].mean(axis=(0,1));ti=LI[r["type"]];return float(a[ti]-np.mean([a[j] for j in range(5) if j!=ti]))
for metric in ["mass","dens","ovn"]:
    a=np.mean([target_delta(r,metric) for r in role_ok]) if role_ok else np.nan
    b=np.mean([target_delta(r,metric) for r in role_bad]) if role_bad else np.nan
    print(f"      {metric.upper():5s} ROLE correct={a:+.8f} wrong={b:+.8f} correct-wrong={a-b:+.8f}")
print(f"      ROLE correct n={len(role_ok)} | wrong n={len(role_bad)}")

print("\n[7/9] Katman profili — X-ray only, hiçbir layer/head seçilmiyor...")
for li in range(NL):
    vals=[]
    for r in R:
        ti=LI[r["type"]];a=r["mass"][li].mean(axis=0);vals.append(float(a[ti]-np.mean([a[j] for j in range(5) if j!=ti])))
    ov=[]
    for r in R:
        ti=LI[r["type"]];a=r["ovn"][li].mean(axis=0);ov.append(float(a[ti]-np.mean([a[j] for j in range(5) if j!=ti])))
    rf=np.mean([r["rflow"][li] for r in R])
    print(f"      L{li:02d} ATT_TARGET-OTHER={np.mean(vals):+.8f} OV_TARGET-OTHER={np.mean(ov):+.8f} RES_FLOW={rf:.6f}")

print("\n[8/9] Order-paired native-binding değişimi...")
# Same world/type, forward vs reverse. No selection.
for typ in ["CURRENT","FORMER","NEAR","ROLE"]:
    F={(r["world"],r["type"]):r for r in R if r["order"]=="FORWARD" and r["type"]==typ}
    V={(r["world"],r["type"]):r for r in R if r["order"]=="REVERSE" and r["type"]==typ}
    dm=[];do=[];flip=0
    for k in F:
        dm.append(target_delta(V[k],"mass")-target_delta(F[k],"mass"))
        do.append(target_delta(V[k],"ovn")-target_delta(F[k],"ovn"))
        flip+=int(F[k]["ok"]!=V[k]["ok"])
    print(f"      {typ:8s} REVERSE-FORWARD ATTΔ={np.mean(dm):+.8f} OVΔ={np.mean(do):+.8f} behavior_flips={flip}/{len(dm)}")

print("\n[9/9] Sentinel / mühür...")
S1=sentinel();OK=S0==S1
BEH={f"{o}_{t}":sum(r["ok"] for r in R if r["order"]==o and r["type"]==t) for o in ["FORWARD","REVERSE"] for t in ["CURRENT","FORMER","NEAR","ROLE"]}
FINAL={"test":TEST,"parent546":PARENT546,"lock_sha":LOCK_SHA,"behavior":BEH,"summary":SUMMARY,"role_correct":len(role_ok),"role_wrong":len(role_bad),"weight_sentinel":OK,"trainable":0,"same_forward":True,"retrieval":False,"selection":False,"router":False,"scorer":False,"threshold":False,"training":False}
RESULT_SHA=hashlib.sha256(json.dumps(FINAL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Weight sentinel              :","PASS" if OK else "FAIL")
print("      Trainable tensors            :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Same-forward context/query   : EVET")
print("      Actual attention weights     : EVET")
print("      Attention-weighted V→O X-ray : EVET")
print("      Retrieval / argmax / winner  : YOK")
print("      Router / scorer / threshold  : YOK")
print("      Tek layer/head adresi        : YOK")
print("      Gold X-ray selection         : YOK")
print("      Training/LoRA/optimizer      : YOK")
print("\n"+"="*170)
print("TEST547 SONUÇ — SAME-FORWARD NATIVE BINDING X-RAY")
print("="*170)
print("MODEL                         :",MODEL_ID)
print("WORLDS                        :",len(W))
print("XRAY FORWARDS                 :",len(R))
print("ROLE CORRECT / WRONG          :",f"{len(role_ok)} / {len(role_bad)}")
print("WEIGHT SENTINEL               :","PASS" if OK else "FAIL")
print("TEST546A RESULT SHA           :",PARENT546)
print("TEST547 ÖN KİLİT SHA          :",LOCK_SHA)
print("TEST547 RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                         : TEST547_NATIVE_BINDING_XRAY_TAMAMLANDI")
print("="*170)
