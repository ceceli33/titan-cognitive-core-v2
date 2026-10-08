# TEST546A — AKBASCORE MAM · VARAN 1 · FULL-CONTEXT UPPER BOUND
# BEHAVIOR FIRST · NATIVE RELATIONAL BINDING · CONFLICT PANEL
# CURRENT / NEAR / FORMER / ROLE / CROSS · DIRECT / PARAPHRASE / TURKISH
# RETRIEVAL YOK · CARTRIDGE YOK · ROUTER YOK · SCORER YOK · TRAINING YOK
import os,sys,subprocess,importlib.util,random,time,hashlib,json,re
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="546A";SEED=546546;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT545="464d31f054d0e42c7bdbc5b936d17e81c14052e6bc240ec5dbefd63932d616f1"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;MAX_NEW=16
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
os.environ["TOKENIZERS_PARALLELISM"]="false"
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
print("="*160)
print("TEST546A — AKBASCORE MAM · VARAN 1 · FULL-CONTEXT UPPER BOUND")
print("NATIVE RELATIONAL BINDING · CURRENT / NEAR / FORMER / ROLE / CROSS · DIRECT / PARAPHRASE / TURKISH")
print("RETRIEVAL YOK · CARTRIDGE YOK · ROUTER YOK · SCORER YOK · TRAINING YOK")
print("="*160);T0=time.perf_counter()
print("\n[1/8] Donmuş Mistral...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;NL=len(model.model.layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | trainable=0")
FP=[model.model.layers[0].self_attn.q_proj.weight,model.model.layers[8].self_attn.o_proj.weight,model.model.layers[16].mlp.down_proj.weight,model.model.layers[24].self_attn.o_proj.weight,model.model.layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()

# Every value is unique across the complete panel.
# Tuple: SUBJECT, CURRENT, LARGEST/NEAR, FORMER, ROLE_TARGET
BASE=[
("Zorvan","Melket","Dravel","Oakhaven","Pelnor"),
("Kelvar","Nareth","Solven","Branik","Tarsen"),
("Tarev","Luneth","Varos","Cedran","Mireth"),
("Belnor","Arven","Dorel","Kesmar","Falven"),
("Ravik","Selora","Terven","Maldor","Nerik"),
("Nemor","Calven","Istral","Pareth","Dovren"),
("Darsen","Velora","Keldin","Orvek","Sarnel"),
("Feron","Talven","Merith","Sovran","Belvik"),
("Larev","Nerith","Calder","Veyron","Tormek"),
("Torven","Elsar","Marvek","Dorin","Kaleth"),
("Selnor","Kareth","Valen","Ordan","Mervek"),
("Mirev","Taldor","Neris","Kelmar","Sorvik"),
("Varen","Solith","Deran","Malvek","Cordan"),
("Kelor","Ardin","Velmar","Toren","Narell"),
("Narev","Belith","Corven","Sareth","Dorvik"),
("Dervan","Mirel","Talvek","Orsen","Kelron"),
("Calnor","Verith","Naldor","Seren","Parvek"),
("Parel","Dorven","Kelith","Maros","Tervik"),
("Sorven","Tarell","Vindor","Nelmar","Calrek"),
("Barel","Corith","Laven","Derik","Solmar"),
("Ralen","Mervor","Talith","Kesven","Noreth"),
("Norel","Valdor","Serith","Calvenor","Darvek"),
("Tervan","Orel","Mardin","Velos","Karven"),
("Karev","Solen","Dareth","Mirven","Talrek")
]
flat=[x for row in BASE for x in row]
if len(flat)!=len(set(flat)):raise RuntimeError("Panelde tekrar eden yapay isim var.")
W=[]
for i,(s,current,near,former,role_target) in enumerate(BASE):
    cross=BASE[(i+1)%len(BASE)]
    cs,cc=cross[0],cross[1]
    facts=[
        ("CURRENT",f"The current capital of {s} is {current}."),
        ("NEAR",f"The largest city of {s} is {near}."),
        ("FORMER",f"The former capital of {s} was {former}."),
        ("ROLE",f"The current capital of {role_target} is {s}."),
        ("CROSS",f"The current capital of {cs} is {cc}.")
    ]
    queries=[
        ("CURRENT_DIRECT",f"What is the current capital of {s}?",current),
        ("CURRENT_PARA",f"Which city presently serves as the administrative seat of {s}?",current),
        ("CURRENT_TR",f"{s} devletinin günümüzdeki yönetim merkezi hangi şehirdir?",current),
        ("FORMER_DIRECT",f"What was the former capital of {s}?",former),
        ("FORMER_PARA",f"Which city previously served as the capital of {s}?",former),
        ("FORMER_TR",f"{s} devletinin eski başkenti hangi şehirdi?",former),
        ("NEAR_DIRECT",f"What is the largest city of {s}?",near),
        ("NEAR_PARA",f"Which city is the largest urban center in {s}?",near),
        ("NEAR_TR",f"{s} devletinin en büyük şehri hangisidir?",near),
        ("ROLE_DIRECT",f"What is the current capital of {role_target}?",s),
        ("ROLE_PARA",f"Which city presently serves as the administrative seat of {role_target}?",s),
        ("ROLE_TR",f"{role_target} devletinin günümüzdeki yönetim merkezi hangi şehirdir?",s)
    ]
    W.append({"id":i,"subject":s,"facts":facts,"queries":queries})

LOCK={
"test":TEST,"parent545":PARENT545,"model":MODEL_ID,"seed":SEED,"worlds":len(W),
"facts_per_world":5,"queries_per_world":12,
"fact_types":["CURRENT","NEAR","FORMER","ROLE","CROSS"],
"query_types":["CURRENT_DIRECT","CURRENT_PARA","CURRENT_TR","FORMER_DIRECT","FORMER_PARA","FORMER_TR","NEAR_DIRECT","NEAR_PARA","NEAR_TR","ROLE_DIRECT","ROLE_PARA","ROLE_TR"],
"context_orders":["FORWARD","REVERSE"],
"free_generation":True,"retrieval":False,"cartridge":False,"router":False,"scorer":False,"threshold":False,"training":False,"lora":False,"gold_in_prompt":False,
"purpose":"BEHAVIORAL_UPPER_BOUND_FOR_NATIVE_RELATIONAL_BINDING"
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      LOCK SHA:",LOCK_SHA)

SYSTEM="Answer the question using only the facts in the provided context. Give only the requested name and nothing else."
def make_prompt(facts,q,reverse=False):
    fs=list(reversed(facts)) if reverse else list(facts)
    context="\n".join(x[1] for x in fs)
    msgs=[{"role":"system","content":SYSTEM},{"role":"user","content":f"CONTEXT:\n{context}\n\nQUESTION:\n{q}\n\nANSWER:"}]
    return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
@torch.no_grad()
def generate(prompt):
    x=tok(prompt,return_tensors="pt",add_special_tokens=False).to(DEVICE)
    n=x.input_ids.shape[1]
    y=model.generate(**x,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    out=tok.decode(y[0,n:],skip_special_tokens=True).strip()
    del x,y
    return out
def norm(s):
    s=s.casefold().strip()
    s=re.sub(r"[^\w\s-]"," ",s,flags=re.UNICODE)
    return " ".join(s.split())
def exact_name(out,gold):
    # Behavioral scoring only after generation; gold never enters model prompt.
    return norm(out)==norm(gold)
def contains_name(out,name):
    return re.search(r"(?<!\w)"+re.escape(norm(name))+r"(?!\w)",norm(out)) is not None

print("\n[2/8] Panel mühürü...")
print(f"      Worlds={len(W)} | facts/world=5 | queries/world=12 | orders=2 | total generations={len(W)*12*2}")
print("      Gold cevaplar prompt içine konmuyor.")
print("      Panel response görülmeden sabit.")
print("      FORWARD ve REVERSE aynı olgular, yalnız sıra ters.")

print("\n[3/8] FORWARD context davranışı...")
RESULTS=[]
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/{len(W)} {w['subject']}")
    for qt,q,gold in w["queries"]:
        out=generate(make_prompt(w["facts"],q,False))
        RESULTS.append({"world":wi,"subject":w["subject"],"order":"FORWARD","type":qt,"q":q,"gold":gold,"out":out,"ok":exact_name(out,gold)})

print("\n[4/8] REVERSE context davranışı...")
for wi,w in enumerate(W):
    print(f"      {wi+1:02d}/{len(W)} {w['subject']}")
    for qt,q,gold in w["queries"]:
        out=generate(make_prompt(w["facts"],q,True))
        RESULTS.append({"world":wi,"subject":w["subject"],"order":"REVERSE","type":qt,"q":q,"gold":gold,"out":out,"ok":exact_name(out,gold)})

print("\n[5/8] Önceden tanımlı davranış tablosu...")
QT=[x[0] for x in W[0]["queries"]]
def subset(order=None,typ=None):
    return [r for r in RESULTS if (order is None or r["order"]==order) and (typ is None or r["type"]==typ)]
def acc(rows):return sum(r["ok"] for r in rows)/len(rows) if rows else float("nan")
for order in ["FORWARD","REVERSE"]:
    print(f"\n      {order}")
    for qt in QT:
        rr=subset(order,qt);print(f"      {qt:16s} {sum(r['ok'] for r in rr):02d}/{len(rr):02d} = {acc(rr):.4f}")
    rr=subset(order);print(f"      {'ALL':16s} {sum(r['ok'] for r in rr):03d}/{len(rr):03d} = {acc(rr):.4f}")

print("\n[6/8] İlişki / dil / sıra dayanıklılığı...")
GROUPS={
"CURRENT":["CURRENT_DIRECT","CURRENT_PARA","CURRENT_TR"],
"FORMER":["FORMER_DIRECT","FORMER_PARA","FORMER_TR"],
"NEAR":["NEAR_DIRECT","NEAR_PARA","NEAR_TR"],
"ROLE":["ROLE_DIRECT","ROLE_PARA","ROLE_TR"],
"DIRECT":["CURRENT_DIRECT","FORMER_DIRECT","NEAR_DIRECT","ROLE_DIRECT"],
"PARA":["CURRENT_PARA","FORMER_PARA","NEAR_PARA","ROLE_PARA"],
"TURKISH":["CURRENT_TR","FORMER_TR","NEAR_TR","ROLE_TR"]
}
SUMMARY={}
for g,types in GROUPS.items():
    rr=[r for r in RESULTS if r["type"] in types]
    SUMMARY[g]=acc(rr);print(f"      {g:8s}: {sum(r['ok'] for r in rr):03d}/{len(rr):03d} = {SUMMARY[g]:.4f}")
F={ (r["world"],r["type"]):r for r in RESULTS if r["order"]=="FORWARD"}
R={ (r["world"],r["type"]):r for r in RESULTS if r["order"]=="REVERSE"}
pairs=list(F)
both=sum(F[k]["ok"] and R[k]["ok"] for k in pairs)
same_out=sum(norm(F[k]["out"])==norm(R[k]["out"]) for k in pairs)
flip_correct=sum(F[k]["ok"]!=R[k]["ok"] for k in pairs)
print(f"      BOTH ORDERS CORRECT : {both}/{len(pairs)} = {both/len(pairs):.4f}")
print(f"      SAME OUTPUT         : {same_out}/{len(pairs)} = {same_out/len(pairs):.4f}")
print(f"      CORRECTNESS FLIP    : {flip_correct}/{len(pairs)} = {flip_correct/len(pairs):.4f}")

print("\n[7/8] Hataların çeldirici analizi...")
# Post-hoc only: classify whether a wrong free-generation output copied another fact value.
ERR=[r for r in RESULTS if not r["ok"]]
DIST={"NEAR_VALUE":0,"FORMER_VALUE":0,"ROLE_SUBJECT":0,"CROSS_VALUE":0,"OTHER":0}
EX=[]
for r in ERR:
    w=W[r["world"]];vals={k:v for k,v in w["facts"]}
    s=w["subject"];near=re.search(r"is ([^.]+)",vals["NEAR"]).group(1);former=re.search(r"was ([^.]+)",vals["FORMER"]).group(1)
    role_answer=s;cross=re.search(r"is ([^.]+)",vals["CROSS"]).group(1)
    hit=None
    for label,val in [("NEAR_VALUE",near),("FORMER_VALUE",former),("ROLE_SUBJECT",role_answer),("CROSS_VALUE",cross)]:
        if contains_name(r["out"],val):hit=label;break
    DIST[hit or "OTHER"]+=1
    if len(EX)<24:EX.append((r["world"],r["subject"],r["order"],r["type"],r["gold"],r["out"],hit or "OTHER"))
print(f"      Errors: {len(ERR)}/{len(RESULTS)}")
for k,v in DIST.items():print(f"      {k:14s}: {v}")
if EX:
    print("\n      İlk hata örnekleri:")
    for x in EX:print(f"      W{x[0]:02d} {x[1]:8s} {x[2]:7s} {x[3]:16s} gold={x[4]:10s} out={x[5]!r} class={x[6]}")

print("\n[8/8] Sentinel / mühür...")
S1=sentinel();OK=S0==S1
ALLACC=acc(RESULTS);FACC=acc(subset("FORWARD"));RACC=acc(subset("REVERSE"))
FINAL={
"test":TEST,"parent545":PARENT545,"lock_sha":LOCK_SHA,"weight_sentinel":OK,
"all_accuracy":ALLACC,"forward_accuracy":FACC,"reverse_accuracy":RACC,
"groups":SUMMARY,"both_orders_correct":both/len(pairs),"same_output":same_out/len(pairs),"correctness_flip":flip_correct/len(pairs),
"errors":len(ERR),"distractors":DIST,
"retrieval":False,"cartridge":False,"router":False,"scorer":False,"threshold":False,"training":False
}
RESULT_SHA=hashlib.sha256(json.dumps(FINAL,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Weight sentinel             :","PASS" if OK else "FAIL")
print("      Trainable tensors           :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Retrieval / cartridge       : YOK")
print("      Router / scorer / threshold : YOK")
print("      Gold in prompt              : YOK")
print("      Training/LoRA/optimizer     : YOK")
print("\n"+"="*160)
print("TEST546A SONUÇ — FULL-CONTEXT UPPER BOUND")
print("="*160)
print("MODEL                         :",MODEL_ID)
print("WORLDS                        :",len(W))
print("TOTAL GENERATIONS             :",len(RESULTS))
print("FORWARD ACC                   :",f"{FACC:.6f}")
print("REVERSE ACC                   :",f"{RACC:.6f}")
print("ALL ACC                       :",f"{ALLACC:.6f}")
print("CURRENT                       :",f"{SUMMARY['CURRENT']:.6f}")
print("FORMER                        :",f"{SUMMARY['FORMER']:.6f}")
print("NEAR                          :",f"{SUMMARY['NEAR']:.6f}")
print("ROLE                          :",f"{SUMMARY['ROLE']:.6f}")
print("DIRECT                        :",f"{SUMMARY['DIRECT']:.6f}")
print("PARAPHRASE                    :",f"{SUMMARY['PARA']:.6f}")
print("TURKISH                       :",f"{SUMMARY['TURKISH']:.6f}")
print("BOTH ORDERS CORRECT           :",f"{both}/{len(pairs)}")
print("SAME OUTPUT ACROSS ORDER      :",f"{same_out}/{len(pairs)}")
print("CORRECTNESS FLIPS             :",f"{flip_correct}/{len(pairs)}")
print("WEIGHT SENTINEL               :","PASS" if OK else "FAIL")
print("TEST545 RESULT SHA            :",PARENT545)
print("TEST546A ÖN KİLİT SHA         :",LOCK_SHA)
print("TEST546A RESULT SHA           :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                         : TEST546A_FULL_CONTEXT_UPPER_BOUND_TAMAMLANDI")
print("="*160)
