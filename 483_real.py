# TEST483 — AKBASCORE NIRVANA · BLIND REAL-WORLD 18-BANK BREAK TEST · QWEN K120/V128/OWN
import os,sys,re,json,time,random,hashlib,gc,subprocess,importlib.util
from collections import Counter
for m,p in [("torch","torch"),("transformers","transformers"),("datasets","datasets"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
TEST="483";SEED=483;N=18;GEN_MODEL="mistralai/Mistral-7B-Instruct-v0.3";MODEL_ID="Qwen/Qwen2.5-7B-Instruct";K_DIM=120;V_DIM=128;MAX_NEW=32
FMT="QUESTION:\n{q}\n\n ANSWER:";SEP="\n\n";DEVICE=torch.device("cuda")
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required")
os.environ["TOKENIZERS_PARALLELISM"]="false";random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_grad_enabled(False)
def norm(s):return re.sub(r"[^\w'-]+"," ",str(s).casefold()).strip()
def firstline(s):return next((x.strip() for x in str(s).splitlines() if x.strip()),"")
def isnone(s):
    n=norm(firstline(s));return n=="none" or n.startswith("none ")
def words(s):return re.findall(r"[\w'-]+",norm(s))
def tokdiff(gt,out):
    g=words(gt);o=words(firstline(out));cg=Counter(g);co=Counter(o);miss=list((cg-co).elements());extra=list((co-cg).elements());inter=sum((cg&co).values());p=inter/max(1,len(o));r=inter/max(1,len(g));f=0.0 if p+r==0 else 2*p*r/(p+r)
    return g,o,miss,extra,f
def answer_hit(raw,gt):
    if isnone(raw):return False
    a=norm(gt);b=norm(firstline(raw));return bool(a and b and (a==b or a in b))
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def dtype_kw():
    v=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
    return "dtype" if v>=(4,56) else "torch_dtype"
DT=dtype_kw()
print("="*150);print("TEST483 — BLIND REAL-WORLD 18-BANK BREAK TEST");print("="*150)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/8] BLIND WIKIPEDIA PANEL")
ds=load_dataset("wikimedia/wikipedia","20231101.en",split="train",streaming=True)
rng=random.Random(SEED);pool=[];seen=0
for e in ds:
    t=re.sub(r"\s+"," ",str(e.get("text",""))).strip()
    if 500<=len(t)<=2200:
        seen+=1
        if len(pool)<96:pool.append(t)
        else:
            j=rng.randrange(seen)
            if j<96:pool[j]=t
    if seen>=3000:break
if len(pool)<N:raise RuntimeError(f"Not enough Wikipedia passages: {len(pool)}")
rng.shuffle(pool);candidate_sources=pool[:N];SOURCE_POOL_SHA=sha(candidate_sources)
print("Wikipedia candidates locked:",len(candidate_sources),"| SOURCE_POOL_SHA:",SOURCE_POOL_SHA)
print("[2/8] MISTRAL FIRST-SHOT QUESTION/ANSWER GENERATION — NO RETRY")
gtok=AutoTokenizer.from_pretrained(GEN_MODEL,use_fast=True)
gmodel=AutoModelForCausalLM.from_pretrained(GEN_MODEL,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in gmodel.parameters():p.requires_grad_(False)
GEN_SYS="""You create one factual memory test from a supplied record. Return exactly three lines:
QUESTION: <one natural English question answerable only from the record>
ANSWER: <short exact answer, preferably 1-6 words>
EVIDENCE: <one exact sentence or short exact span from the record that proves the answer>
Do not use outside knowledge. Do not ask yes/no questions. Do not explain."""
@torch.inference_mode()
def gen_case(src):
    msgs=[{"role":"system","content":GEN_SYS},{"role":"user","content":"RECORD:\n"+src}]
    z=gtok.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,return_tensors="pt",return_dict=True)
    if not hasattr(z,"keys") or "input_ids" not in z:raise RuntimeError("Mistral chat template did not return input_ids")
    z={k:v.to(DEVICE) for k,v in z.items() if torch.is_tensor(v)}
    ids=z["input_ids"];o=gmodel.generate(**z,max_new_tokens=128,do_sample=False,pad_token_id=gtok.eos_token_id)
    raw=gtok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
    q=re.search(r"(?im)^QUESTION:\s*(.+)$",raw);a=re.search(r"(?im)^ANSWER:\s*(.+)$",raw);e=re.search(r"(?im)^EVIDENCE:\s*(.+)$",raw)
    return raw,(q.group(1).strip() if q else ""),(a.group(1).strip() if a else ""),(e.group(1).strip() if e else "")
PANEL=[]
for i,src in enumerate(candidate_sources):
    raw,q,a,e=gen_case(src);valid=bool(q and a and e and norm(a) in norm(src) and norm(e) in norm(src))
    PANEL.append(dict(i=i+1,source=src,question=q,answer=a,evidence=e,generator_raw=raw,generator_valid=valid));print(f" GEN {i+1:02d}/{N} valid={valid}")
PANEL_SHA=sha(PANEL);print("PANEL_SHA:",PANEL_SHA);print("GENERATOR_VALID:",sum(x["generator_valid"] for x in PANEL),"/",N)
del gmodel,gtok;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
print("[3/8] QWEN LOAD — TEST482 ENGINE CONSTANTS")
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None)or H//NH;KVD=NKV*HD;NL=len(layers)
if (NL,H,NH,NKV,HD)!=(28,3584,28,4,128):raise RuntimeError(f"Qwen architecture mismatch: {(NL,H,NH,NKV,HD)}")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_ge=model.generation_config.eos_token_id
EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple))else[_ge])}-{None})
if not EOS:raise RuntimeError("No EOS token")
enc=lambda s:tok(s,add_special_tokens=False).input_ids
FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[13].mlp.down_proj.weight,layers[23].self_attn.o_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            z=j*(n-256)//15;h.update(a[z:z+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
SENT0=sentinel()
@torch.inference_mode()
def kv_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],dtype=torch.long,device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True);K=[];V=[]
    for L in range(NL):
        z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn;K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
    return K,V
def forge(s):return kv_from_ids([PAD]+enc(s+SEP))
@torch.inference_mode()
def install(K,V):
    T=K[0].shape[0];pos=torch.arange(T,dtype=torch.long,device=DEVICE)[None];cos,sin=model.model.rotary_emb(K[0][None],pos);KK=[];VV=[]
    for L in range(NL):
        k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
        KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
    return tuple(KK),tuple(VV),T
CB=[]
@torch.inference_mode()
def build_codebook():
    global CB
    CO=[forge(s) for s in CORPUS];CB=[]
    for L in range(NL):
        e={}
        for j,n in enumerate(("K","V")):
            R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];BB=[]
            for h in range(NKV):
                X=R[:,h];mu=X.mean(0);_,_,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(128,Vh.shape[0]);b=Vh[:m].T.contiguous()
                if m<128:b=torch.nn.functional.pad(b,(0,128-m))
                MU.append(mu);BB.append(b)
            e[n]=(torch.stack(MU),torch.stack(BB))
        CB.append(e)
    del CO;gc.collect();torch.cuda.empty_cache()
@torch.inference_mode()
def packet_from_kv(K,V):
    out={}
    for n,X,d in (("K",K,K_DIM),("V",V,V_DIM)):
        rows=[]
        for L in range(NL):
            mu,B=CB[L][n];coeff=torch.einsum("thi,hid->thd",X[L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
            content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16);rows.append(torch.cat([X[L][:1],content]))
        out[n]=rows
    return out["K"],out["V"]
def packet(source):
    K,V=forge(source);return packet_from_kv(K,V)
@torch.inference_mode()
def batch(q,kvs):
    if not kvs:raise RuntimeError("Empty cartridge list")
    qids=enc(FMT.format(q=q))
    if not qids:raise RuntimeError("Empty question tokenization")
    BN=len(kvs);Tm=max(kv[2] for kv in kvs);nq=len(qids)
    try:cache=DynamicCache(config=cfg)
    except TypeError:cache=DynamicCache()
    for L in range(NL):
        Ks=[];Vs=[]
        for kv in kvs:
            k,v,p=kv[0][L],kv[1][L],Tm-kv[2]
            if p:
                k=torch.cat([k.new_zeros(1,NKV,p,HD),k],2);v=torch.cat([v.new_zeros(1,NKV,p,HD),v],2)
            Ks.append(k);Vs.append(v)
        cache.update(torch.cat(Ks,0).clone(),torch.cat(Vs,0).clone(),L)
    mask=torch.zeros(BN,Tm+nq,dtype=torch.long,device=DEVICE)
    for b,kv in enumerate(kvs):mask[b,Tm-kv[2]:]=1
    pos=torch.tensor([kv[2] for kv in kvs],dtype=torch.long,device=DEVICE)[:,None]+torch.arange(nq,dtype=torch.long,device=DEVICE)[None]
    ids=torch.tensor([qids]*BN,dtype=torch.long,device=DEVICE);outs=[[] for _ in range(BN)];done=[False]*BN
    for _ in range(MAX_NEW):
        o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True);nxt=o.logits[:,-1].float().argmax(-1).tolist()
        for b,t in enumerate(nxt):
            if not done[b]:
                if t in EOS:done[b]=True
                else:outs[b].append(t)
        if all(done):break
        feed=[EOS[0] if done[b] else nxt[b] for b in range(BN)];ids=torch.tensor([[t] for t in feed],dtype=torch.long,device=DEVICE);pos=pos[:,-1:]+1;mask=torch.cat([mask,torch.ones(BN,1,dtype=torch.long,device=DEVICE)],1)
    return [tok.decode(x,skip_special_tokens=True).strip() for x in outs]
@torch.inference_mode()
def direct_source(src,q):
    prompt="RECORD:\n"+src+"\n\nQUESTION:\n"+q+"\n\nAnswer only from the record. Give only the short answer on the first line."
    ids=torch.tensor([enc(prompt)],dtype=torch.long,device=DEVICE);mask=torch.ones_like(ids)
    o=model.generate(input_ids=ids,attention_mask=mask,max_new_tokens=MAX_NEW,do_sample=False,pad_token_id=PAD,eos_token_id=EOS)
    return tok.decode(o[0,ids.shape[1]:],skip_special_tokens=True).strip()
print("[4/8] FIXED TEST482 PCA CODEBOOK")
t=time.perf_counter();build_codebook();torch.cuda.synchronize();print("CODEBOOK:",round(time.perf_counter()-t,2),"s")
print("[5/8] FORGE 18 RAW + K120/V128 CARTRIDGES")
RAW=[];COMP=[];t=time.perf_counter()
for i,x in enumerate(PANEL):
    K,V=forge(x["source"]);RAW.append(install(K,V));Kc,Vc=packet_from_kv(K,V);COMP.append(install(Kc,Vc));del K,V,Kc,Vc
    print(f" FORGE {i+1:02d}/{N}")
torch.cuda.synchronize();print("FORGE:",round(time.perf_counter()-t,2),"s")
if len(RAW)!=N or len(COMP)!=N:raise RuntimeError("Cartridge bank size mismatch")
if not all(kv[2]>1 for kv in RAW+COMP):raise RuntimeError("Invalid cartridge length")
READ_SUFFIX=" Answer only the answer if it is stated in this record; otherwise answer NONE."
RESULT=[];print("[6/8] SOURCE / RAWKV / ORACLE / 18-BANK")
for i,x in enumerate(PANEL):
    gt=x["answer"];q=x["question"]+READ_SUFFIX
    src=direct_source(x["source"],x["question"]);raw=batch(q,[RAW[i]])[0];oracle=batch(q,[COMP[i]])[0];rows=batch(q,COMP)
    if len(rows)!=N:raise RuntimeError(f"Bank output mismatch at case {i+1}")
    bank=rows[i];off_non_none=[j for j,r in enumerate(rows) if j!=i and not isnone(r)]
    srcok=answer_hit(src,gt);rawok=answer_hit(raw,gt);orok=answer_hit(oracle,gt);bankok=answer_hit(bank,gt);cleanoff=len(off_non_none)==0;linked=bankok and cleanoff
    g,o,miss,extra,f1=tokdiff(gt,bank)
    if not x["generator_valid"]:cls="PANEL_INVALID"
    elif srcok and rawok and orok and linked:cls="CLEAN_PASS"
    elif srcok and rawok and not orok:cls="COMPRESSION_OR_COMPRESSED_READOUT_FAILURE"
    elif srcok and not rawok:cls="RAWKV_OR_READOUT_FAILURE"
    elif srcok and orok and not linked:cls="BANK_RETRIEVAL_OR_ABSTENTION_FAILURE"
    elif not srcok and linked:cls="MEMORY_PASS_BASE_FAIL"
    else:cls="BASE_MODEL_OR_MIXED_CONFOUND"
    RESULT.append(dict(i=i+1,generator_valid=x["generator_valid"],source_ok=srcok,rawkv_ok=rawok,oracle_ok=orok,bank_ok=bankok,offtarget_clean=cleanoff,linked=linked,offtarget_non_none=len(off_non_none),class_=cls,source_raw=src,rawkv_raw=raw,oracle_raw=oracle,bank_raw=bank,bank_rows=rows,gt_tokens=g,out_tokens=o,missing=miss,extra=extra,word_f1=f1))
    print(f" CASE {i+1:02d}/{N} SRC={int(srcok)} RAW={int(rawok)} ORACLE={int(orok)} BANK={int(bankok)} OFF={len(off_non_none)} CLASS={cls}")
print("[7/8] FULL HUMAN-READABLE EVIDENCE")
for i,(x,r) in enumerate(zip(PANEL,RESULT),1):
    print("\n"+"-"*150);print(f"[{i:02d}] CLASS={r['class_']} | GENERATOR_VALID={r['generator_valid']} | SOURCE={r['source_ok']} RAWKV={r['rawkv_ok']} ORACLE={r['oracle_ok']} BANK={r['bank_ok']} OFFTARGET_NON_NONE={r['offtarget_non_none']}")
    print("SOURCE:",x["source"]);print("QUESTION:",x["question"]);print("GROUND_TRUTH:",x["answer"]);print("EVIDENCE:",x["evidence"]);print("MISTRAL_RAW:",x["generator_raw"])
    print("QWEN_SOURCE_RAW:",r["source_raw"]);print("RAWKV_ORACLE_RAW:",r["rawkv_raw"]);print("K120V128_ORACLE_RAW:",r["oracle_raw"]);print("K120V128_BANK_RAW:",r["bank_raw"])
    print("GT_WORDS:",r["gt_tokens"]);print("OUT_WORDS:",r["out_tokens"]);print("MISSING:",r["missing"]);print("EXTRA:",r["extra"]);print("WORD_F1:",f"{r['word_f1']:.4f}")
    if r["offtarget_non_none"]:print("OFFTARGET_NON_NONE_ROWS:",[(j+1,r["bank_rows"][j]) for j in range(N) if j!=i-1 and not isnone(r["bank_rows"][j])])
print("\n[8/8] FINAL")
valid=sum(x["generator_valid"] for x in PANEL);src=sum(r["source_ok"] for r in RESULT);raw=sum(r["rawkv_ok"] for r in RESULT);oracle=sum(r["oracle_ok"] for r in RESULT);bank=sum(r["bank_ok"] for r in RESULT);linked=sum(r["linked"] for r in RESULT);off=sum(r["offtarget_non_none"] for r in RESULT)
classes=Counter(r["class_"] for r in RESULT);meanf=sum(r["word_f1"] for r in RESULT)/N;SENT1=sentinel()
FINAL=dict(test=TEST,seed=SEED,n=N,generator=GEN_MODEL,test_model=MODEL_ID,engine="TEST482 K120/V128/OWN",source_pool_sha=SOURCE_POOL_SHA,panel_sha=PANEL_SHA,generator_valid=f"{valid}/{N}",source_qwen=f"{src}/{N}",rawkv_oracle=f"{raw}/{N}",compressed_oracle=f"{oracle}/{N}",bank_target=f"{bank}/{N}",strict_linked=f"{linked}/{N}",offtarget_non_none=f"{off}/{N*(N-1)}",mean_word_f1=meanf,classes=dict(classes),weight_sentinel_unchanged=SENT1==SENT0)
print(json.dumps(FINAL,indent=2,ensure_ascii=False))
print("SOURCE_POOL_SHA:",SOURCE_POOL_SHA);print("PANEL_SHA:",PANEL_SHA);print("WEIGHT_SENTINEL_START:",SENT0);print("WEIGHT_SENTINEL_END  :",SENT1);print("WEIGHTS_UNCHANGED:",SENT1==SENT0)
if valid<N:verdict="PANEL_GENERATION_CONFOUND"
elif linked==N and off==0:verdict="PASS_TEST483_BLIND_REALWORLD_18_BANK"
elif src<N:verdict="MIXED_FAIL_WITH_BASE_MODEL_CONFOUND"
else:verdict="FAIL_TEST483_NIRVANA_REALWORLD"
print("VERDICT:",verdict);print("="*150)
