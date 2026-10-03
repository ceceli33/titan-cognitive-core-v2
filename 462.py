# TEST462 — IBR BATCH EQUIVALENCE X-RAY
# PURPOSE: localize TEST460B's 20/32 full-transcript agreement despite 32/32 semantic agreement.
# NO architecture change. NO tuning. NO F3/F4. Development/technical audit only.
#
# SAME ENGINE AS TEST460B:
# Qwen/Qwen2.5-7B-Instruct | BF16 | SDPA | frozen
# K120 / V128 / OWN | same fixed neutral codebook | same forge/packet/install
#
# SAME installed single-module PKV is decoded through:
#   SEQ_MANUAL  : canonical one-row manual greedy
#   BATCH_B1    : exact batch decoder with batch size 1
#   BATCH_MULTI : exact same batch decoder with multiple module rows
#
# Comparison levels:
#   TOKEN1      : first generated token id
#   FIRST       : normalized first answer/sentence used by the experiment
#   SEMANTIC    : parser-level answer identity
#   FULL        : complete decoded transcript
#
# PRECOMMITTED INTERPRETATION:
# A) B1 FULL=100%, MULTI TOKEN1/FIRST/SEM=100%, only MULTI FULL differs:
#    BATCH_TAIL_ONLY_DIVERGENCE -> IBR behaviorally validated; freeze and proceed to TEST461.
# B) B1 FULL=100%, MULTI TOKEN1<100% or FIRST/SEM<100%:
#    MULTIROW_BATCH_NUMERICAL_DIVERGENCE -> batch composition changes answer path; repair before TEST461.
# C) B1 FULL<100%:
#    BATCH_CORE_SEMANTICS_MISMATCH -> batch decoder itself differs even at B=1; repair before TEST461.
# D) technical errors/non-frozen:
#    TECHNICAL_INVALID.
import os,re,json,time,random,hashlib
from collections import Counter
SEED=460;N_ENT=4;MAX_NEW=32
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain."
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.",
"The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.",
"A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.",
"The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.",
"The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.",
"An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.",
"The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.",
"The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.",
"Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.",
"A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.",
"Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.",
"Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.",
"Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.",
"Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),
     "F2":("Container {T} holds the {O}.","The {P} houses container {T}.")}
MW1="Which container holds the {obj}? If this record does not say which container holds the {obj}, answer NONE."+STYLE
MW2="Where is container {cid} located according to this record? Give the place name. If this record does not say where container {cid} is, answer NONE."+STYLE
P_ADJ=["cinder","dove","ermine","flint","heather","iris","lemon","nutmeg","onyx","rosewood"]
P_NOUN=["abbey","bakery","citadel","dairy","forum","grotto","hostel","kennel","rampart","wharf"]
O_ADJ=["dusty","enameled","frayed","gilt","mottled","varnished"]
O_NOUN=["anvil","basin","chime","dagger","easel","flask"]
ID_RE=re.compile(r"\b([A-Z]{2}-\d{3})\b",re.I)
AMBIG={"not","no","never","nor","or","either","maybe","perhaps","possibly","probably","might","unclear","but","unsure"}

def norm(x):return re.sub(r"[^\w]+"," ",str(x).casefold()).strip()
def first(text):
    line=next((s.strip() for s in str(text).splitlines() if s.strip()),"")
    line=re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",line,flags=re.I).strip(" *`\"'")
    return re.split(r"(?<=[.!?])\s+",line)[0] if line else ""
def classify(text):
    fl=first(text);n=norm(fl);w=n.split();ids=sorted({m.upper() for m in ID_RE.findall(fl)})
    if not w:return "empty",ids,fl
    if set(w)<={"none","unknown"}:return "none",ids,fl
    if (set(w)&AMBIG) or len(ids)>1 or (set(w)&{"none","unknown"}):return "ambiguous",ids,fl
    return "answer",ids,fl
def place_key(a):return re.sub(r"^(?:the|in the|at the|in|at)\s+","",norm(a)).strip()
def semantic_key(text):
    c,ids,fl=classify(text)
    if c=="none":return ("none",)
    if c=="answer" and len(ids)==1:return ("id",ids[0])
    if c=="answer":return ("answer",place_key(fl))
    return (c,tuple(ids),place_key(fl))
def first_key(text):return norm(first(text))
def make_entities():
    rng=random.Random(SEED);places=[f"{a} {n}" for a in P_ADJ for n in P_NOUN];objs=[f"{a} {n}" for a in O_ADJ for n in O_NOUN]
    rng.shuffle(places);rng.shuffle(objs);ids=set();out=[]
    def nid():
        while True:
            x="".join(rng.choice("BCDFGHJKLMNPRSTVWXZ") for _ in range(2))+"-"+str(rng.randint(100,999))
            if x not in ids:ids.add(x);return x
    for i in range(N_ENT):
        P=places.pop(0);Q=places.pop(next(j for j,q in enumerate(places) if not set(q.split())&set(P.split())))
        T=nid();D=nid();D2=nid();fam="F1" if i%2==0 else "F2";A,L=FAM[fam]
        mods={"A":A.format(O=objs[i],T=T),"B":L.format(T=T,P=P),"X":L.format(T=D,P=Q),"W":L.format(T=D2,P=P)}
        out.append(dict(e=i,obj=objs[i],P=P,Q=Q,T=T,D=D,D2=D2,fam=fam,mods=mods))
    return out

ENTS=make_entities();PROBES=[]
for e in ENTS:
    for role in ("A","B","X","W"):
        PROBES.append(dict(e=e["e"],fam=e["fam"],role=role,kind="S1",q=MW1.format(obj=e["obj"]),text=e["mods"][role]))
        PROBES.append(dict(e=e["e"],fam=e["fam"],role=role,kind="S2",q=MW2.format(cid=e["T"]),text=e["mods"][role]))
assert len(PROBES)==32
print(f"[TEST462] CPU self-test PASS: {len(PROBES)} probes; F1/F2 only; F3/F4 untouched")

def run_gpu():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    assert torch.cuda.is_available(),"A100 GPU runtime required"
    torch.manual_seed(SEED);DEV=torch.device("cuda");torch.set_grad_enabled(False)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    try:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=torch.bfloat16).eval()
    except TypeError:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",torch_dtype=torch.bfloat16).eval()
    for p in model.parameters():p.requires_grad_(False)
    cfg=model.config;layers=model.model.layers;NL=len(layers);NKV=cfg.num_key_value_heads
    HD=getattr(cfg,"head_dim",None) or cfg.hidden_size//cfg.num_attention_heads;KVD=NKV*HD
    PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple)) else [ge])}-{None})
    enc=lambda s:tok(s,add_special_tokens=False).input_ids
    def sentinel():
        h=hashlib.sha256()
        for t in (layers[0].self_attn.q_proj.weight,layers[NL//2].mlp.down_proj.weight,model.lm_head.weight):
            h.update(t.detach().reshape(-1)[:4096].float().cpu().numpy().tobytes())
        return h.hexdigest()
    SENT0=sentinel()
    @torch.inference_mode()
    def kv_from_ids(ids):
        o=model(input_ids=torch.tensor([ids],device=DEV),output_hidden_states=True,use_cache=False,return_dict=True);K=[];V=[]
        for L in range(NL):
            z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn
            K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
        return K,V
    def forge(s):return kv_from_ids([PAD]+enc(s+SEP))
    @torch.inference_mode()
    def install(K,V):
        T=K[0].shape[0];cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEV)[None]);KK=[];VV=[]
        for L in range(NL):
            k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
            KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
        return tuple(KK),tuple(VV),T
    def mkcache(kv):
        try:c=DynamicCache(config=cfg)
        except TypeError:c=DynamicCache()
        for L in range(NL):c.update(kv[0][L].clone(),kv[1][L].clone(),L)
        return c

    print("[TEST462] building TEST460/460B fixed neutral codebook ...")
    with torch.inference_mode():
        CO=[forge(s) for s in CORPUS];CB=[]
        for L in range(NL):
            e={}
            for j,n in enumerate(("K","V")):
                R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];B=[]
                for h in range(NKV):
                    X=R[:,h];mu=X.mean(0);_,_,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(128,Vh.shape[0]);bb=Vh[:m].T.contiguous()
                    if m<128:bb=torch.nn.functional.pad(bb,(0,128-m))
                    MU.append(mu);B.append(bb)
                e[n]=(torch.stack(MU),torch.stack(B))
            CB.append(e)
        del CO
    @torch.inference_mode()
    def packet(source):
        K,V=forge(source);out={}
        for n,X,d in (("K",K,120),("V",V,128)):
            rows=[]
            for L in range(NL):
                mu,B=CB[L][n];coeff=torch.einsum("thi,hid->thd",X[L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
                content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)
                rows.append(torch.cat([X[L][:1],content]))
            out[n]=rows
        return out["K"],out["V"]
    BANK={}
    def getkv(text):
        if text not in BANK:BANK[text]=install(*packet(text))
        return BANK[text]

    @torch.inference_mode()
    def seq_manual(q,kv):
        qids=enc(FMT.format(q=q));T=kv[2];cache=mkcache(kv);ids=torch.tensor([qids],device=DEV)
        mask=torch.ones(1,T+len(qids),dtype=torch.long,device=DEV);pos=torch.arange(T,T+len(qids),device=DEV)[None]
        out=[];tokens=[]
        for _ in range(MAX_NEW):
            o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
            t=int(o.logits[0,-1].float().argmax());tokens.append(t)
            if t in EOS:break
            out.append(t);ids=torch.tensor([[t]],device=DEV);pos=pos[:,-1:]+1;mask=torch.cat([mask,torch.ones(1,1,dtype=torch.long,device=DEV)],1)
        return tok.decode(out,skip_special_tokens=True).strip(),tokens

    @torch.inference_mode()
    def batch_manual(q,kvs):
        qids=enc(FMT.format(q=q));BN=len(kvs);Tm=max(kv[2] for kv in kvs);nq=len(qids)
        try:cache=DynamicCache(config=cfg)
        except TypeError:cache=DynamicCache()
        for L in range(NL):
            Ks=[];Vs=[]
            for kv in kvs:
                k,v,p=kv[0][L],kv[1][L],Tm-kv[2]
                if p:
                    zk=k.new_zeros(1,NKV,p,HD);zv=v.new_zeros(1,NKV,p,HD);k=torch.cat([zk,k],2);v=torch.cat([zv,v],2)
                Ks.append(k);Vs.append(v)
            cache.update(torch.cat(Ks).clone(),torch.cat(Vs).clone(),L)
        mask=torch.zeros(BN,Tm+nq,dtype=torch.long,device=DEV)
        for b,kv in enumerate(kvs):mask[b,Tm-kv[2]:]=1
        pos=torch.tensor([kv[2] for kv in kvs],device=DEV)[:,None]+torch.arange(nq,device=DEV)[None]
        ids=torch.tensor([qids]*BN,device=DEV);outs=[[] for _ in range(BN)];tokens=[[] for _ in range(BN)];done=[False]*BN
        for _ in range(MAX_NEW):
            o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
            nxt=o.logits[:,-1].float().argmax(-1).tolist()
            for b,t in enumerate(nxt):
                if not done[b]:
                    tokens[b].append(t)
                    if t in EOS:done[b]=True
                    else:outs[b].append(t)
            if all(done):break
            feed=[EOS[0] if done[b] and EOS else nxt[b] for b in range(BN)]
            ids=torch.tensor([[t] for t in feed],device=DEV);pos=pos[:,-1:]+1;mask=torch.cat([mask,torch.ones(BN,1,dtype=torch.long,device=DEV)],1)
        return [(tok.decode(outs[b],skip_special_tokens=True).strip(),tokens[b]) for b in range(BN)]

    t0=time.time();errors=[];rows=[]
    # Compute sequential and B=1 independently for every probe.
    SINGLE={}
    for i,p in enumerate(PROBES):
        try:
            kv=getkv(p["text"]);s,st=seq_manual(p["q"],kv);b1=batch_manual(p["q"],[kv])[0]
            SINGLE[i]=(s,st,b1[0],b1[1])
        except Exception as ex:errors.append(dict(probe=i,stage="single",error=f"{type(ex).__name__}: {ex}"))

    # True multi-row batches contain only identical questions, exactly as TEST460B.
    MULTI={}
    groups={}
    for i,p in enumerate(PROBES):groups.setdefault(p["q"],[]).append((i,p))
    for q,g in groups.items():
        try:
            z=batch_manual(q,[getkv(p["text"]) for _,p in g])
            for (i,_),v in zip(g,z):MULTI[i]=v
        except Exception as ex:errors.append(dict(stage="multi",question=q,error=f"{type(ex).__name__}: {ex}"))

    for i,p in enumerate(PROBES):
        if i not in SINGLE or i not in MULTI:continue
        s,st,b1,b1t=SINGLE[i];bm,bmt=MULTI[i]
        def cmp(a,at,b,bt):
            return dict(token1=bool(at and bt and at[0]==bt[0]),first=first_key(a)==first_key(b),
                        semantic=semantic_key(a)==semantic_key(b),full=a==b)
        c1=cmp(s,st,b1,b1t);cm=cmp(s,st,bm,bmt);c1m=cmp(b1,b1t,bm,bmt)
        row=dict(e=p["e"],fam=p["fam"],role=p["role"],kind=p["kind"],seq=s,b1=b1,multi=bm,
                 seq_token1=st[0] if st else None,b1_token1=b1t[0] if b1t else None,multi_token1=bmt[0] if bmt else None,
                 seq_vs_b1=c1,seq_vs_multi=cm,b1_vs_multi=c1m,
                 seq_sem=semantic_key(s),b1_sem=semantic_key(b1),multi_sem=semantic_key(bm))
        rows.append(row)
        print(f"[{i+1:02d}/{len(PROBES)}] e{p['e']} {p['fam']} {p['role']}/{p['kind']} "
              f"S=B1 T/F/S/X={int(c1['token1'])}/{int(c1['first'])}/{int(c1['semantic'])}/{int(c1['full'])} "
              f"S=BM T/F/S/X={int(cm['token1'])}/{int(cm['first'])}/{int(cm['semantic'])}/{int(cm['full'])} | "
              f"S={first(s)[:24]!r} B1={first(b1)[:24]!r} BM={first(bm)[:24]!r}",flush=True)

    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    n=len(rows)
    def score(path,key):return sum(r[path][key] for r in rows)
    summary={"n":n}
    for path in ("seq_vs_b1","seq_vs_multi","b1_vs_multi"):
        summary[path]={k:f"{score(path,k)}/{n}" for k in ("token1","first","semantic","full")}
    summary["by_family"]={}
    for fam in ("F1","F2"):
        rr=[r for r in rows if r["fam"]==fam];nf=len(rr);summary["by_family"][fam]={}
        for path in ("seq_vs_b1","seq_vs_multi"):
            summary["by_family"][fam][path]={k:f"{sum(r[path][k] for r in rr)}/{nf}" for k in ("token1","first","semantic","full")}
    if errors or n!=len(PROBES) or not frozen:verdict="TECHNICAL_INVALID"
    elif score("seq_vs_b1","full")<n:verdict="BATCH_CORE_SEMANTICS_MISMATCH"
    elif score("seq_vs_multi","token1")<n or score("seq_vs_multi","first")<n or score("seq_vs_multi","semantic")<n:verdict="MULTIROW_BATCH_NUMERICAL_DIVERGENCE"
    elif score("seq_vs_multi","full")<n:verdict="BATCH_TAIL_ONLY_DIVERGENCE"
    else:verdict="FULL_BATCH_EQUIVALENCE"
    summary["verdict"]=verdict;summary["errors"]=len(errors);summary["frozen_weights_ok"]=frozen;summary["minutes"]=round((time.time()-t0)/60,1)
    path="/content/test462_results.json" if os.path.isdir("/content") else "test462_results.json"
    json.dump(dict(summary=summary,rows=rows,errors=errors),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*110)
    for k,v in summary.items():print(f"{k}: {v}")
    print("="*110)
    print(f"TEST462 VERDICT: {verdict}  (results: {path})")
    if verdict in ("BATCH_TAIL_ONLY_DIVERGENCE","FULL_BATCH_EQUIVALENCE"):
        print("NEXT: freeze IBR batch implementation; TEST461 remains untouched and may now be run as final F1/F2 + unseen F3/F4 held-out validation.")
    elif verdict=="MULTIROW_BATCH_NUMERICAL_DIVERGENCE":
        print("NEXT: do not consume TEST461; multi-row batching changes answer-path behavior and must be repaired.")
    elif verdict=="BATCH_CORE_SEMANTICS_MISMATCH":
        print("NEXT: do not consume TEST461; B=1 batch decoder already differs from sequential manual decoding.")
    else:
        print("NEXT: technical repair only; no scientific interpretation.")

if os.environ.get("T462_CPU_ONLY")!="1":run_gpu()
