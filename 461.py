# TEST460B — IBR IMPLEMENTATION EQUIVALENCE AUDIT
# PURPOSE ONLY: explain TEST460 batch_stage1_agreement=96/160 while SEQ_MW=BMW_K0=BMW_K6=64/64.
# NO new memory architecture, NO F3/F4, NO parameter tuning.
# Same Qwen/K120-V128/OWN engine and fixed neutral codebook as TEST460.
#
# Three decode paths over THE SAME installed single-module PKV:
#   LEGACY_GEN   = TEST452/460 reader(): generate([PAD]*cache_len + qids, past_key_values=cache)
#   SEQ_MANUAL   = one module, manual greedy, qids only + cache, explicit local position_ids
#   BATCH_MANUAL = same manual greedy semantics, modules batched with left padding/mask
#
# PRIMARY TEST:
#   SEQ_MANUAL vs BATCH_MANUAL exact transcript agreement.
# SECONDARY:
#   LEGACY_GEN vs SEQ_MANUAL, to determine whether TEST460's 60% agreement came from
#   generate/dummy-prefix semantics rather than batching.
#
# PRECOMMITTED VERDICT:
#   MANUAL_BATCH_EQUIVALENT:
#       SEQ_MANUAL==BATCH_MANUAL >=99% exact transcripts AND parsed decisions 100%.
#       => IBR implementation validated; TEST460 BATCH_PATH_MISMATCH was a protocol mismatch.
#   BATCH_SEMANTICS_MISMATCH:
#       manual exact <99% or parsed decisions differ.
#       => fix batch mask/position/cache semantics before TEST461.
#   LEGACY_PROTOCOL_DIVERGENCE:
#       manual=batch passes, but legacy differs materially. This is EXPECTED/diagnostic and
#       does NOT invalidate IBR.
#
# Development audit: 4 TEST460-style entities, F1/F2 only. F3/F4 untouched.
import os,re,json,time,random,hashlib
from collections import Counter
SEED=460
N_ENT=4
MAX_NEW=32
FMT="QUESTION:\n{q}\n\nANSWER:"
SEP="\n\n"
STYLE=" Give only the answer on the first line; do not explain."
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
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

def norm(x): return re.sub(r"[^\w]+"," ",str(x).casefold()).strip()
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
def place_key(a): return re.sub(r"^(?:the|in the|at the|in|at)\s+","",norm(a)).strip()
def semantic_key(text):
    c,ids,fl=classify(text)
    if c=="none":return ("none",)
    if c=="answer" and len(ids)==1:return ("id",ids[0])
    if c=="answer":return ("answer",place_key(fl))
    return (c,tuple(ids),place_key(fl))
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
ENTS=make_entities()
PROBES=[]
for e in ENTS:
    # All four module roles are audited. Questions intentionally include both positive and negative reads.
    for role in ("A","B","X","W"):
        PROBES.append(dict(e=e["e"],fam=e["fam"],role=role,text=e["mods"][role],kind="S1",q=MW1.format(obj=e["obj"])))
        PROBES.append(dict(e=e["e"],fam=e["fam"],role=role,text=e["mods"][role],kind="S2",q=MW2.format(cid=e["T"])))
assert len(PROBES)==N_ENT*8
print(f"[TEST460B] CPU self-test PASS: {len(PROBES)} module-question probes ({N_ENT} entities, F1/F2 only)")

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
    def mkcache(kv,batch=1):
        try:c=DynamicCache(config=cfg)
        except TypeError:c=DynamicCache()
        for L in range(NL):
            k=kv[0][L].clone();v=kv[1][L].clone()
            if batch>1:k=k.expand(batch,-1,-1,-1).contiguous();v=v.expand(batch,-1,-1,-1).contiguous()
            c.update(k,v,L)
        return c
    print("[TEST460B] building TEST460 fixed neutral codebook ...")
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

    # Exact TEST460/TEST452 legacy sequential protocol.
    @torch.inference_mode()
    def legacy_generate(q,kv):
        qids=enc(FMT.format(q=q));pre=[PAD]*kv[2];cache=mkcache(kv)
        ids=torch.tensor([pre+qids],device=DEV)
        y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,max_new_tokens=MAX_NEW,do_sample=False,
                         repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
        return tok.decode(y[0,ids.shape[1]:].tolist(),skip_special_tokens=True).strip()

    # Canonical manual protocol: cache already represents prefix, therefore only qids are submitted.
    @torch.inference_mode()
    def seq_manual(q,kv):
        qids=enc(FMT.format(q=q));nq=len(qids);T=kv[2];cache=mkcache(kv)
        ids=torch.tensor([qids],device=DEV);mask=torch.ones(1,T+nq,dtype=torch.long,device=DEV)
        pos=torch.arange(T,T+nq,device=DEV)[None];out=[]
        for _ in range(MAX_NEW):
            o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
            t=int(o.logits[0,-1].float().argmax())
            if t in EOS:break
            out.append(t);ids=torch.tensor([[t]],device=DEV);pos=pos[:,-1:]+1
            mask=torch.cat([mask,torch.ones(1,1,dtype=torch.long,device=DEV)],1)
        return tok.decode(out,skip_special_tokens=True).strip()

    # TEST460 batching semantics, but used here on groups sharing one question.
    @torch.inference_mode()
    def batch_manual(q,kvs):
        qids=enc(FMT.format(q=q));B=len(kvs);Tm=max(kv[2] for kv in kvs);nq=len(qids)
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
        mask=torch.zeros(B,Tm+nq,dtype=torch.long,device=DEV)
        for b,kv in enumerate(kvs):mask[b,Tm-kv[2]:]=1
        # local RoPE positions: question starts immediately after each module's own logical length.
        pos=torch.tensor([kv[2] for kv in kvs],device=DEV)[:,None]+torch.arange(nq,device=DEV)[None]
        ids=torch.tensor([qids]*B,device=DEV);outs=[[] for _ in range(B)];done=[False]*B
        for _ in range(MAX_NEW):
            o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
            nxt=o.logits[:,-1].float().argmax(-1).tolist()
            for b,t in enumerate(nxt):
                if not done[b]:
                    if t in EOS:done[b]=True
                    else:outs[b].append(t)
            if all(done):break
            # Finished rows continue with EOS only as inert batch companions; their decoded output is frozen.
            feed=[EOS[0] if done[b] and EOS else nxt[b] for b in range(B)]
            ids=torch.tensor([[t] for t in feed],device=DEV);pos=pos[:,-1:]+1
            mask=torch.cat([mask,torch.ones(B,1,dtype=torch.long,device=DEV)],1)
        return [tok.decode(x,skip_special_tokens=True).strip() for x in outs]

    t0=time.time();rows=[];errors=[]
    # Batch only probes with identical q. This avoids changing question tokenization merely to create a batch.
    groups={}
    for i,p in enumerate(PROBES):groups.setdefault(p["q"],[]).append((i,p))
    BATCH_OUT={}
    for q,g in groups.items():
        try:
            outs=batch_manual(q,[getkv(p["text"]) for _,p in g])
            for (i,_),o in zip(g,outs):BATCH_OUT[i]=o
        except Exception as ex:
            errors.append(dict(group=q,error=f"{type(ex).__name__}: {ex}"))
    for i,p in enumerate(PROBES):
        try:
            kv=getkv(p["text"]);lg=legacy_generate(p["q"],kv);sm=seq_manual(p["q"],kv);bm=BATCH_OUT[i]
            row=dict(e=p["e"],fam=p["fam"],role=p["role"],kind=p["kind"],legacy=lg,seq_manual=sm,batch_manual=bm,
                     legacy_seq_exact=lg==sm,seq_batch_exact=sm==bm,
                     legacy_seq_sem=semantic_key(lg)==semantic_key(sm),seq_batch_sem=semantic_key(sm)==semantic_key(bm),
                     legacy_key=semantic_key(lg),seq_key=semantic_key(sm),batch_key=semantic_key(bm))
            rows.append(row)
            print(f"[{i+1:02d}/{len(PROBES)}] e{p['e']} {p['fam']} {p['role']}/{p['kind']} "
                  f"L=S:{int(row['legacy_seq_exact'])} S=B:{int(row['seq_batch_exact'])} "
                  f"semLS:{int(row['legacy_seq_sem'])} semSB:{int(row['seq_batch_sem'])} | "
                  f"L={first(lg)[:22]!r} S={first(sm)[:22]!r} B={first(bm)[:22]!r}",flush=True)
        except Exception as ex:
            errors.append(dict(probe=i,error=f"{type(ex).__name__}: {ex}"))
    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    n=len(rows);xSB=sum(r["seq_batch_exact"] for r in rows);sSB=sum(r["seq_batch_sem"] for r in rows)
    xLS=sum(r["legacy_seq_exact"] for r in rows);sLS=sum(r["legacy_seq_sem"] for r in rows)
    byfam={}
    for f in ("F1","F2"):
        rr=[r for r in rows if r["fam"]==f]
        byfam[f]=dict(n=len(rr),seq_batch_exact=sum(r["seq_batch_exact"] for r in rr),
                      seq_batch_sem=sum(r["seq_batch_sem"] for r in rr),legacy_seq_exact=sum(r["legacy_seq_exact"] for r in rr),
                      legacy_seq_sem=sum(r["legacy_seq_sem"] for r in rr))
    if errors or n!=len(PROBES) or not frozen:verdict="TECHNICAL_INVALID"
    elif xSB/n>=.99 and sSB==n:
        verdict="MANUAL_BATCH_EQUIVALENT"
        legacy_note="LEGACY_PROTOCOL_DIVERGENCE" if xLS/n<.95 else "LEGACY_ALSO_EQUIVALENT"
    else:
        verdict="BATCH_SEMANTICS_MISMATCH";legacy_note="UNRESOLVED"
    S=dict(n=n,seq_batch_exact=f"{xSB}/{n}",seq_batch_semantic=f"{sSB}/{n}",
           legacy_seq_exact=f"{xLS}/{n}",legacy_seq_semantic=f"{sLS}/{n}",
           by_family=byfam,verdict=verdict,legacy_note=legacy_note,errors=len(errors),
           frozen_weights_ok=frozen,minutes=round((time.time()-t0)/60,1))
    path="/content/test460b_results.json" if os.path.isdir("/content") else "test460b_results.json"
    json.dump(dict(summary=S,rows=rows,errors=errors),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*100)
    for k,v in S.items():print(f"{k}: {v}")
    print("="*100)
    print(f"TEST460B VERDICT: {verdict} | {legacy_note}  (results: {path})")
    if verdict=="MANUAL_BATCH_EQUIVALENT":
        print("NEXT: freeze IBR manual-batch implementation and run untouched TEST461 with new entities + unseen F3/F4 + module-count scaling.")
    elif verdict=="BATCH_SEMANTICS_MISMATCH":
        print("NEXT: DO NOT consume TEST461. Audit left-padding attention mask / local position_ids / cache-position semantics first.")
    else:
        print("NEXT: technical repair only; do not interpret scientifically.")

if os.environ.get("T460B_CPU_ONLY")!="1":run_gpu()
