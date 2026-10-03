# TEST464 — FINAL F2 STAGE-2 FAILURE X-RAY
# LAST DIAGNOSTIC TEST BEFORE DEMO
#
# TEST461 IS SEALED AND REMAINS: FAIL_FINAL_HELDOUT.
# THIS TEST DOES NOT RESCORE, RETUNE OR REDEFINE TEST461.
#
# QUESTION:
# Why did TEST461 F2 linked Stage-2 fail while Stage-1 recovered the correct container?
#
# SAME FROZEN ENGINE:
# Qwen/Qwen2.5-7B-Instruct | BF16 | SDPA | frozen
# K120 / V128 / OWN | same 32-sentence neutral codebook
# same independent packet forge | same TEST462-validated IBR batch decoder
#
# F2 ONLY — same deterministic TEST461 entities (SEED=461):
#   B_SOLO      : true F2 B packet alone
#   B_PLUS_1    : true B + 1 unrelated module
#   B_PLUS_7    : true B + 7 unrelated modules
#   B_PLUS_15   : true B + 15 unrelated modules
#
# For every condition record:
#   - raw output from EVERY batch row
#   - true-B row output
#   - parsed place from true-B row
#   - all parsed places
#   - deterministic aggregate
#
# PRECOMMITTED DIAGNOSIS:
# 1) If true B is wrong/UNKNOWN already in B_SOLO:
#       F2_SINGLE_MODULE_READOUT_FAILURE
# 2) If B_SOLO is correct but true-B row itself changes/fails in larger batch:
#       F2_MULTIROW_READOUT_INTERFERENCE
# 3) If true-B row stays correct but aggregate becomes UNKNOWN:
#       F2_AGGREGATION_COLLISION
# 4) If all tested F2 B rows stay correct and aggregate correct:
#       F2_FAILURE_NOT_REPRODUCED
#
# NO intervention follows from this test. It closes the experimental chain.
import os,re,json,time,random,hashlib
from collections import Counter,defaultdict
SEED=461;MAX_NEW=32
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain."
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={
"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),
"F2":("Container {T} holds the {O}.","The {P} houses container {T}."),
"F3":("Inside container {T} rests the {O}.","Container {T} can be found at the {P}."),
"F4":("The {O} is kept within container {T}.","At the {P}, container {T} is stored.")
}
MW2="Where is container {cid} located according to this record? Give the place name. If this record does not say where container {cid} is, answer NONE."+STYLE
P_ADJ=["amber","birch","cobalt","drift","elm","frost","ginger","hazel","indigo","juniper","khaki","lilac","marble","navy","ochre","pearl","quartz","russet","silver","teal","umber","violet","willow","xanthic"]
P_NOUN=["arcade","bastion","chapel","depot","estate","foundry","granary","harbor","inn","junction","lodge","mill","nursery","observatory","pier","quarry","refinery","station","tower","warehouse","yard","zoo"]
O_ADJ=["aged","brass","carved","dented","etched","faded","glazed","hinged","ivory","lacquered","painted","polished","riveted","stained","woven","yellowed"]
O_NOUN=["barometer","casket","drum","figurine","goblet","harp","inkwell","lantern","medallion","plaque","reel","satchel","tablet","urn","vase","whistle"]
ID_RE=re.compile(r"\b([A-Z]{2}-\d{3})\b",re.I)
AMBIG={"not","no","never","nor","or","either","maybe","perhaps","possibly","probably","might","unclear","but","unsure"}
SIZES=(1,2,8,16)

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
def parse_place(text,known):
    c,ids,fl=classify(text)
    if c!="answer" or ids:return None
    k=place_key(fl);hits=[p for p in known if place_key(p)==k or place_key(p) in k]
    return hits[0] if len(hits)==1 else None
def make_entities():
    rng=random.Random(SEED)
    places=[f"{a} {n}" for a in P_ADJ for n in P_NOUN];objs=[f"{a} {n}" for a in O_ADJ for n in O_NOUN]
    rng.shuffle(places);rng.shuffle(objs);used=set();out=[]
    def nid():
        while True:
            x="".join(rng.choice("BCDFGHJKLMNPRSTVWXZ") for _ in range(2))+"-"+str(rng.randint(100,999))
            if x not in used:used.add(x);return x
    for i in range(16):
        P=places.pop(0);Q=places.pop(next(j for j,q in enumerate(places) if not set(q.split())&set(P.split())))
        T=nid();D=nid();fam=("F1","F2","F3","F4")[i%4];A,B=FAM[fam]
        mods={"A":A.format(O=objs[i],T=T),"B":B.format(T=T,P=P),"X":B.format(T=D,P=Q)}
        out.append(dict(e=i,fam=fam,obj=objs[i],P=P,Q=Q,T=T,D=D,mods=mods))
    return out

ENTS=make_entities();F2=[e for e in ENTS if e["fam"]=="F2"]
assert [e["e"] for e in F2]==[1,5,9,13]
print("[TEST464] CPU lock PASS")
print(" TEST461 deterministic F2 entities:",[(e["e"],e["T"],e["P"]) for e in F2])
print(" conditions: B_SOLO / B+1 / B+7 / B+15")
print(" TEST461 remains sealed; diagnostic only.")

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

    print("[TEST464] building frozen TEST461 neutral codebook ...")
    with torch.inference_mode():
        CO=[forge(s) for s in CORPUS];CB=[]
        for L in range(NL):
            d={}
            for j,n in enumerate(("K","V")):
                R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];B=[]
                for h in range(NKV):
                    X=R[:,h];mu=X.mean(0);_,_,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(128,Vh.shape[0]);bb=Vh[:m].T.contiguous()
                    if m<128:bb=torch.nn.functional.pad(bb,(0,128-m))
                    MU.append(mu);B.append(bb)
                d[n]=(torch.stack(MU),torch.stack(B))
            CB.append(d)
        del CO

    @torch.inference_mode()
    def packet(source):
        K,V=forge(source);out={}
        for n,X,d in (("K",K,120),("V",V,128)):
            rows=[]
            for L in range(NL):
                mu,B=CB[L][n]
                coeff=torch.einsum("thi,hid->thd",X[L][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
                content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)
                rows.append(torch.cat([X[L][:1],content]))
            out[n]=rows
        return out["K"],out["V"]

    BANK={}
    def getkv(text):
        if text not in BANK:BANK[text]=install(*packet(text))
        return BANK[text]

    @torch.inference_mode()
    def batch(q,kvs):
        qids=enc(FMT.format(q=q));BN=len(kvs);Tm=max(kv[2] for kv in kvs);nq=len(qids)
        try:cache=DynamicCache(config=cfg)
        except TypeError:cache=DynamicCache()
        for L in range(NL):
            Ks=[];Vs=[]
            for kv in kvs:
                k,v,p=kv[0][L],kv[1][L],Tm-kv[2]
                if p:
                    k=torch.cat([k.new_zeros(1,NKV,p,HD),k],2)
                    v=torch.cat([v.new_zeros(1,NKV,p,HD),v],2)
                Ks.append(k);Vs.append(v)
            cache.update(torch.cat(Ks).clone(),torch.cat(Vs).clone(),L)
        mask=torch.zeros(BN,Tm+nq,dtype=torch.long,device=DEV)
        for b,kv in enumerate(kvs):mask[b,Tm-kv[2]:]=1
        pos=torch.tensor([kv[2] for kv in kvs],device=DEV)[:,None]+torch.arange(nq,device=DEV)[None]
        ids=torch.tensor([qids]*BN,device=DEV);outs=[[] for _ in range(BN)];done=[False]*BN
        for _ in range(MAX_NEW):
            o=model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,return_dict=True)
            nxt=o.logits[:,-1].float().argmax(-1).tolist()
            for b,t in enumerate(nxt):
                if not done[b]:
                    if t in EOS:done[b]=True
                    else:outs[b].append(t)
            if all(done):break
            feed=[EOS[0] if done[b] and EOS else nxt[b] for b in range(BN)]
            ids=torch.tensor([[t] for t in feed],device=DEV);pos=pos[:,-1:]+1
            mask=torch.cat([mask,torch.ones(BN,1,dtype=torch.long,device=DEV)],1)
        return [tok.decode(x,skip_special_tokens=True).strip() for x in outs]

    print("[TEST464] forging same deterministic TEST461 module bank ...")
    MODULES=[]
    for e in ENTS:
        for role in ("A","B","X"):
            MODULES.append(dict(owner=e["e"],fam=e["fam"],role=role,text=e["mods"][role],kv=getkv(e["mods"][role])))
    KNOWN=[e["P"] for e in ENTS]+[e["Q"] for e in ENTS]
    rng=random.Random(SEED+991)

    # Freeze one nested distractor ordering per F2 entity:
    # size 2 = B + first distractor
    # size 8 = same B + same first 7
    # size16 = same B + same first 15
    ORDERS={}
    for e in F2:
        trueB=next(m for m in MODULES if m["owner"]==e["e"] and m["role"]=="B")
        pool=[m for m in MODULES if m is not trueB]
        rng.shuffle(pool);ORDERS[e["e"]]=[trueB]+pool[:15]

    def aggregate(texts):
        parsed=[parse_place(x,KNOWN) for x in texts]
        vals=[x for x in parsed if x is not None];uniq=sorted(set(vals))
        return (uniq[0] if len(uniq)==1 else None),parsed,uniq

    rows=[];errors=[];t0=time.time()
    print("[TEST464] FINAL DIAGNOSTIC START")
    for e in F2:
        q=MW2.format(cid=e["T"]);ordered=ORDERS[e["e"]]
        print("-"*110)
        print(f"e{e['e']:02d} F2 | target={e['T']} -> {e['P']} | B={e['mods']['B']}")
        for n in SIZES:
            try:
                mods=ordered[:n];outs=batch(q,[m["kv"] for m in mods]);agg,parsed,uniq=aggregate(outs)
                true_raw=outs[0];true_parsed=parsed[0]
                true_ok=true_parsed==e["P"];agg_ok=agg==e["P"]
                foreign=[p for p in parsed[1:] if p is not None]
                row=dict(e=e["e"],fam="F2",size=n,target_id=e["T"],target_place=e["P"],
                         true_B_raw=true_raw,true_B_first=first(true_raw),true_B_parsed=true_parsed,
                         true_B_ok=int(true_ok),aggregate=agg,aggregate_ok=int(agg_ok),
                         parsed_places=parsed,unique_places=uniq,foreign_parsed=foreign,
                         module_rows=[dict(owner=m["owner"],fam=m["fam"],role=m["role"],text=m["text"],
                                           raw=outs[i],first=first(outs[i]),parsed=parsed[i]) for i,m in enumerate(mods)])
                rows.append(row)
                print(f"[B+{n-1:02d}] TRUE={int(true_ok)} AGG={int(agg_ok)} "
                      f"true='{first(true_raw)[:40]}' parsed={true_parsed or 'NONE'} "
                      f"foreign={foreign} aggregate={agg or 'UNKNOWN'}")
                for i,(m,o,p) in enumerate(zip(mods,outs,parsed)):
                    print(f"   r{i:02d} owner=e{m['owner']:02d} {m['fam']} {m['role']} | "
                          f"{first(o)[:52]!r} -> {p or 'NONE'}")
            except Exception as ex:
                errors.append(dict(e=e["e"],size=n,error=f"{type(ex).__name__}: {ex}"))
                print(f"[B+{n-1:02d}] ERROR {type(ex).__name__}: {ex}")

    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    solo=[r for r in rows if r["size"]==1]
    larger=[r for r in rows if r["size"]>1]
    solo_true=sum(r["true_B_ok"] for r in solo)
    larger_true=sum(r["true_B_ok"] for r in larger)
    larger_agg=sum(r["aggregate_ok"] for r in larger)
    changed_true=[]
    for e in F2:
        rr=[r for r in rows if r["e"]==e["e"]]
        s=next((r for r in rr if r["size"]==1),None)
        if s:
            for r in rr:
                if r["size"]>1 and r["true_B_parsed"]!=s["true_B_parsed"]:
                    changed_true.append((e["e"],r["size"]))
    collision=[r for r in larger if r["true_B_ok"] and not r["aggregate_ok"]]

    if errors or len(rows)!=len(F2)*len(SIZES) or not frozen:
        verdict="TECHNICAL_INVALID"
    elif solo_true<len(F2):
        verdict="F2_SINGLE_MODULE_READOUT_FAILURE"
    elif changed_true or larger_true<len(larger):
        verdict="F2_MULTIROW_READOUT_INTERFERENCE"
    elif collision:
        verdict="F2_AGGREGATION_COLLISION"
    else:
        verdict="F2_FAILURE_NOT_REPRODUCED"

    summary={
        "test":464,
        "purpose":"final F2 Stage-2 diagnostic; TEST461 remains sealed",
        "F2_entities":[e["e"] for e in F2],
        "sizes":list(SIZES),
        "solo_true_B":f"{solo_true}/{len(solo)}",
        "larger_true_B":f"{larger_true}/{len(larger)}",
        "larger_aggregate":f"{larger_agg}/{len(larger)}",
        "true_B_changed_vs_solo":changed_true,
        "aggregation_collisions":len(collision),
        "errors":len(errors),
        "frozen_weights_ok":frozen,
        "minutes":round((time.time()-t0)/60,1),
        "verdict":verdict
    }
    path="/content/test464_results.json" if os.path.isdir("/content") else "test464_results.json"
    json.dump(dict(summary=summary,rows=rows,errors=errors),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*110)
    print("TEST464 — FINAL F2 STAGE-2 X-RAY")
    for k,v in summary.items():print(f"{k}: {v}")
    print("="*110)
    print("TEST464 VERDICT:",verdict)
    print("results:",path)
    print("EXPERIMENTAL CHAIN CLOSED — preserve TEST461 and TEST464 exactly as observed; no retuning.")

if os.environ.get("T464_CPU_ONLY")!="1":run_gpu()
