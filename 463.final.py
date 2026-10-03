# TEST461 — NIRVANA IBR FINAL HELD-OUT VALIDATION
# DEMO-PRECEDING FROZEN TEST
#
# ENGINE FROZEN FROM TEST460/462:
# Qwen/Qwen2.5-7B-Instruct | BF16 | SDPA | frozen weights
# K120 / V128 / OWN | fixed 32-sentence neutral PCA codebook
# independent compressed PKV modules | isolated batched read (IBR)
# NO source text at readout | NO training | NO tuning | NO concat rescue
#
# FINAL HELD-OUT PANEL:
#   16 NEW entities
#   F1/F2 known structural families
#   F3/F4 FIRST-USE unseen surface families
#   linked + unlinked/UNKNOWN
#   memory scale: 2 / 8 / 16 modules
#   NOMEM control
#
# FROZEN FAMILIES — DO NOT CHANGE AFTER THIS RUN:
# F1 A: "The {O} sits in container {T}."
#    B: "Container {T} is located at the {P}."
# F2 A: "Container {T} holds the {O}."
#    B: "The {P} houses container {T}."
# F3 A: "Inside container {T} rests the {O}."
#    B: "Container {T} can be found at the {P}."
# F4 A: "The {O} is kept within container {T}."
#    B: "At the {P}, container {T} is stored."
#
# PRECOMMITTED PASS — NO POST-HOC CHANGES:
#   overall linked >= .80
#   overall unlinked/UNKNOWN >= .85
#   false-link <= .10
#   each family linked >= .70
#   16-module linked drop vs 2-module <= .10
#   16-module UNKNOWN drop vs 2-module <= .10
#   NOMEM strict UNKNOWN success = 0
#
# IMPORTANT:
# 2/8/16 refers to number of independently compressed module rows exposed to IBR.
# Routing/index is metadata only; it stores module membership, never answer content.
# Ambiguous multiple IDs or multiple places => UNKNOWN.
import os,re,json,time,random,hashlib,math
from collections import Counter,defaultdict
SEED=461;N_ENT=16;MAX_NEW=32
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain."
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={
"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),
"F2":("Container {T} holds the {O}.","The {P} houses container {T}."),
"F3":("Inside container {T} rests the {O}.","Container {T} can be found at the {P}."),
"F4":("The {O} is kept within container {T}.","At the {P}, container {T} is stored.")
}
MW1="Which container holds the {obj}? If this record does not say which container holds the {obj}, answer NONE."+STYLE
MW2="Where is container {cid} located according to this record? Give the place name. If this record does not say where container {cid} is, answer NONE."+STYLE
P_ADJ=["amber","birch","cobalt","drift","elm","frost","ginger","hazel","indigo","juniper","khaki","lilac","marble","navy","ochre","pearl","quartz","russet","silver","teal","umber","violet","willow","xanthic"]
P_NOUN=["arcade","bastion","chapel","depot","estate","foundry","granary","harbor","inn","junction","lodge","mill","nursery","observatory","pier","quarry","refinery","station","tower","warehouse","yard","zoo"]
O_ADJ=["aged","brass","carved","dented","etched","faded","glazed","hinged","ivory","lacquered","painted","polished","riveted","stained","woven","yellowed"]
O_NOUN=["barometer","casket","drum","figurine","goblet","harp","inkwell","lantern","medallion","plaque","reel","satchel","tablet","urn","vase","whistle"]
ID_RE=re.compile(r"\b([A-Z]{2}-\d{3})\b",re.I)
AMBIG={"not","no","never","nor","or","either","maybe","perhaps","possibly","probably","might","unclear","but","unsure"}
SCALES=(2,8,16)

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
def parse_id(text):
    c,ids,_=classify(text)
    return ids[0] if c=="answer" and len(ids)==1 else None
def parse_place(text,known):
    c,ids,fl=classify(text)
    if c!="answer" or ids:return None
    k=place_key(fl);hits=[p for p in known if place_key(p)==k or place_key(p) in k]
    return hits[0] if len(hits)==1 else None
def wilson(k,n,z=1.96):
    if not n:return 0.0
    p=k/n;d=1+z*z/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/d
def make_entities():
    rng=random.Random(SEED)
    places=[f"{a} {n}" for a in P_ADJ for n in P_NOUN];objs=[f"{a} {n}" for a in O_ADJ for n in O_NOUN]
    rng.shuffle(places);rng.shuffle(objs);used=set();out=[]
    def nid():
        while True:
            x="".join(rng.choice("BCDFGHJKLMNPRSTVWXZ") for _ in range(2))+"-"+str(rng.randint(100,999))
            if x not in used:used.add(x);return x
    for i in range(N_ENT):
        P=places.pop(0);Q=places.pop(next(j for j,q in enumerate(places) if not set(q.split())&set(P.split())))
        T=nid();D=nid();fam=("F1","F2","F3","F4")[i%4];A,B=FAM[fam]
        mods={"A":A.format(O=objs[i],T=T),"B":B.format(T=T,P=P),"X":B.format(T=D,P=Q)}
        out.append(dict(e=i,fam=fam,obj=objs[i],P=P,Q=Q,T=T,D=D,mods=mods))
    return out
ENTS=make_entities()
assert len({e["obj"] for e in ENTS})==N_ENT and len({e["P"] for e in ENTS})==N_ENT and len({e["T"] for e in ENTS})==N_ENT
print("[TEST461] CPU panel lock")
print(" entities:",N_ENT,"families:",Counter(e["fam"] for e in ENTS),"scales:",SCALES)
print(" F3/F4 FIRST USE — frozen before GPU results")
print(" PASS: L>=.80 U>=.85 false-link<=.10 each-family-L>=.70 scale-drop<=.10 NOMEM-U=0")

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
    print("[TEST461] building frozen neutral codebook ...")
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
    def batch(q,kvs):
        qids=enc(FMT.format(q=q));BN=len(kvs);Tm=max(kv[2] for kv in kvs);nq=len(qids)
        try:cache=DynamicCache(config=cfg)
        except TypeError:cache=DynamicCache()
        for L in range(NL):
            Ks=[];Vs=[]
            for kv in kvs:
                k,v,p=kv[0][L],kv[1][L],Tm-kv[2]
                if p:
                    k=torch.cat([k.new_zeros(1,NKV,p,HD),k],2);v=torch.cat([v.new_zeros(1,NKV,p,HD),v],2)
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
    @torch.inference_mode()
    def nomem_batch(q,n):
        # Real PAD-only cache, one slot, no source-derived information.
        K,V=kv_from_ids([PAD]);kv=install(K,V)
        return batch(q,[kv]*n)

    # Build independently compressed module bank ONCE. Source strings are used only here, then readout uses PKV.
    print("[TEST461] forging independent held-out modules ...")
    MODULES=[]
    for e in ENTS:
        for role in ("A","B","X"):
            text=e["mods"][role];MODULES.append(dict(owner=e["e"],role=role,text=text,kv=getkv(text)))
    KNOWN_PLACES=[e["P"] for e in ENTS]+[e["Q"] for e in ENTS]
    rng=random.Random(SEED+991)

    # Frozen metadata routing:
    # object -> its A module candidate; ID -> all B/X modules potentially containing that ID.
    # Metadata contains module addresses only, never target place answers.
    A_BY_E={e["e"]:next(m for m in MODULES if m["owner"]==e["e"] and m["role"]=="A") for e in ENTS}
    REL=[m for m in MODULES if m["role"] in ("B","X")]

    def fill(base,n,exclude_owner=None):
        out=list(base);pool=[m for m in MODULES if m not in out and (exclude_owner is None or m["owner"]!=exclude_owner)]
        rng.shuffle(pool)
        for m in pool:
            if len(out)>=n:break
            out.append(m)
        return out[:n]

    def decide_id(texts):
        ids=[parse_id(x) for x in texts];ids=[x for x in ids if x]
        u=sorted(set(ids))
        return u[0] if len(u)==1 else None
    def decide_place(texts):
        ps=[parse_place(x,KNOWN_PLACES) for x in texts];ps=[x for x in ps if x]
        u=sorted(set(ps))
        return u[0] if len(u)==1 else None

    rows=[];errors=[];t0=time.time()
    print("[TEST461] FINAL HELD-OUT START — no parameter/template changes after this point")
    for scale in SCALES:
        for ei,e in enumerate(ENTS):
            try:
                # LINKED: A + true B must exist.
                # Stage1 candidate set is padded to requested memory scale.
                trueA=A_BY_E[e["e"]];trueB=next(m for m in MODULES if m["owner"]==e["e"] and m["role"]=="B")
                s1mods=fill([trueA],scale,exclude_owner=None)
                s1=batch(MW1.format(obj=e["obj"]),[m["kv"] for m in s1mods]);cid=decide_id(s1)
                if cid:
                    # Stage2 includes true relation plus distractors. Routing is address metadata only.
                    s2mods=fill([trueB],scale,exclude_owner=None)
                    s2=batch(MW2.format(cid=cid),[m["kv"] for m in s2mods]);place=decide_place(s2)
                else:s2=[];place=None
                lok=(cid==e["T"] and place==e["P"])

                # UNLINKED: A exists but its matching B is deliberately absent.
                # X and unrelated modules remain, testing false linking.
                ux=next(m for m in MODULES if m["owner"]==e["e"] and m["role"]=="X")
                u1mods=fill([trueA],scale,exclude_owner=None)
                u1=batch(MW1.format(obj=e["obj"]),[m["kv"] for m in u1mods]);ucid=decide_id(u1)
                # Explicitly exclude owner's B from unlinked stage2.
                upool=[m for m in MODULES if not (m["owner"]==e["e"] and m["role"]=="B")]
                base=[ux] if ux in upool else []
                u2mods=list(base);pp=upool[:];rng.shuffle(pp)
                for m in pp:
                    if m not in u2mods and len(u2mods)<scale:u2mods.append(m)
                if ucid:
                    u2=batch(MW2.format(cid=ucid),[m["kv"] for m in u2mods]);uplace=decide_place(u2)
                else:u2=[];uplace=None
                uok=(ucid==e["T"] and uplace is None)
                false_link=(uplace is not None)

                row=dict(scale=scale,e=e["e"],fam=e["fam"],linked=int(lok),unlinked=int(uok),false_link=int(false_link),
                         linked_cid=cid,linked_place=place,unlinked_cid=ucid,unlinked_place=uplace,
                         s1_n=len(s1mods),s2_n=len(s2mods) if cid else 0,u1_n=len(u1mods),u2_n=len(u2mods) if ucid else 0)
                rows.append(row)
                print(f"[S{scale:02d} {ei+1:02d}/{N_ENT}] e{e['e']:02d} {e['fam']} "
                      f"L={int(lok)} U={int(uok)} FL={int(false_link)} "
                      f"cid={cid or 'NONE':7s} place={place or 'UNKNOWN'}",flush=True)
            except Exception as ex:
                errors.append(dict(scale=scale,e=e["e"],error=f"{type(ex).__name__}: {ex}"))
                print(f"[S{scale:02d} {ei+1:02d}/{N_ENT}] ERROR {type(ex).__name__}: {ex}",flush=True)

    # NOMEM strict control: no source-derived cache. Correct target must NOT be recoverable.
    nomem_hits=0;nomem_rows=[]
    print("[TEST461] NOMEM control ...")
    for e in ENTS:
        try:
            a=nomem_batch(MW1.format(obj=e["obj"]),1)[0];cid=parse_id(a)
            if cid:
                b=nomem_batch(MW2.format(cid=cid),1)[0];pl=parse_place(b,KNOWN_PLACES)
            else:b="";pl=None
            hit=(cid==e["T"] and pl==e["P"]);nomem_hits+=int(hit)
            nomem_rows.append(dict(e=e["e"],fam=e["fam"],stage1=a,stage2=b,cid=cid,place=pl,strict_hit=int(hit)))
        except Exception as ex:errors.append(dict(control="NOMEM",e=e["e"],error=f"{type(ex).__name__}: {ex}"))

    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    def met(rr):
        n=len(rr);L=sum(r["linked"] for r in rr);U=sum(r["unlinked"] for r in rr);FL=sum(r["false_link"] for r in rr)
        return dict(n=n,L=f"{L}/{n}",U=f"{U}/{n}",false_link=f"{FL}/{n}",L_rate=L/n if n else 0,U_rate=U/n if n else 0,
                    false_link_rate=FL/n if n else 0,L_lo=round(wilson(L,n),3),U_lo=round(wilson(U,n),3))
    overall=met(rows);by_scale={s:met([r for r in rows if r["scale"]==s]) for s in SCALES}
    by_family={f:met([r for r in rows if r["fam"]==f]) for f in FAM}
    fam_scale={f:{s:met([r for r in rows if r["fam"]==f and r["scale"]==s]) for s in SCALES} for f in FAM}
    ldrop=by_scale[2]["L_rate"]-by_scale[16]["L_rate"];udrop=by_scale[2]["U_rate"]-by_scale[16]["U_rate"]
    gates={
        "overall_L_ge_080":overall["L_rate"]>=.80,
        "overall_U_ge_085":overall["U_rate"]>=.85,
        "false_link_le_010":overall["false_link_rate"]<=.10,
        "each_family_L_ge_070":all(by_family[f]["L_rate"]>=.70 for f in FAM),
        "scale16_L_drop_le_010":ldrop<=.10+1e-12,
        "scale16_U_drop_le_010":udrop<=.10+1e-12,
        "NOMEM_strict_zero":nomem_hits==0,
        "frozen_weights":frozen,
        "no_errors":not errors
    }
    verdict="PASS_DEMO_READY" if all(gates.values()) else "FAIL_FINAL_HELDOUT"
    summary=dict(test=461,panel="FINAL_HELDOUT",entities=N_ENT,families=list(FAM),scales=list(SCALES),
                 overall=overall,by_scale=by_scale,by_family=by_family,family_by_scale=fam_scale,
                 scale_drop_2_to_16=dict(L=round(ldrop,4),U=round(udrop,4)),
                 NOMEM_strict=f"{nomem_hits}/{N_ENT}",gates=gates,errors=len(errors),
                 frozen_weights_ok=frozen,minutes=round((time.time()-t0)/60,1),verdict=verdict)
    path="/content/test461_results.json" if os.path.isdir("/content") else "test461_results.json"
    json.dump(dict(summary=summary,rows=rows,nomem=nomem_rows,errors=errors,FAMILIES=FAM),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*120)
    print("TEST461 — FINAL HELD-OUT SUMMARY")
    print("overall:",overall)
    for s in SCALES:print(f"SCALE {s:02d}:",by_scale[s])
    for f in FAM:print(f"{f}:",by_family[f])
    print("scale_drop_2_to_16:",summary["scale_drop_2_to_16"])
    print("NOMEM strict:",summary["NOMEM_strict"])
    print("gates:",gates)
    print("errors:",len(errors))
    print("frozen_weights_ok:",frozen)
    print("minutes:",summary["minutes"])
    print("="*120)
    print("TEST461 VERDICT:",verdict)
    print("results:",path)
    if verdict=="PASS_DEMO_READY":
        print("FINAL: frozen NIRVANA IBR architecture passed the precommitted demo-preceding held-out gates. Proceed to demo packaging without retuning.")
    else:
        print("FINAL: held-out gate failure. Preserve this result unchanged; do not retune or redefine this TEST461 panel.")

if os.environ.get("T461_CPU_ONLY")!="1":run_gpu()
