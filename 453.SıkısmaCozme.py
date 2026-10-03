# TEST453 (compact). TEST452 K120/V128/OWN engine unchanged.
# Question: why does MW-readable memory fail after joint-cache composition?
# 8 entities x AB/BA. SOLO vs PADMATCH (same joint length/target slot, other module -> equal-length PAD) vs JOINT.
# Stage1 target=A (object->T), Stage2 target=B (T->place). Frozen model; no source text at read time.
import os,re,json,time,random,hashlib
from collections import Counter
SEED=452;N_ENT=8;MAX_NEW=32;FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain."
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),"F2":("Container {T} holds the {O}.","The {P} houses container {T}.")}
HOP1="Which container holds the {obj}? Give only the container ID on the first line; do not explain."
HOP2="Name the place (not a container) where container {cid} is located. If the records do not connect container {cid} to any place, answer UNKNOWN."+STYLE
P_ADJ=["cerise","dun","flax","jasper","lapis","moss","oxblood","pearl","sepia","tawny"];P_NOUN=["bastion","cellar","gallery","jetty","manor","oratory","pavilion","refectory","silo","vault"]
O_ADJ=["burnished","cracked","filigree","glazed","knotted","lidded"];O_NOUN=["amphora","bugle","candelabra","decanter","figurine","gauntlet"]
ID_RE=re.compile(r"\b([A-Z]{2}-\d{3})\b",re.I)
def norm(x):return re.sub(r"[^\w]+"," ",str(x).casefold()).strip()
def first(x):
    s=next((z.strip() for z in str(x).splitlines() if z.strip()),"")
    s=re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",s,flags=re.I).strip(" *`\"'")
    return re.split(r"(?<=[.!?])\s+",s)[0] if s else ""
def place_key(x):return re.sub(r"^(?:the|in the|at the|in|at)\s+","",norm(first(x))).strip()
def make_entities():
    rng=random.Random(SEED);places=[f"{a} {n}" for a in P_ADJ for n in P_NOUN];objs=[f"{a} {n}" for a in O_ADJ for n in O_NOUN];rng.shuffle(places);rng.shuffle(objs);ids=set();out=[]
    def nid():
        while True:
            x="".join(rng.choice("BCDFGHJKLMNPRSTVWXZ") for _ in range(2))+"-"+str(rng.randint(100,999))
            if x not in ids:ids.add(x);return x
    for i in range(N_ENT):
        P=places.pop(0);Q=places.pop(next(j for j,q in enumerate(places) if not set(q.split())&set(P.split())))
        out.append(dict(e=i,obj=objs[i],P=P,Q=Q,T=nid(),D=nid(),D2=nid(),fam="F1" if i%2==0 else "F2"))
    return out
def make_cases():
    out=[]
    for e in make_entities():
        A,L=FAM[e["fam"]];mods={"A":A.format(O=e["obj"],T=e["T"]),"B":L.format(T=e["T"],P=e["P"])}
        for br in ("AB","BA"):out.append(dict(e,id=f"t453-{e['e']}:{br}",br=br,mods=mods,order=list(br)))
    return out
CASES=make_cases()
assert len(CASES)==16 and all(set(c["order"])=={"A","B"} for c in CASES)
print(f"[TEST453] CPU self-test PASS: {len(CASES)} cases (8 entities x AB/BA)")

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
    cfg=model.config;layers=model.model.layers;NL=len(layers);NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or cfg.hidden_size//cfg.num_attention_heads;KVD=NKV*HD
    PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple)) else [ge])}-{None})
    enc=lambda s:tok(s,add_special_tokens=False).input_ids
    def sentinel():
        h=hashlib.sha256()
        for t in (layers[0].self_attn.q_proj.weight,layers[NL//2].mlp.down_proj.weight,model.lm_head.weight):h.update(t.detach().reshape(-1)[:4096].float().cpu().numpy().tobytes())
        return h.hexdigest()
    SENT0=sentinel()
    @torch.inference_mode()
    def kv_from_ids(ids):
        o=model(input_ids=torch.tensor([ids],device=DEV),output_hidden_states=True,use_cache=False,return_dict=True);K=[];V=[]
        for L in range(NL):
            z=layers[L].input_layernorm(o.hidden_states[L][0]);a=layers[L].self_attn;K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
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
        assert c.get_seq_length()==kv[2];return c
    print("[TEST453] building TEST452 codebook ...")
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
                content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16);rows.append(torch.cat([X[L][:1],content]))
            out[n]=rows
        return out["K"],out["V"]
    def cat_raw(parts):
        return [torch.cat([x[0][L] for x in parts],0) for L in range(NL)],[torch.cat([x[1][L] for x in parts],0) for L in range(NL)]
    def pad_raw(T):return kv_from_ids([PAD]*T)
    def matched(order,bank,target,pads):
        parts=[bank[r] if r==target else pads[r] for r in order];K,V=cat_raw(parts);return install(K,V)
    STATS=Counter()
    @torch.inference_mode()
    def reader(q,kv):
        qids=enc(FMT.format(q=q));pre=[PAD]*kv[2];cache=mkcache(kv);ids=torch.tensor([pre+qids],device=DEV)
        y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
        new=y[0,ids.shape[1]:].tolist();STATS["generations"]+=1
        if cache.get_seq_length()!=len(pre)+len(qids)+len(new)-1:raise RuntimeError("cache length mismatch")
        return tok.decode(new,skip_special_tokens=True).strip()
    def run_hop(c,s1mem,s2mem):
        t1=reader(HOP1.format(obj=c["obj"]),s1mem);m=ID_RE.search(first(t1));cid=m.group(1).upper() if m else None
        t2=reader(HOP2.format(cid=cid),s2mem) if cid else "UNKNOWN"
        id_ok=cid==c["T"];place_ok=place_key(t2)==place_key(c["P"]);return dict(stage1=t1,id=cid,stage2=t2,id_ok=id_ok,place_ok=place_ok,strict=id_ok and place_ok)
    def finite(kv):return all(bool(torch.isfinite(t).all()) for t in list(kv[0])+list(kv[1]))
    rows=[];errors=[];t0=time.time()
    for i,c in enumerate(CASES,1):
        try:
            bank={r:packet(c["mods"][r]) for r in ("A","B")};pads={r:pad_raw(bank[r][0][0].shape[0]) for r in ("A","B")}
            soloA=install(*bank["A"]);soloB=install(*bank["B"])
            pmA=matched(c["order"],bank,"A",pads);pmB=matched(c["order"],bank,"B",pads)
            KJ,VJ=cat_raw([bank[r] for r in c["order"]]);joint=install(KJ,VJ)
            allkv=[soloA,soloB,pmA,pmB,joint]
            if not all(finite(x) for x in allkv):raise RuntimeError("non-finite cache")
            solo=run_hop(c,soloA,soloB);padmatch=run_hop(c,pmA,pmB);joint_r=run_hop(c,joint,joint)
            row=dict(id=c["id"],e=c["e"],br=c["br"],fam=c["fam"],P=c["P"],T=c["T"],SOLO=solo,PADMATCH=padmatch,JOINT=joint_r);rows.append(row)
            print(f"[{i:02d}/16] {c['id']:<11} SOLO={int(solo['strict'])} PADMATCH={int(padmatch['strict'])} JOINT={int(joint_r['strict'])} | "
                  f"S1 {int(solo['id_ok'])}/{int(padmatch['id_ok'])}/{int(joint_r['id_ok'])} S2 {int(solo['place_ok'])}/{int(padmatch['place_ok'])}/{int(joint_r['place_ok'])}",flush=True)
        except Exception as ex:
            errors.append(dict(case=c["id"],error=f"{type(ex).__name__}: {ex}"));print("ERROR",errors[-1],flush=True)
    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    if not frozen:errors.append(dict(case=None,error="weight sentinel / frozen check failed"))
    def score(arm,key="strict"):return sum(r[arm][key] for r in rows)
    S={a:dict(strict=f"{score(a)}/{len(rows)}",stage1=f"{score(a,'id_ok')}/{len(rows)}",stage2=f"{score(a,'place_ok')}/{len(rows)}") for a in ("SOLO","PADMATCH","JOINT")}
    S["paired"]=dict(
        SOLO_to_PADMATCH_loss=sum(r["SOLO"]["strict"] and not r["PADMATCH"]["strict"] for r in rows),
        PADMATCH_to_JOINT_loss=sum(r["PADMATCH"]["strict"] and not r["JOINT"]["strict"] for r in rows),
        JOINT_to_PADMATCH_gain=sum(not r["JOINT"]["strict"] and r["PADMATCH"]["strict"] for r in rows))
    if errors or len(rows)!=len(CASES):verdict="TECHNICAL_INVALID"
    else:
        ps=score("SOLO")/len(rows);pp=score("PADMATCH")/len(rows);pj=score("JOINT")/len(rows)
        if ps>=.75 and pp>=.75 and pj<=pp-.25:verdict="CONTENT_INTERFERENCE_SUPPORTED"
        elif ps>=.75 and pp<=ps-.25:verdict="POSITION_LENGTH_EFFECT_SUPPORTED"
        elif ps>=.75 and pp<ps-.10 and pj<pp-.10:verdict="MIXED_EFFECT"
        else:verdict="INCONCLUSIVE"
    S["verdict"]=verdict;S["generations"]=STATS["generations"];S["minutes"]=round((time.time()-t0)/60,1);S["frozen_weights_ok"]=frozen
    path="/content/test453_results.json" if os.path.isdir("/content") else "test453_results.json";json.dump(dict(summary=S,rows=rows,errors=errors),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*90)
    for k,v in S.items():print(f"{k}: {v}")
    print("="*90);print(f"TEST453 VERDICT: {verdict}  results={path}")
    print("Interpretation: PADMATCH preserves joint total length and target-module slot while replacing the other module with equal-length PAD-derived KV.")
    print("If SOLO≈PADMATCH >> JOINT, competing module content is implicated. If SOLO >> PADMATCH, joint layout/position/length itself contributes.")
if os.environ.get("T453_CPU_ONLY")!="1":run_gpu()
