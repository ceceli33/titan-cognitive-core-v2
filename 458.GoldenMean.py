# TEST458 — KV PROJECTION DOSE-RESPONSE BOUNDARY MAP
# TEST452 K120/V128/OWN engine unchanged. Same frozen TEST454-457 mechanism panel.
# TEST456 KV guard only; lambda=0.00..1.00 step .10. Measures correct-answer logP + greedy behavior.
# Purpose: determine whether destructive rescue/preserved loss is a stable threshold region or a narrow isolated event.
import os,re,json,time,random,hashlib
from collections import Counter
import numpy as np
SEED=452;N_ENT=8;MAX_NEW=32;FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain.";MODEL_ID="Qwen/Qwen2.5-7B-Instruct";LAM=tuple(round(x/10,1) for x in range(11))
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),"F2":("Container {T} holds the {O}.","The {P} houses container {T}.")}
HOP2="Name the place (not a container) where container {cid} is located. If the records do not connect container {cid} to any place, answer UNKNOWN."+STYLE
P_ADJ=["cerise","dun","flax","jasper","lapis","moss","oxblood","pearl","sepia","tawny"];P_NOUN=["bastion","cellar","gallery","jetty","manor","oratory","pavilion","refectory","silo","vault"]
O_ADJ=["burnished","cracked","filigree","glazed","knotted","lidded"];O_NOUN=["amphora","bugle","candelabra","decanter","figurine","gauntlet"]
T454={0:"PRESERVED",1:"DESTRUCTIVE",2:"RESCUE",3:"DESTRUCTIVE",4:"PRESERVED",5:"UNREADABLE",6:"PRESERVED",7:"UNREADABLE"}
def norm(x):return re.sub(r"[^\w]+"," ",str(x).casefold()).strip()
def first(x):
    s=next((z.strip() for z in str(x).splitlines() if z.strip()),"");s=re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*","",s,flags=re.I).strip(" *`\"'")
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
ENTS=make_entities();assert len(ENTS)==8 and len(LAM)==11 and LAM[0]==0 and LAM[-1]==1 and Counter(T454.values())==Counter({"PRESERVED":3,"DESTRUCTIVE":2,"RESCUE":1,"UNREADABLE":2})
print(f"[TEST458] CPU self-test PASS: 8 entities x {len(LAM)} KV doses = {8*len(LAM)} greedy generations + logP")
def run_gpu():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    assert torch.cuda.is_available(),"A100 GPU runtime required"
    torch.manual_seed(SEED);DEV=torch.device("cuda");torch.set_grad_enabled(False)
    tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
    try:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",dtype=torch.bfloat16).eval()
    except TypeError:model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",torch_dtype=torch.bfloat16).eval()
    for p in model.parameters():p.requires_grad_(False)
    cfg=model.config;layers=model.model.layers;NL=len(layers);NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or cfg.hidden_size//cfg.num_attention_heads;KVD=NKV*HD
    PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple)) else [ge])}-{None});enc=lambda s:tok(s,add_special_tokens=False).input_ids
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
    print("[TEST458] building TEST452 codebook ...")
    with torch.inference_mode():
        CO=[forge(s) for s in CORPUS];CB=[]
        for L in range(NL):
            e={}
            for j,n in enumerate(("K","V")):
                R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];BAS=[]
                for h in range(NKV):
                    X=R[:,h];mu=X.mean(0);_,_,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(128,Vh.shape[0]);bb=Vh[:m].T.contiguous()
                    if m<128:bb=F.pad(bb,(0,128-m))
                    MU.append(mu);BAS.append(bb)
                e[n]=(torch.stack(MU),torch.stack(BAS))
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
    def project_guard(A,B,lam):
        if lam==0:return A.clone(),0.0
        aa=A.clone();a=A[1:].float().reshape(-1);b=B[1:].float().reshape(-1);m=min(a.numel(),b.numel());av=a[:m];bv=b[:m];alpha=torch.dot(av,bv)/(torch.dot(bv,bv)+1e-12);g=av-lam*alpha*bv;flat=aa[1:].float().reshape(-1);flat[:m]=g;aa[1:]=flat.reshape_as(aa[1:]).to(aa.dtype);return aa,float(alpha)
    def compose(bank,lam):
        K=[];V=[];ak=[];av=[]
        for L in range(NL):
            A_K,xk=project_guard(bank["A"][0][L],bank["B"][0][L],lam);A_V,xv=project_guard(bank["A"][1][L],bank["B"][1][L],lam)
            K.append(torch.cat([A_K,bank["B"][0][L]],0));V.append(torch.cat([A_V,bank["B"][1][L]],0));ak.append(xk);av.append(xv)
        return K,V,ak,av
    @torch.inference_mode()
    def install(K,V):
        T=K[0].shape[0];cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEV)[None]);KK=[];VV=[]
        for L in range(NL):
            k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2);KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
        return tuple(KK),tuple(VV),T
    def mkcache(kv):
        try:c=DynamicCache(config=cfg)
        except TypeError:c=DynamicCache()
        for L in range(NL):c.update(kv[0][L].clone(),kv[1][L].clone(),L)
        assert c.get_seq_length()==kv[2];return c
    def finite(kv):return all(bool(torch.isfinite(t).all()) for t in list(kv[0])+list(kv[1]))
    GEN=0;EVAL=0
    @torch.inference_mode()
    def greedy(q,kv):
        nonlocal GEN
        qids=enc(FMT.format(q=q));pre=[PAD]*kv[2];cache=mkcache(kv);ids=torch.tensor([pre+qids],device=DEV)
        y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
        new=y[0,ids.shape[1]:].tolist();GEN+=1
        if cache.get_seq_length()!=len(pre)+len(qids)+len(new)-1:raise RuntimeError("cache length mismatch")
        return tok.decode(new,skip_special_tokens=True).strip()
    @torch.inference_mode()
    def answer_logp(q,answer,kv):
        nonlocal EVAL
        qids=enc(FMT.format(q=q));aids=enc(answer)
        if not aids:raise RuntimeError("empty answer tokenization")
        pre=[PAD]*kv[2];prefix=pre+qids;ids=torch.tensor([prefix+aids],device=DEV);cache=mkcache(kv)
        o=model(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,use_cache=False,return_dict=True);logits=o.logits[0];start=len(prefix)-1;lp=F.log_softmax(logits[start:start+len(aids)].float(),dim=-1);vals=lp[torch.arange(len(aids),device=DEV),torch.tensor(aids,device=DEV)];EVAL+=1
        return float(vals.mean()),float(vals.sum()),len(aids)
    rows=[];errors=[];t0=time.time()
    for i,e in enumerate(ENTS,1):
        try:
            a,b=FAM[e["fam"]];bank={"A":packet(a.format(O=e["obj"],T=e["T"])),"B":packet(b.format(T=e["T"],P=e["P"]))};q=HOP2.format(cid=e["T"]);curve=[]
            for lam in LAM:
                K,V,ak,av=compose(bank,lam);kv=install(K,V)
                if not finite(kv):raise RuntimeError(f"non-finite cache lambda={lam}")
                txt=greedy(q,kv);lp,lpsum,nt=answer_logp(q,e["P"],kv);curve.append(dict(lam=lam,ok=place_key(txt)==place_key(e["P"]),logp_mean=lp,logp_sum=lpsum,answer_tokens=nt,text=txt,alphaK_mean=float(np.mean(ak)),alphaV_mean=float(np.mean(av))))
            rows.append(dict(e=e["e"],fam=e["fam"],effect=T454[e["e"]],T=e["T"],P=e["P"],curve=curve))
            bits="".join("1" if z["ok"] else "0" for z in curve);lps=" ".join(f"{z['logp_mean']:+.2f}" for z in curve)
            print(f"[{i:02d}/08] e{e['e']} {T454[e['e']]:<11} behavior={bits} | logP {lps}",flush=True)
        except Exception as ex:
            errors.append(dict(e=e["e"],error=f"{type(ex).__name__}: {ex}"));print("ERROR",errors[-1],flush=True)
    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    if not frozen:errors.append(dict(e=None,error="weight sentinel / frozen check failed"))
    dose=[]
    for j,lam in enumerate(LAM):
        rr=[(r,next(z for z in r["curve"] if z["lam"]==lam)) for r in rows]
        def cnt(cls):return sum(z["ok"] for r,z in rr if r["effect"]==cls)
        dose.append(dict(lam=lam,ALL=sum(z["ok"] for _,z in rr),DES=cnt("DESTRUCTIVE"),PRE=cnt("PRESERVED"),RES=cnt("RESCUE"),UNR=cnt("UNREADABLE"),mean_logp=float(np.mean([z["logp_mean"] for _,z in rr]))))
    boundaries={}
    for r in rows:
        bits=[z["ok"] for z in r["curve"]];trans=[LAM[j] for j in range(1,len(LAM)) if bits[j]!=bits[j-1]];success=[z["lam"] for z in r["curve"] if z["ok"]]
        boundaries[str(r["e"])]=dict(effect=r["effect"],transitions=trans,success_doses=success,n_transitions=len(trans))
    selective=[d["lam"] for d in dose if d["DES"]>=1 and d["PRE"]==3 and d["RES"]==1]
    stable=[]
    for j in range(len(dose)-1):
        if dose[j]["DES"]>=1 and dose[j]["PRE"]==3 and dose[j]["RES"]==1 and dose[j+1]["DES"]>=1 and dose[j+1]["PRE"]==3 and dose[j+1]["RES"]==1:stable.append((dose[j]["lam"],dose[j+1]["lam"]))
    des_success=sum(d["DES"]>0 for d in dose[1:]);des_isolated=des_success==1
    if errors or len(rows)!=8 or GEN!=88 or EVAL!=88:verdict="TECHNICAL_INVALID"
    elif stable:verdict="STABLE_SELECTIVE_WINDOW_OBSERVED"
    elif selective:verdict="ISOLATED_SELECTIVE_DOSE_OBSERVED"
    elif des_success>=2:verdict="DESTRUCTIVE_RESCUE_REGION_NONSELECTIVE"
    elif des_isolated:verdict="NARROW_DESTRUCTIVE_RESCUE_EVENT"
    else:verdict="NO_DESTRUCTIVE_RESCUE"
    S=dict(panel="same frozen TEST454-457 mechanism panel",lambda_grid=list(LAM),dose=dose,boundaries=boundaries,selective_doses=selective,stable_selective_intervals=stable,destructive_success_doses=des_success,generations=GEN,logp_evaluations=EVAL,minutes=round((time.time()-t0)/60,1),frozen_weights_ok=frozen,verdict=verdict)
    path="/content/test458_results.json" if os.path.isdir("/content") else "test458_results.json";json.dump(dict(summary=S,rows=rows,errors=errors),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*118);print("λ    ALL  DES  PRE  RES  UNR | mean correct-answer logP")
    for d in dose:print(f"{d['lam']:.1f}  {d['ALL']:>3}/8 {d['DES']:>3}/2 {d['PRE']:>3}/3 {d['RES']:>3}/1 {d['UNR']:>3}/2 | {d['mean_logp']:+.5f}")
    print("-"*118);print("BOUNDARIES:",boundaries);print("selective_doses:",selective);print("stable_selective_intervals:",stable);print("generations:",GEN,"| logp_evaluations:",EVAL,"| frozen_weights_ok:",frozen,"| minutes:",S["minutes"]);print("="*118)
    print(f"TEST458 VERDICT: {verdict}  results={path}")
    print("Scope: same-panel KV dose-response map; diagnostic boundary assay, not held-out/generalization evidence.")
if os.environ.get("T458_CPU_ONLY")!="1":run_gpu()
