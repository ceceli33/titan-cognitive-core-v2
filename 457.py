# TEST457 — LOCAL BEHAVIORAL CACHE-SENSITIVITY ASSAY
# TEST452 K120/V128/OWN engine unchanged. Same frozen TEST454-456 mechanism panel.
# No generation, no guard. Measures correct-place token log-prob sensitivity to small ±A-direction perturbations of B K/V.
# Central finite difference, eps={0.01,0.02}; K-only/V-only/KV; all 28 layers jointly. Diagnostic, not intervention.
import os,re,json,time,random,hashlib
from collections import Counter
import numpy as np
SEED=452;N_ENT=8;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:";STYLE=" Give only the answer on the first line; do not explain.";MODEL_ID="Qwen/Qwen2.5-7B-Instruct";EPS=(.01,.02)
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),"F2":("Container {T} holds the {O}.","The {P} houses container {T}.")}
HOP2="Name the place (not a container) where container {cid} is located. If the records do not connect container {cid} to any place, answer UNKNOWN."+STYLE
P_ADJ=["cerise","dun","flax","jasper","lapis","moss","oxblood","pearl","sepia","tawny"];P_NOUN=["bastion","cellar","gallery","jetty","manor","oratory","pavilion","refectory","silo","vault"]
O_ADJ=["burnished","cracked","filigree","glazed","knotted","lidded"];O_NOUN=["amphora","bugle","candelabra","decanter","figurine","gauntlet"]
T454={0:"PRESERVED",1:"DESTRUCTIVE",2:"RESCUE",3:"DESTRUCTIVE",4:"PRESERVED",5:"UNREADABLE",6:"PRESERVED",7:"UNREADABLE"}
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
ENTS=make_entities();assert len(ENTS)==8 and Counter(T454.values())==Counter({"PRESERVED":3,"DESTRUCTIVE":2,"RESCUE":1,"UNREADABLE":2})
print("[TEST457] CPU self-test PASS: 8 frozen entities; local behavioral sensitivity; generation-free")
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
    PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id;enc=lambda s:tok(s,add_special_tokens=False).input_ids
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
    print("[TEST457] building TEST452 codebook ...")
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
    def direction(A,B):
        # A-specific direction around B: center content tokens before normalization; slot0 excluded.
        D=[]
        for L in range(NL):
            a=A[L][1:].float();b=B[L][1:].float();m=min(a.shape[0],b.shape[0]);d=a[:m]-b[:m];n=torch.linalg.vector_norm(d)
            D.append(d/(n+1e-12))
        return D
    def perturb(B,D,eps):
        out=[]
        for L in range(NL):
            x=B[L].clone();m=min(x.shape[0]-1,D[L].shape[0]);base=x[1:1+m].float();scale=torch.linalg.vector_norm(base)/(torch.linalg.vector_norm(D[L][:m])+1e-12)
            x[1:1+m]=(base+eps*scale*D[L][:m]).to(x.dtype);out.append(x)
        return out
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
    def joint(A,B,space=None,eps=0.0):
        AK,AV=A;BK,BV=B
        if space:
            if space in ("K","KV"):BK=perturb(BK,direction(AK,BK),eps)
            if space in ("V","KV"):BV=perturb(BV,direction(AV,BV),eps)
        return install([torch.cat([AK[L],BK[L]],0) for L in range(NL)],[torch.cat([AV[L],BV[L]],0) for L in range(NL)])
    @torch.inference_mode()
    def answer_logp(q,answer,kv):
        # Teacher-forced full answer log-prob per token; avoids first-token-only artifact.
        qids=enc(FMT.format(q=q));aids=enc(answer)
        if not aids:raise RuntimeError("empty answer tokenization")
        pre=[PAD]*kv[2];prefix=pre+qids;ids=torch.tensor([prefix+aids],device=DEV);cache=mkcache(kv)
        o=model(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,use_cache=False,return_dict=True)
        logits=o.logits[0];start=len(prefix)-1;lp=F.log_softmax(logits[start:start+len(aids)].float(),dim=-1)
        vals=lp[torch.arange(len(aids),device=DEV),torch.tensor(aids,device=DEV)]
        return float(vals.mean()),float(vals.sum()),len(aids)
    rows=[];errors=[];t0=time.time();EVALS=0
    for i,e in enumerate(ENTS,1):
        try:
            a,b=FAM[e["fam"]];A=packet(a.format(O=e["obj"],T=e["T"]));B=packet(b.format(T=e["T"],P=e["P"]));q=HOP2.format(cid=e["T"])
            raw=joint(A,B);base_mean,base_sum,nt=answer_logp(q,e["P"],raw);EVALS+=1;res={}
            for space in ("K","V","KV"):
                z={}
                for ep in EPS:
                    pp=joint(A,B,space,+ep);pm=joint(A,B,space,-ep);fp,_,_=answer_logp(q,e["P"],pp);fm,_,_=answer_logp(q,e["P"],pm);EVALS+=2
                    deriv=(fp-fm)/(2*ep);curv=(fp-2*base_mean+fm)/(ep*ep);z[str(ep)]=dict(plus=fp,minus=fm,derivative=deriv,curvature=curv,symmetry=abs((fp-base_mean)+(fm-base_mean)))
                res[space]=z
            rows.append(dict(e=e["e"],fam=e["fam"],effect=T454[e["e"]],base_logp_mean=base_mean,base_logp_sum=base_sum,answer_tokens=nt,sensitivity=res))
            s=lambda sp:np.mean([abs(res[sp][str(ep)]["derivative"]) for ep in EPS])
            print(f"[{i:02d}/08] e{e['e']} {T454[e['e']]:<11} base={base_mean:+.4f} | |dK|={s('K'):.4f} |dV|={s('V'):.4f} |dKV|={s('KV'):.4f}",flush=True)
        except Exception as ex:
            errors.append(dict(e=e["e"],error=f"{type(ex).__name__}: {ex}"));print("ERROR",errors[-1],flush=True)
    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    if not frozen:errors.append(dict(e=None,error="weight sentinel / frozen check failed"))
    classes=("DESTRUCTIVE","PRESERVED","RESCUE","UNREADABLE");summary={}
    for sp in ("K","V","KV"):
        summary[sp]={}
        for c in classes:
            rr=[r for r in rows if r["effect"]==c];vals=[]
            for r in rr:
                for ep in EPS:vals.append(abs(r["sensitivity"][sp][str(ep)]["derivative"]))
            summary[sp][c]=float(np.mean(vals)) if vals else None
        summary[sp]["DminusP"]=summary[sp]["DESTRUCTIVE"]-summary[sp]["PRESERVED"]
        summary[sp]["ratio_D_P"]=summary[sp]["DESTRUCTIVE"]/(summary[sp]["PRESERVED"]+1e-12)
    consistency={}
    for sp in ("K","V","KV"):
        consistency[sp]={}
        for c in classes:
            rr=[r for r in rows if r["effect"]==c];same=0;tot=0
            for r in rr:
                d=[r["sensitivity"][sp][str(ep)]["derivative"] for ep in EPS]
                same+=int(d[0]*d[1]>0);tot+=1
            consistency[sp][c]=f"{same}/{tot}"
    if errors or len(rows)!=8:verdict="TECHNICAL_INVALID"
    else:
        best=max(("K","V","KV"),key=lambda s:summary[s]["ratio_D_P"])
        ratio=summary[best]["ratio_D_P"]
        verdict="BEHAVIORAL_SENSITIVITY_CANDIDATE" if ratio>=1.5 else "NO_CLEAR_SENSITIVITY_SEPARATION"
    S=dict(panel="same frozen TEST454-456 mechanism panel",eps=list(EPS),metric="central finite-difference of mean correct-answer token log-prob",summary=summary,sign_consistency=consistency,evaluations=EVALS,generations=0,minutes=round((time.time()-t0)/60,1),frozen_weights_ok=frozen,verdict=verdict)
    path="/content/test457_results.json" if os.path.isdir("/content") else "test457_results.json";json.dump(dict(summary=S,rows=rows,errors=errors),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*108);print("SPACE | |d| DES     PRE     RES     UNR   | D-P      D/P")
    for sp in ("K","V","KV"):
        z=summary[sp];print(f"{sp:<5} | {z['DESTRUCTIVE']:.6f} {z['PRESERVED']:.6f} {z['RESCUE']:.6f} {z['UNREADABLE']:.6f} | {z['DminusP']:+.6f} {z['ratio_D_P']:.3f}")
    print("-"*108);print("sign consistency eps .01/.02:",consistency);print("evaluations:",EVALS,"| generations: 0 | frozen_weights_ok:",frozen,"| minutes:",S["minutes"]);print("="*108)
    print(f"TEST457 VERDICT: {verdict}  results={path}")
    print("Scope: same-panel local finite-difference assay; diagnostic only. No guard, causal rescue or generalization claim.")
if os.environ.get("T457_CPU_ONLY")!="1":run_gpu()
