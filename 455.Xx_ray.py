# TEST455 — V-SPACE INTERFERENCE COMPONENT ASSAY
# TEST452 K120/V128/OWN engine unchanged. No generation, no intervention.
# Same 8-entity mechanism panel. Measures A_V decomposition relative to B_V:
# parallel projection, orthogonal residual, energy fraction and cosine by layer.
import os,re,json,time,random,hashlib
from collections import Counter
import numpy as np
SEED=452;N_ENT=8;SEP="\n\n";MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),"F2":("Container {T} holds the {O}.","The {P} houses container {T}.")}
P_ADJ=["cerise","dun","flax","jasper","lapis","moss","oxblood","pearl","sepia","tawny"];P_NOUN=["bastion","cellar","gallery","jetty","manor","oratory","pavilion","refectory","silo","vault"]
O_ADJ=["burnished","cracked","filigree","glazed","knotted","lidded"];O_NOUN=["amphora","bugle","candelabra","decanter","figurine","gauntlet"]
# Frozen behavioral labels from TEST454; TEST455 does not regenerate or relabel outcomes.
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
ENTS=make_entities()
assert len(ENTS)==8 and Counter(T454.values())==Counter({"PRESERVED":3,"DESTRUCTIVE":2,"RESCUE":1,"UNREADABLE":2})
print("[TEST455] CPU self-test PASS: 8 frozen TEST454 entities; generation-free assay")

def run_gpu():
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer,AutoModelForCausalLM
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
    print("[TEST455] building TEST452 codebook ...")
    with torch.inference_mode():
        CO=[forge(s) for s in CORPUS];CB=[]
        for L in range(NL):
            e={}
            for j,n in enumerate(("K","V")):
                R=torch.cat([c[j][L][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];B=[]
                for h in range(NKV):
                    X=R[:,h];mu=X.mean(0);_,_,Vh=torch.linalg.svd(X-mu,full_matrices=False);m=min(128,Vh.shape[0]);bb=Vh[:m].T.contiguous()
                    if m<128:bb=F.pad(bb,(0,128-m))
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
    def common_flat(x,y):
        a=x[1:].float().reshape(-1);b=y[1:].float().reshape(-1);m=min(a.numel(),b.numel());return a[:m],b[:m]
    def decomp(a,b):
        eps=1e-12;aa=float(torch.dot(a,a));bb=float(torch.dot(b,b));ab=float(torch.dot(a,b))
        alpha=ab/(bb+eps);par=alpha*b;orth=a-par
        na=float(torch.linalg.vector_norm(a));nb=float(torch.linalg.vector_norm(b));np_=float(torch.linalg.vector_norm(par));no=float(torch.linalg.vector_norm(orth))
        cos=ab/(na*nb+eps);frac=(np_*np_)/(na*na+eps);orth_frac=(no*no)/(na*na+eps)
        return dict(cos=cos,alpha=alpha,parallel_frac=frac,orthogonal_frac=orth_frac,A_norm=na,B_norm=nb,parallel_norm=np_,orthogonal_norm=no)
    rows=[];errors=[];t0=time.time()
    for i,e in enumerate(ENTS,1):
        try:
            A,L=FAM[e["fam"]];pA=packet(A.format(O=e["obj"],T=e["T"]));pB=packet(L.format(T=e["T"],P=e["P"]));layers_out=[]
            for z in range(NL):
                ka,kb=common_flat(pA[0][z],pB[0][z]);va,vb=common_flat(pA[1][z],pB[1][z])
                layers_out.append(dict(layer=z,K=decomp(ka,kb),V=decomp(va,vb)))
            r=dict(e=e["e"],fam=e["fam"],effect=T454[e["e"]],layers=layers_out);rows.append(r)
            vm=np.mean([x["V"]["parallel_frac"] for x in layers_out]);km=np.mean([x["K"]["parallel_frac"] for x in layers_out])
            print(f"[{i:02d}/08] e{e['e']} {T454[e['e']]:<11} Kpar={km:.5f} Vpar={vm:.5f}",flush=True)
        except Exception as ex:
            errors.append(dict(e=e["e"],error=f"{type(ex).__name__}: {ex}"));print("ERROR",errors[-1],flush=True)
    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    if not frozen:errors.append(dict(e=None,error="weight sentinel / frozen check failed"))
    classes=("DESTRUCTIVE","PRESERVED","RESCUE","UNREADABLE")
    summary=[]
    for L in range(NL):
        z={"layer":L}
        for c in classes:
            rr=[r for r in rows if r["effect"]==c]
            for space in ("K","V"):
                for metric in ("cos","alpha","parallel_frac","orthogonal_frac"):
                    z[f"{space}_{metric}_{c[:3]}"]=float(np.mean([r["layers"][L][space][metric] for r in rr])) if rr else None
        if any(r["effect"]=="DESTRUCTIVE" for r in rows) and any(r["effect"]=="PRESERVED" for r in rows):
            z["K_DminusP"]=z["K_parallel_frac_DES"]-z["K_parallel_frac_PRE"];z["V_DminusP"]=z["V_parallel_frac_DES"]-z["V_parallel_frac_PRE"]
        summary.append(z)
    rankedV=sorted(summary,key=lambda x:abs(x["V_DminusP"]),reverse=True);rankedK=sorted(summary,key=lambda x:abs(x["K_DminusP"]),reverse=True)
    topV=[dict(layer=x["layer"],delta=x["V_DminusP"],des=x["V_parallel_frac_DES"],pre=x["V_parallel_frac_PRE"]) for x in rankedV[:8]]
    topK=[dict(layer=x["layer"],delta=x["K_DminusP"],des=x["K_parallel_frac_DES"],pre=x["K_parallel_frac_PRE"]) for x in rankedK[:8]]
    # Exploratory only: small fixed panel, no significance/generalization claim.
    posV=sum(x["V_DminusP"]>0 for x in summary);posK=sum(x["K_DminusP"]>0 for x in summary)
    if errors or len(rows)!=N_ENT:verdict="TECHNICAL_INVALID"
    elif posV>=20:verdict="V_PARALLEL_ENRICHMENT_CANDIDATE"
    elif posK>=20:verdict="K_PARALLEL_ENRICHMENT_CANDIDATE"
    else:verdict="NO_GLOBAL_PARALLEL_ENRICHMENT"
    S=dict(panel="same frozen TEST454 mechanism panel",counts=dict(Counter(T454.values())),V_DminusP_positive_layers=f"{posV}/{NL}",K_DminusP_positive_layers=f"{posK}/{NL}",topV=topV,topK=topK,generations=0,minutes=round((time.time()-t0)/60,1),frozen_weights_ok=frozen,verdict=verdict)
    path="/content/test455_results.json" if os.path.isdir("/content") else "test455_results.json"
    json.dump(dict(summary=S,layer_summary=summary,rows=rows,errors=errors),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*104)
    print("layer | Vpar DES   PRE    Δ(D-P) | Kpar DES   PRE    Δ(D-P)")
    for x in summary:print(f"{x['layer']:02d}    | {x['V_parallel_frac_DES']:.5f}  {x['V_parallel_frac_PRE']:.5f} {x['V_DminusP']:+.5f} | {x['K_parallel_frac_DES']:.5f}  {x['K_parallel_frac_PRE']:.5f} {x['K_DminusP']:+.5f}")
    print("-"*104)
    print("V Δ>0:",S["V_DminusP_positive_layers"],"| K Δ>0:",S["K_DminusP_positive_layers"])
    print("TOP V:",[(x["layer"],round(x["delta"],5)) for x in topV])
    print("TOP K:",[(x["layer"],round(x["delta"],5)) for x in topK])
    print("generations: 0 | frozen_weights_ok:",frozen,"| minutes:",S["minutes"])
    print("="*104);print(f"TEST455 VERDICT: {verdict}  results={path}")
    print("Scope: exploratory geometry assay on the same TEST454 panel; no intervention, no causal or generalization claim.")
if os.environ.get("T455_CPU_ONLY")!="1":run_gpu()
