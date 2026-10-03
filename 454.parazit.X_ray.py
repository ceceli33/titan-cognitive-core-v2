# TEST454 — COMPACT CACHE-COMPOSITION X-RAY
# TEST452 K120/V128/OWN engine unchanged. Stage2 only: gold T -> place P.
# B-SOLO vs AB vs BA + generation-free layerwise K/V overlap. Frozen weights; no source text at read time.
import os,re,json,time,random,hashlib
from collections import Counter
import numpy as np
SEED=452;N_ENT=8;MAX_NEW=32;FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain."
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),"F2":("Container {T} holds the {O}.","The {P} houses container {T}.")}
HOP2="Name the place (not a container) where container {cid} is located. If the records do not connect container {cid} to any place, answer UNKNOWN."+STYLE
P_ADJ=["cerise","dun","flax","jasper","lapis","moss","oxblood","pearl","sepia","tawny"];P_NOUN=["bastion","cellar","gallery","jetty","manor","oratory","pavilion","refectory","silo","vault"]
O_ADJ=["burnished","cracked","filigree","glazed","knotted","lidded"];O_NOUN=["amphora","bugle","candelabra","decanter","figurine","gauntlet"]
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
ENTS=make_entities()
assert len(ENTS)==8 and len({e["T"] for e in ENTS})==8
print("[TEST454] CPU self-test PASS: 8 entities; 24 Stage2 generations expected")

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
    print("[TEST454] building TEST452 codebook ...")
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
    def compose(order,bank):
        K=[torch.cat([bank[r][0][L] for r in order],0) for L in range(NL)];V=[torch.cat([bank[r][1][L] for r in order],0) for L in range(NL)]
        return install(K,V)
    STATS=Counter()
    @torch.inference_mode()
    def reader(q,kv):
        qids=enc(FMT.format(q=q));pre=[PAD]*kv[2];cache=mkcache(kv);ids=torch.tensor([pre+qids],device=DEV)
        y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
        new=y[0,ids.shape[1]:].tolist();STATS["generations"]+=1
        if cache.get_seq_length()!=len(pre)+len(qids)+len(new)-1:raise RuntimeError("cache length mismatch")
        return tok.decode(new,skip_special_tokens=True).strip()
    def overlap(A,B):
        out={"K":[],"V":[]}
        for j,n in enumerate(("K","V")):
            for L in range(NL):
                a=A[j][L][1:].float().reshape(-1);b=B[j][L][1:].float().reshape(-1)
                m=min(a.numel(),b.numel());out[n].append(float(F.cosine_similarity(a[:m][None],b[:m][None]).item()))
        return out
    def finite(kv):return all(bool(torch.isfinite(t).all()) for t in list(kv[0])+list(kv[1]))
    rows=[];errors=[];t0=time.time()
    for i,e in enumerate(ENTS,1):
        try:
            A,L=FAM[e["fam"]];srcA=A.format(O=e["obj"],T=e["T"]);srcB=L.format(T=e["T"],P=e["P"]);bank={"A":packet(srcA),"B":packet(srcB)}
            solo=install(*bank["B"]);ab=compose("AB",bank);ba=compose("BA",bank)
            if not all(finite(x) for x in (solo,ab,ba)):raise RuntimeError("non-finite cache")
            q=HOP2.format(cid=e["T"]);ta=reader(q,solo);tab=reader(q,ab);tba=reader(q,ba)
            ok=lambda x:place_key(x)==place_key(e["P"])
            s,a,b=ok(ta),ok(tab),ok(tba)
            if s and not(a and b):cls="DESTRUCTIVE"
            elif not s and (a or b):cls="RESCUE"
            elif s and a and b:cls="PRESERVED"
            else:cls="UNREADABLE"
            ov=overlap(bank["A"],bank["B"])
            row=dict(e=e["e"],fam=e["fam"],T=e["T"],P=e["P"],SOLO=dict(text=ta,ok=s),AB=dict(text=tab,ok=a),BA=dict(text=tba,ok=b),effect=cls,K_cos=ov["K"],V_cos=ov["V"])
            rows.append(row)
            print(f"[{i:02d}/08] e{i-1} {e['P']:<18} SOLO={int(s)} AB={int(a)} BA={int(b)} effect={cls:<11} K={np.mean(ov['K']):+.4f} V={np.mean(ov['V']):+.4f}",flush=True)
        except Exception as ex:
            errors.append(dict(e=e["e"],error=f"{type(ex).__name__}: {ex}"));print("ERROR",errors[-1],flush=True)
    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    if not frozen:errors.append(dict(e=None,error="weight sentinel / frozen check failed"))
    def n(arm):return sum(r[arm]["ok"] for r in rows)
    effects=Counter(r["effect"] for r in rows)
    layer={}
    for L in range(NL):
        d=[r for r in rows if r["effect"]=="DESTRUCTIVE"];p=[r for r in rows if r["effect"]=="PRESERVED"];rs=[r for r in rows if r["effect"]=="RESCUE"]
        layer[L]=dict(K_all=round(float(np.mean([r["K_cos"][L] for r in rows])),5),V_all=round(float(np.mean([r["V_cos"][L] for r in rows])),5),
                      K_des=round(float(np.mean([r["K_cos"][L] for r in d])),5) if d else None,V_des=round(float(np.mean([r["V_cos"][L] for r in d])),5) if d else None,
                      K_pre=round(float(np.mean([r["K_cos"][L] for r in p])),5) if p else None,V_pre=round(float(np.mean([r["V_cos"][L] for r in p])),5) if p else None,
                      K_res=round(float(np.mean([r["K_cos"][L] for r in rs])),5) if rs else None,V_res=round(float(np.mean([r["V_cos"][L] for r in rs])),5) if rs else None)
    order_diff=sum(r["AB"]["ok"]!=r["BA"]["ok"] for r in rows)
    S=dict(SOLO=f"{n('SOLO')}/{len(rows)}",AB=f"{n('AB')}/{len(rows)}",BA=f"{n('BA')}/{len(rows)}",effects=dict(effects),order_sensitive=f"{order_diff}/{len(rows)}",
           generations=STATS["generations"],minutes=round((time.time()-t0)/60,1),frozen_weights_ok=frozen)
    if errors or len(rows)!=N_ENT or STATS["generations"]!=3*N_ENT:S["verdict"]="TECHNICAL_INVALID"
    elif effects["DESTRUCTIVE"] or effects["RESCUE"]:S["verdict"]="COMPOSITION_EFFECT_OBSERVED"
    else:S["verdict"]="NO_COMPOSITION_EFFECT"
    path="/content/test454_results.json" if os.path.isdir("/content") else "test454_results.json"
    json.dump(dict(summary=S,layer_xray=layer,rows=rows,errors=errors),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*90)
    for k,v in S.items():print(f"{k}: {v}")
    print("-"*90)
    print("layer  K_all    V_all    K_des    V_des    K_pre    V_pre    K_res    V_res")
    for L,x in layer.items():print(f"{L:02d}  {x['K_all']:+.5f} {x['V_all']:+.5f} {str(x['K_des']):>8} {str(x['V_des']):>8} {str(x['K_pre']):>8} {str(x['V_pre']):>8} {str(x['K_res']):>8} {str(x['V_res']):>8}")
    print("="*90);print(f"TEST454 VERDICT: {S['verdict']}  results={path}")
    print("Scope: gold-ID Stage2 mechanism probe only. K/V cosine is diagnostic; it does not by itself identify a causal interference direction.")
if os.environ.get("T454_CPU_ONLY")!="1":run_gpu()
