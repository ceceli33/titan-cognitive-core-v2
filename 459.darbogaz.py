TEST459'u kilitli held-out doğrulama olarak kuruyorum: yeni 32 entity, yeni seed; λ=0.20 TEST458'den donduruluyor ve test sonucuna göre hiçbir ayar yapılmıyor. Aynı entity üzerinde B-SOLO, RAW A+B ve FROZEN KV020 A+B ölçülüyor; böylece yalnız RAW→guard geçişini değil, joint composition'ın SOLO'ya göre gerçekten ne yaptığını da göreceğiz.


# TEST459 — FROZEN KV020 HELD-OUT GENERALIZATION
# TEST452 K120/V128/OWN engine unchanged. TEST458 selected lambda=0.20 is FROZEN before this test.
# NEW 32 entities, NEW seed, no tuning. B-SOLO vs RAW A+B vs frozen KV020 A+B.
# Primary: RAW->KV020 rescued/damaged/net benefit + retention. Secondary: SOLO reference for composition.
import os,re,json,time,random,hashlib,math
import numpy as np
SEED=459;N_ENT=32;LAM=.20;MAX_NEW=32;FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n";STYLE=" Give only the answer on the first line; do not explain.";MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.","Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.","The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.","Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.","Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
FAM={"F1":("The {O} sits in container {T}.","Container {T} is located at the {P}."),"F2":("Container {T} holds the {O}.","The {P} houses container {T}.")}
HOP2="Name the place (not a container) where container {cid} is located. If the records do not connect container {cid} to any place, answer UNKNOWN."+STYLE
P_ADJ=["cerise","dun","flax","jasper","lapis","moss","oxblood","pearl","sepia","tawny","umber","verdant","saffron","indigo","russet","ivory","cobalt","amber","slate","coral"]
P_NOUN=["bastion","cellar","gallery","jetty","manor","oratory","pavilion","refectory","silo","vault","arcade","depot","foundry","hangar","lodge","rotunda","terrace","workshop","courtyard","archive"]
O_ADJ=["burnished","cracked","filigree","glazed","knotted","lidded","etched","woven","polished","carved","lacquered","riveted"]
O_NOUN=["amphora","bugle","candelabra","decanter","figurine","gauntlet","lantern","medallion","reliquary","tripod","urn","whistle"]
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
        out.append(dict(e=i,obj=objs[i],P=P,Q=Q,T=nid(),fam="F1" if i%2==0 else "F2"))
    return out
ENTS=make_entities();assert len(ENTS)==32 and len({e["P"] for e in ENTS})==32 and len({e["T"] for e in ENTS})==32
print(f"[TEST459] CPU self-test PASS: HELD-OUT seed={SEED}, N={N_ENT}, frozen lambda={LAM:.2f}; 3 arms/entity = {N_ENT*3} generations")
def wilson(k,n,z=1.96):
    if not n:return 0.,1.
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
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
    print("[TEST459] building frozen TEST452 codebook ...")
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
    def project_guard(A,B):
        aa=A.clone();a=A[1:].float().reshape(-1);b=B[1:].float().reshape(-1);m=min(a.numel(),b.numel());av=a[:m];bv=b[:m];alpha=torch.dot(av,bv)/(torch.dot(bv,bv)+1e-12);g=av-LAM*alpha*bv;flat=aa[1:].float().reshape(-1);flat[:m]=g;aa[1:]=flat.reshape_as(aa[1:]).to(aa.dtype);return aa,float(alpha)
    @torch.inference_mode()
    def install(K,V):
        T=K[0].shape[0];cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEV)[None]);KK=[];VV=[]
        for L in range(NL):
            k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2);KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
        return tuple(KK),tuple(VV),T
    def solo(B):return install(B[0],B[1])
    def raw(A,B):return install([torch.cat([A[0][L],B[0][L]],0) for L in range(NL)],[torch.cat([A[1][L],B[1][L]],0) for L in range(NL)])
    def guard(A,B):
        K=[];V=[];aK=[];aV=[]
        for L in range(NL):
            k,xk=project_guard(A[0][L],B[0][L]);v,xv=project_guard(A[1][L],B[1][L]);K.append(torch.cat([k,B[0][L]],0));V.append(torch.cat([v,B[1][L]],0));aK.append(xk);aV.append(xv)
        return install(K,V),aK,aV
    def mkcache(kv):
        try:c=DynamicCache(config=cfg)
        except TypeError:c=DynamicCache()
        for L in range(NL):c.update(kv[0][L].clone(),kv[1][L].clone(),L)
        assert c.get_seq_length()==kv[2];return c
    def finite(kv):return all(bool(torch.isfinite(t).all()) for t in list(kv[0])+list(kv[1]))
    GEN=0
    @torch.inference_mode()
    def reader(q,kv):
        nonlocal GEN
        qids=enc(FMT.format(q=q));pre=[PAD]*kv[2];cache=mkcache(kv);ids=torch.tensor([pre+qids],device=DEV)
        y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
        new=y[0,ids.shape[1]:].tolist();GEN+=1
        if cache.get_seq_length()!=len(pre)+len(qids)+len(new)-1:raise RuntimeError("cache length mismatch")
        return tok.decode(new,skip_special_tokens=True).strip()
    rows=[];errors=[];t0=time.time()
    for i,e in enumerate(ENTS,1):
        try:
            a,b=FAM[e["fam"]];A=packet(a.format(O=e["obj"],T=e["T"]));B=packet(b.format(T=e["T"],P=e["P"]));g,aK,aV=guard(A,B);caches={"SOLO":solo(B),"RAW":raw(A,B),"KV020":g}
            if not all(finite(x) for x in caches.values()):raise RuntimeError("non-finite cache")
            q=HOP2.format(cid=e["T"]);res={}
            for arm in ("SOLO","RAW","KV020"):
                txt=reader(q,caches[arm]);res[arm]={"ok":place_key(txt)==place_key(e["P"]),"text":txt}
            ro=res["RAW"]["ok"];go=res["KV020"]["ok"]
            transition="KEEP_OK" if ro and go else "DAMAGED" if ro and not go else "RESCUED" if not ro and go else "KEEP_FAIL"
            rows.append(dict(e=e["e"],fam=e["fam"],obj=e["obj"],T=e["T"],P=e["P"],results=res,transition=transition,alphaK_mean=float(np.mean(aK)),alphaV_mean=float(np.mean(aV))))
            print(f"[{i:02d}/32] e{e['e']:02d} {e['fam']} SOLO={int(res['SOLO']['ok'])} RAW={int(ro)} KV020={int(go)} {transition:<9} P={e['P']}",flush=True)
        except Exception as ex:
            errors.append(dict(e=e["e"],error=f"{type(ex).__name__}: {ex}"));print("ERROR",errors[-1],flush=True)
    frozen=sentinel()==SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    if not frozen:errors.append(dict(e=None,error="weight sentinel / frozen check failed"))
    n=len(rows);solo_ok=sum(r["results"]["SOLO"]["ok"] for r in rows);raw_ok=sum(r["results"]["RAW"]["ok"] for r in rows);guard_ok=sum(r["results"]["KV020"]["ok"] for r in rows)
    rescued=sum(r["transition"]=="RESCUED" for r in rows);damaged=sum(r["transition"]=="DAMAGED" for r in rows);keep_ok=sum(r["transition"]=="KEEP_OK" for r in rows);keep_fail=sum(r["transition"]=="KEEP_FAIL" for r in rows)
    raw_fail=n-raw_ok;retention=keep_ok/raw_ok if raw_ok else float("nan");rescue_rate=rescued/raw_fail if raw_fail else float("nan");net=rescued-damaged
    rlo,rhi=wilson(keep_ok,raw_ok);qlo,qhi=wilson(rescued,raw_fail)
    # McNemar exact two-sided on discordant RAW/KV020 pairs.
    disc=rescued+damaged
    if disc:
        from math import comb
        tail=sum(comb(disc,k) for k in range(0,min(rescued,damaged)+1))/(2**disc);mcnemar_p=min(1.0,2*tail)
    else:mcnemar_p=1.0
    # Precommitted interpretation: guard candidate requires positive net benefit and >=90% retention; strong requires >=95%.
    if errors or n!=N_ENT or GEN!=N_ENT*3:verdict="TECHNICAL_INVALID"
    elif net>0 and retention>=.95:verdict="HELDOUT_GUARD_SIGNAL_STRONG"
    elif net>0 and retention>=.90:verdict="HELDOUT_GUARD_SIGNAL_CANDIDATE"
    elif net>0:verdict="POSITIVE_NET_BUT_RETENTION_COST"
    elif net==0:verdict="NO_NET_HELDOUT_BENEFIT"
    else:verdict="HELDOUT_GUARD_HARM"
    byfam={}
    for f in ("F1","F2"):
        rr=[r for r in rows if r["fam"]==f];byfam[f]={"n":len(rr),"SOLO":sum(r["results"]["SOLO"]["ok"] for r in rr),"RAW":sum(r["results"]["RAW"]["ok"] for r in rr),"KV020":sum(r["results"]["KV020"]["ok"] for r in rr),"rescued":sum(r["transition"]=="RESCUED" for r in rr),"damaged":sum(r["transition"]=="DAMAGED" for r in rr)}
    S=dict(test="459",heldout=True,seed=SEED,n=N_ENT,frozen_lambda=LAM,arms=["B-SOLO","RAW A+B","FROZEN KV020 A+B"],SOLO=f"{solo_ok}/{n}",RAW=f"{raw_ok}/{n}",KV020=f"{guard_ok}/{n}",rescued=rescued,damaged=damaged,keep_ok=keep_ok,keep_fail=keep_fail,net_benefit=net,raw_success_retention=retention,retention_wilson95=[rlo,rhi],raw_failure_rescue_rate=rescue_rate,rescue_wilson95=[qlo,qhi],mcnemar_exact_p=mcnemar_p,by_family=byfam,generations=GEN,frozen_weights_ok=frozen,minutes=round((time.time()-t0)/60,1),verdict=verdict)
    path="/content/test459_results.json" if os.path.isdir("/content") else "test459_results.json";json.dump(dict(summary=S,rows=rows,errors=errors),open(path,"w"),indent=1,ensure_ascii=False)
    print("="*116);print(f"HELD-OUT N={n} | lambda FROZEN={LAM:.2f} | seed={SEED}")
    print(f"SOLO {solo_ok}/{n} | RAW {raw_ok}/{n} | KV020 {guard_ok}/{n}")
    print(f"RAW→KV020: RESCUED={rescued} DAMAGED={damaged} KEEP_OK={keep_ok} KEEP_FAIL={keep_fail} | NET={net:+d}")
    print(f"RAW-success retention={retention:.3f}  Wilson95=[{rlo:.3f},{rhi:.3f}] | RAW-failure rescue={rescue_rate:.3f}  Wilson95=[{qlo:.3f},{qhi:.3f}]")
    print(f"McNemar exact p={mcnemar_p:.6f} | F1={byfam['F1']} | F2={byfam['F2']}")
    print(f"generations={GEN} | frozen_weights_ok={frozen} | minutes={S['minutes']}")
    print("="*116);print(f"TEST459 VERDICT: {verdict}  results={path}")
    print("Scope: first frozen held-out test of KV020 selected on TEST458; no lambda/layer/entity tuning permitted from this panel.")
if os.environ.get("T459_CPU_ONLY")!="1":run_gpu()



