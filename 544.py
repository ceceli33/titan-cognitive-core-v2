# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST544
# VARAN 1 — COMPLEMENTARITY X-RAY
# STATIC · QUERY-Q↔WRITE-K · QUERY-GATE↔WRITE-UP · QUERY-J↔WRITE-OV
# 2×2 DOUBLE-DISSOCIATION · SUBJECT / RELATION / ROLE / TIME
# RETRIEVAL YOK · SEÇİM YOK · EŞİK YOK · ROUTER YOK · EĞİTİM YOK
import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,math
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="544";SEED=544544;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3";PARENT543="e759d757db399443d9f6f273e070a5fe28306a0af24dc528b46ef66736c14ba9"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;NPROBE=8;JIDX=list(range(8))
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
print("="*156);print("TEST544 — AKBASCORE MAM · VARAN 1 · COMPLEMENTARITY X-RAY");print("STATIC · Q↔K · GATE↔UP · J↔OV · 2×2 DOUBLE-DISSOCIATION · ROLE/TIME/SUBJECT · RETRIEVAL YOK");print("="*156);T0=time.perf_counter()
print("\n[1/9] Donmuş Mistral...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH;REP=QH//KVH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L H={H} QH={QH} KVH={KVH} HD={HD} | trainable=0")
FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[16].mlp.down_proj.weight,layers[24].self_attn.o_proj.weight,layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()
BASE=[("Zorvan","Melket","Dravel","Oakhaven"),("Kelvar","Nareth","Solven","Branik"),("Tarev","Luneth","Varos","Cedran"),("Belnor","Arven","Dorel","Kesmar"),("Ravik","Selora","Terven","Maldor"),("Nemor","Calven","Istral","Pareth"),("Darsen","Velora","Keldin","Orvek"),("Feron","Talven","Merith","Sovran"),("Larev","Nerith","Calder","Veyron"),("Torven","Elsar","Marvek","Dorin"),("Selnor","Kareth","Valen","Ordan"),("Mirev","Taldor","Neris","Kelmar"),("Varen","Solith","Deran","Malvek"),("Kelor","Ardin","Velmar","Toren"),("Narev","Belith","Corven","Sareth"),("Dervan","Mirel","Talvek","Orsen"),("Calnor","Verith","Naldor","Seren"),("Parel","Dorven","Kelith","Maros"),("Sorven","Tarell","Vindor","Nelmar"),("Barel","Corith","Laven","Derik"),("Ralen","Mervor","Talith","Kesven"),("Norel","Valdor","Serith","Calven"),("Tervan","Orel","Mardin","Velos"),("Karev","Solen","Dareth","Mirven")]
W=[]
for s,cap,big,old in BASE:
    W.append({"s":s,"cap":cap,"big":big,"old":old,"wc":f"The current capital of {s} is {cap}.","wl":f"The largest city of {s} is {big}.","wo":f"The former capital of {s} was {old}.","role":f"The capital of {cap} is {s}.","qc":f"Which city currently serves as the seat of government for the state of {s}?","ql":f"Which urban center has the largest population in {s}?","qt":f"Which city used to serve as the capital of {s}?","qc2":f"What city functions as {s}'s present administrative seat?","qtr":f"{s} devletinin günümüzdeki yönetim merkezi hangi şehirdir?"})
LOCK={"test":TEST,"parent543":PARENT543,"model":MODEL_ID,"seed":SEED,"purpose":"COMPLEMENTARITY_XRAY","retrieval":False,"selection":False,"winner":False,"threshold":False,"router":False,"training":False,"gold_in_reader":False,"single_head_address":False,"single_layer_address":False,"channels":["STATIC","QK","GATE_UP","J_OV"],"j_method":"FIXED_RADEMACHER_LOGIT_VJP_SKETCH_FROM_GRAD_ENABLED_INPUT_EMBEDDING","nprobe":NPROBE,"j_worlds":JIDX,"fixes":["KV_EXPAND_HEAD_AXIS","TRUE_MLP_INPUT_HOOK","OV_HEAD_BLOCKS","J_INPUT_EMBEDDING_GRAD_LEAF","J_LAYER_INPUT_RESIDUAL","FIXED_SHARED_RADEMACHER_PROBES","J_SIGNED_ABS_RMS"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest();print("      LOCK SHA:",LOCK_SHA)
def ids(x):return tok(x,add_special_tokens=True,return_tensors="pt").input_ids.to(DEVICE)
def expand_kv(x):
    if x.shape[-2]!=KVH:raise RuntimeError(f"KV head uyuşmazlığı: {tuple(x.shape)}")
    return x.repeat_interleave(REP,dim=-2)
def cos(a,b,dim=-1):return F.cosine_similarity(a,b,dim=dim)
def arrmean(x):return np.mean(np.stack(x),axis=0)
@torch.inference_mode()
def trace(text):
    mlpin=[None]*NL;hooks=[]
    for li,L in enumerate(layers):
        def hk(m,a,li=li):mlpin[li]=a[0][0,-1].detach()
        hooks.append(L.mlp.register_forward_pre_hook(hk))
    o=model(input_ids=ids(text),use_cache=False,output_hidden_states=True,return_dict=True)
    for h in hooks:h.remove()
    if any(x is None for x in mlpin):raise RuntimeError("MLP input hook eksik.")
    R=[];Q=[];K=[];V=[];G=[];U=[];OV=[]
    for l,L in enumerate(layers):
        hin=o.hidden_states[l][0,-1];hout=o.hidden_states[l+1][0,-1]
        hn=L.input_layernorm(hin.to(device=L.input_layernorm.weight.device,dtype=L.input_layernorm.weight.dtype))
        q=L.self_attn.q_proj(hn).view(QH,HD);k=L.self_attn.k_proj(hn).view(KVH,HD);v=L.self_attn.v_proj(hn).view(KVH,HD)
        mi=mlpin[l].to(device=L.mlp.gate_proj.weight.device,dtype=L.mlp.gate_proj.weight.dtype);g=F.silu(L.mlp.gate_proj(mi));u=L.mlp.up_proj(mi)
        vv=expand_kv(v)
        # Linear o_proj: output[o]=sum_h,d W[o,h,d]*v[h,d]; her Q-head'in residual payload katkısı.
        ow=L.self_attn.o_proj.weight.reshape(H,QH,HD).permute(1,0,2)
        ov=torch.einsum("hd,hod->ho",vv,ow)
        R.append(hout.float().cpu());Q.append(q.float().cpu());K.append(k.float().cpu());V.append(v.float().cpu());G.append(g.float().cpu());U.append(u.float().cpu());OV.append(ov.float().cpu())
    del o,mlpin
    return {k:torch.stack(v) for k,v in {"R":R,"Q":Q,"K":K,"V":V,"G":G,"U":U,"OV":OV}.items()}
# Bütün sorgularda AYNI sabit output-space probları. Gold/token/cevap bilgisi kullanılmaz.
VZ=model.config.vocab_size
pg=torch.Generator(device="cpu");pg.manual_seed(SEED)
PROBES=torch.randint(0,2,(NPROBE,VZ),generator=pg,dtype=torch.int8).float().mul_(2).sub_(1).div_(math.sqrt(VZ))
def jtrace(text):
    # Ağırlıklar frozen kalır. Hesap grafiğini yalnız input embedding leaf'i başlatır.
    x=ids(text)
    with torch.no_grad():emb=model.model.embed_tokens(x).detach()
    emb=emb.requires_grad_(True)
    saved=[None]*NL;hooks=[]
    for li,L in enumerate(layers):
        def pre(m,a,li=li):
            h=a[0]
            if not h.requires_grad:raise RuntimeError(f"L{li:02d} residual grad grafiğinden kopuk.")
            saved[li]=h
        hooks.append(L.register_forward_pre_hook(pre))
    try:
        with torch.enable_grad():
            out=model(inputs_embeds=emb,use_cache=False,return_dict=True)
            logits=out.logits[0,-1].float()
        if not logits.requires_grad or logits.grad_fn is None:raise RuntimeError("J forward logits grad grafiğine bağlı değil.")
        if any(s is None for s in saved):raise RuntimeError("J layer-input hook eksik.")
        P=[]
        for p in range(NPROBE):
            r=PROBES[p].to(device=logits.device,dtype=logits.dtype)
            z=torch.dot(logits,r)
            gr=torch.autograd.grad(z,saved,retain_graph=p<NPROBE-1,create_graph=False,allow_unused=False)
            P.append(torch.stack([g[0,-1].detach().float().cpu() for g in gr]))
        ans=torch.stack(P)
        if ans.shape!=(NPROBE,NL,H):raise RuntimeError(f"J shape hatası: {tuple(ans.shape)}")
    finally:
        for h in hooks:h.remove()
    del out,logits,emb,saved;gc.collect();torch.cuda.empty_cache()
    return ans
print("\n[2/9] WRITE/QUERY izleri...")
DATA=[]
for i,w in enumerate(W):
    print(f"      {i+1:02d}/{len(W)}")
    DATA.append({k:trace(w[k.lower()]) for k in ["WC","WL","WO","ROLE","QC","QL","QT","QC2","QTR"]})
gc.collect();torch.cuda.empty_cache()
def pair(q,w):
    st=cos(q["R"],w["R"],-1)
    ek=expand_kv(w["K"])
    if q["Q"].shape!=ek.shape:raise RuntimeError(f"QK shape: {tuple(q['Q'].shape)} != {tuple(ek.shape)}")
    qk=cos(q["Q"],ek,-1)
    if q["G"].shape!=w["U"].shape:raise RuntimeError(f"Gate/Up shape: {tuple(q['G'].shape)} != {tuple(w['U'].shape)}")
    gu=q["G"]*w["U"];gu_signed=gu.mean(-1);gu_surv=gu.norm(dim=-1)/(w["U"].norm(dim=-1)+1e-12)
    return st.numpy(),qk.numpy(),gu_signed.numpy(),gu_surv.numpy()
print("\n[3/9] STATIC / Q↔K / GATE↔UP 2×2...")
CH={k:[] for k in ["ST_I","QK_I","GUS_I","GUN_I","ST_ROLE","QK_ROLE","GUN_ROLE","ST_TIME","QK_TIME","GUN_TIME","ST_PARA","QK_PARA","GUN_PARA","ST_TR","QK_TR","GUN_TR"]}
for d in DATA:
    cc=pair(d["QC"],d["WC"]);cl=pair(d["QC"],d["WL"]);lc=pair(d["QL"],d["WC"]);ll=pair(d["QL"],d["WL"])
    for n,j in [("ST_I",0),("QK_I",1),("GUS_I",2),("GUN_I",3)]:CH[n].append(cc[j]+ll[j]-cl[j]-lc[j])
    cr=pair(d["QC"],d["ROLE"]);ct=pair(d["QC"],d["WO"]);cp=pair(d["QC2"],d["WC"]);ctr=pair(d["QTR"],d["WC"])
    for pre,j in [("ST",0),("QK",1),("GUN",3)]:
        CH[pre+"_ROLE"].append(cc[j]-cr[j]);CH[pre+"_TIME"].append(cc[j]-ct[j]);CH[pre+"_PARA"].append(cp[j]);CH[pre+"_TR"].append(ctr[j])
STI=arrmean(CH["ST_I"]);QKI=arrmean(CH["QK_I"]);GUSI=arrmean(CH["GUS_I"]);GUNI=arrmean(CH["GUN_I"])
print(f"      STATIC_I       global={STI.mean():+.6f} min={STI.min():+.6f} max={STI.max():+.6f}")
print(f"      QK_I           global={QKI.mean():+.6f} min={QKI.min():+.6f} max={QKI.max():+.6f}")
print(f"      GATEUP_SIGNED  global={GUSI.mean():+.6f} min={GUSI.min():+.6f} max={GUSI.max():+.6f}")
print(f"      GATEUP_SURV_I  global={GUNI.mean():+.6f} min={GUNI.min():+.6f} max={GUNI.max():+.6f}")
print("\n[4/9] SUBJECT / ROLE / TIME / PARAPHRASE / TÜRKÇE...")
SUB={k:[] for k in ["ST","QK","GUN"]}
for i,d in enumerate(DATA):
    good=pair(d["QC"],d["WC"]);bad=pair(d["QC"],DATA[(i+1)%len(DATA)]["WC"])
    for n,j in [("ST",0),("QK",1),("GUN",3)]:SUB[n].append(good[j]-bad[j])
for n in ["ST","QK","GUN"]:
    print(f"      {n:4s} SUBJECT={arrmean(SUB[n]).mean():+.6f} ROLE={arrmean(CH[n+'_ROLE']).mean():+.6f} TIME={arrmean(CH[n+'_TIME']).mean():+.6f} PARA={arrmean(CH[n+'_PARA']).mean():+.6f} TR={arrmean(CH[n+'_TR']).mean():+.6f}")
print("\n[5/9] Layer/head profilleri — seçim YOK...")
for l in range(NL):print(f"      L{l:02d} STATIC_I={STI[l]:+.5f} QK_I={QKI[l].mean():+.5f} QK_std={QKI[l].std():.5f} GU_SIGN={GUSI[l]:+.5f} GU_SURV={GUNI[l]:+.5f}")
print(f"\n[6/9] J↔OV VJP X-ray · sabit worlds={JIDX} · probes={NPROBE}...")
JDATA={}
for z,i in enumerate(JIDX):
    print(f"      {z+1:02d}/{len(JIDX)} world={i:02d} QC/QL/QC2/QTR")
    JDATA[i]={"QC":jtrace(W[i]["qc"]),"QL":jtrace(W[i]["ql"]),"QC2":jtrace(W[i]["qc2"]),"QTR":jtrace(W[i]["qtr"])}
def jp(J,P):
    if J.shape!=(NPROBE,NL,H) or P.shape!=(NL,QH,H):raise RuntimeError(f"JP shape: J={tuple(J.shape)} P={tuple(P.shape)}")
    return torch.einsum("pld,lhd->plh",J,P)/math.sqrt(H)
JI=[];JROLE=[];JTIME=[];JSUB=[];JPARA=[];JTR=[]
for i in JIDX:
    d=DATA[i];j=JDATA[i]
    cc=jp(j["QC"],d["WC"]["OV"]);cl=jp(j["QC"],d["WL"]["OV"]);lc=jp(j["QL"],d["WC"]["OV"]);ll=jp(j["QL"],d["WL"]["OV"])
    JI.append((cc+ll-cl-lc).numpy());JROLE.append((cc-jp(j["QC"],d["ROLE"]["OV"])).numpy());JTIME.append((cc-jp(j["QC"],d["WO"]["OV"])).numpy());JSUB.append((cc-jp(j["QC"],DATA[(i+1)%len(DATA)]["WC"]["OV"])).numpy());JPARA.append(jp(j["QC2"],d["WC"]["OV"]).numpy());JTR.append(jp(j["QTR"],d["WC"]["OV"]).numpy())
JA=np.stack(JI);JM=JA.mean(axis=(0,1));JABS=np.abs(JA).mean(axis=(0,1));JRMS=np.sqrt(np.mean(JA.astype(np.float64)**2,axis=(0,1)))
print(f"      J_OV_I signed={JM.mean():+.8f} abs={JABS.mean():.8f} rms={JRMS.mean():.8f}")
print(f"      J_OV_ROLE abs={np.abs(np.stack(JROLE)).mean():.8f} TIME abs={np.abs(np.stack(JTIME)).mean():.8f} SUBJECT abs={np.abs(np.stack(JSUB)).mean():.8f}")
print("\n[7/9] J↔OV layer profili — tek layer/head seçimi YOK...")
for l in range(NL):print(f"      L{l:02d} J_I={JM[l].mean():+.8f} |J_I|={JABS[l].mean():.8f} RMS={JRMS[l].mean():.8f}")
print("\n[8/9] Kör özet — winner/retrieval kararı YOK...")
NUM={"STATIC_I":float(STI.mean()),"QK_I":float(QKI.mean()),"GATEUP_SIGNED_I":float(GUSI.mean()),"GATEUP_SURV_I":float(GUNI.mean()),"J_OV_I":float(JM.mean()),"J_OV_I_ABS":float(JABS.mean()),"J_OV_I_RMS":float(JRMS.mean()),"STATIC_SUBJECT":float(arrmean(SUB["ST"]).mean()),"QK_SUBJECT":float(arrmean(SUB["QK"]).mean()),"GU_SUBJECT":float(arrmean(SUB["GUN"]).mean()),"STATIC_ROLE":float(arrmean(CH["ST_ROLE"]).mean()),"QK_ROLE":float(arrmean(CH["QK_ROLE"]).mean()),"GU_ROLE":float(arrmean(CH["GUN_ROLE"]).mean()),"STATIC_TIME":float(arrmean(CH["ST_TIME"]).mean()),"QK_TIME":float(arrmean(CH["QK_TIME"]).mean()),"GU_TIME":float(arrmean(CH["GUN_TIME"]).mean()),"QK_PARAPHRASE":float(arrmean(CH["QK_PARA"]).mean()),"QK_TURKISH":float(arrmean(CH["QK_TR"]).mean()),"GU_PARAPHRASE":float(arrmean(CH["GUN_PARA"]).mean()),"GU_TURKISH":float(arrmean(CH["GUN_TR"]).mean()),"J_ROLE_ABS":float(np.abs(np.stack(JROLE)).mean()),"J_TIME_ABS":float(np.abs(np.stack(JTIME)).mean()),"J_SUBJECT_ABS":float(np.abs(np.stack(JSUB)).mean()),"J_PARA_ABS":float(np.abs(np.stack(JPARA)).mean()),"J_TURKISH_ABS":float(np.abs(np.stack(JTR)).mean())}
for k,v in NUM.items():print(f"      {k:20s}: {v:+.8f}")
print("\n[9/9] Sentinel / mühür...")
S1=sentinel();OK=S0==S1
SUMMARY={"test":TEST,"parent543":PARENT543,"lock_sha":LOCK_SHA,"weight_sentinel":OK,"retrieval":False,"selection":False,"training":False,"numbers":NUM,"verdict":"TEST544_COMPLEMENTARITY_XRAY_TAMAMLANDI"}
RESULT_SHA=hashlib.sha256(json.dumps(SUMMARY,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Weight sentinel              :","PASS" if OK else "FAIL")
print("      Trainable tensors            :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Retrieval / argmax / winner  : YOK")
print("      Threshold / router / scorer  : YOK")
print("      Memory-ID / token-ID address : YOK")
print("      Tek layer/head adresi        : YOK")
print("      Gold reader/Jacobian         : YOK")
print("      Forward intervention         : YOK")
print("      Training/LoRA/optimizer      : YOK")
print("\n"+"="*156);print("TEST544 SONUÇ — COMPLEMENTARITY X-RAY");print("="*156)
print("MODEL                         :",MODEL_ID)
print("DÜNYA                         :",len(W),"| J-VJP dünya:",len(JIDX),"| fixed probes:",NPROBE)
print("STATIC 2×2 I                 :",f"{NUM['STATIC_I']:+.8f}")
print("QUERY-Q ↔ WRITE-K 2×2 I      :",f"{NUM['QK_I']:+.8f}")
print("QUERY-GATE ↔ WRITE-UP 2×2 I  :",f"{NUM['GATEUP_SURV_I']:+.8f}")
print("QUERY-J ↔ WRITE-OV 2×2 I     :",f"{NUM['J_OV_I']:+.8f}","| abs",f"{NUM['J_OV_I_ABS']:.8f}","| rms",f"{NUM['J_OV_I_RMS']:.8f}")
print("WEIGHT SENTINEL               :","PASS" if OK else "FAIL")
print("TEST543 RESULT SHA            :",PARENT543)
print("TEST544 ÖN KİLİT SHA          :",LOCK_SHA)
print("TEST544 RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                         : TEST544_COMPLEMENTARITY_XRAY_TAMAMLANDI")
print("YORUM KURALI                  : Tek layer/head seçme; STATIC, 2×2, subject/role/time ve paraphrase/TR yapılarını birlikte değerlendir.")
print("="*156)
