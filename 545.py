# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST545
# VARAN 1 — TRANSFORMATION-RESPONSE TRAJECTORY X-RAY
# STATIC vs LOCAL RESPONSE · TRUE / NEAR / FAR / CROSS
# SABİT PROBE BANK · TEK LAYER/HEAD/VEKTÖR ADRESİ YOK · RETRIEVAL/ROUTER/EŞİK/EĞİTİM YOK
import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
TEST="545";SEED=545545;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT544="4e22925805401354061dcf00aaeeb92beaa1991fbb93cfab845408c4dd7b253b"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16;NPROBE=8
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
print("="*156);print("TEST545 — AKBASCORE MAM · VARAN 1 · TRANSFORMATION-RESPONSE TRAJECTORY X-RAY");print("STATIC vs LOCAL FULL-MODEL JVP RESPONSE · TRUE / NEAR / FAR / CROSS · RETRIEVAL YOK · SEÇİM YOK · EŞİK YOK");print("="*156);T0=time.perf_counter()
print("\n[1/8] Donmuş Mistral...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
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
    W.append({"s":s,"write":f"The current capital of {s} is {cap}.","near":f"The largest city of {s} is {big}.","temporal":f"The former capital of {s} was {old}.","role":f"The capital of {cap} is {s}.","query":f"Which city currently serves as the seat of government for the state of {s}?","para":f"What city functions as {s}'s present administrative seat?","tr":f"{s} devletinin günümüzdeki yönetim merkezi hangi şehirdir?"})
LOCK={"test":TEST,"parent544":PARENT544,"model":MODEL_ID,"seed":SEED,"purpose":"TRANSFORMATION_RESPONSE_TRAJECTORY_XRAY","probes":NPROBE,"probe_type":"FIXED_ORTHONORMAL_RADEMACHER","jvp_path":"FULL_MODEL_NATIVE_FORWARD_HOOK","retrieval":False,"selection":False,"argmax":False,"winner":False,"threshold":False,"router":False,"training":False,"gold_selection":False,"single_layer_address":False,"single_head_address":False,"single_vector_address":False,"parent_result_sha":PARENT544}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest();print("      LOCK SHA:",LOCK_SHA)
def ids(x):return tok(x,add_special_tokens=True,return_tensors="pt").input_ids.to(DEVICE)
def cos(a,b,dim=-1):return F.cosine_similarity(a,b,dim=dim)
pg=torch.Generator(device="cpu");pg.manual_seed(SEED)
P=torch.randint(0,2,(NPROBE,H),generator=pg,dtype=torch.int8).float().mul_(2).sub_(1)
P=torch.linalg.qr(P.T,mode="reduced").Q.T.contiguous()
if P.shape!=(NPROBE,H):raise RuntimeError(f"Probe shape hatası: {P.shape}")
@torch.no_grad()
def static_trace(text):
    o=model(input_ids=ids(text),use_cache=False,output_hidden_states=True,return_dict=True)
    r=torch.stack([o.hidden_states[l+1][0,-1].float().cpu() for l in range(NL)])
    del o
    if r.shape!=(NL,H):raise RuntimeError(f"Static shape hatası: {r.shape}")
    return r
# Modelin normal forward yolunu kullanır. RoPE/position_embeddings/mask/cache HF tarafından üretilir.
# Probe yalnız hedef layer girişinin final-token residual'ına eklenir.
# Hedef layer çıkışının probe'a göre JVP'si ölçülür; ağırlıklar frozen kalır.
def one_layer_response(input_ids,li,probe):
    eps=torch.zeros((1,1,H),device=DEVICE,dtype=DTYPE,requires_grad=True)
    p=probe.to(DEVICE,dtype=DTYPE).view(1,1,H)
    captured={}
    def pre_hook(module,args):
        x=args[0]
        add=torch.zeros_like(x)
        add[:,-1:,:]=eps*p
        return (x+add,)+args[1:]
    def out_hook(module,args,out):
        y=out[0] if isinstance(out,(tuple,list)) else out
        captured["y"]=y[:,-1,:]
    hp=layers[li].register_forward_pre_hook(pre_hook);ho=layers[li].register_forward_hook(out_hook)
    try:
        with torch.enable_grad():model(input_ids=input_ids,use_cache=False,return_dict=True)
        if "y" not in captured:raise RuntimeError(f"L{li:02d} output hook çalışmadı.")
        y=captured["y"]
        # eps [1,1,H] fakat perturbation eps⊙probe. Diagonal directional response:
        # d y_j / d eps_j, ardından probe işareti/magnitüdü zaten girişte uygulanmıştır.
        # Full Jacobian gerektirmeden içerikten bağımsız sabit yönün yerel response fingerprint'i.
        g=[]
        # Çıktıyı tek tek 4096 backward ile taramak yerine sabit output probe aynı P bankından alınır.
        # Bu bir adres değil; bütün testlerde değişmeyen X-ray projeksiyonudur.
        for op in P:
            q=op.to(DEVICE,dtype=y.dtype).view(1,H)
            s=(y*q).sum()
            ge=torch.autograd.grad(s,eps,retain_graph=True,create_graph=False)[0][0,0]
            g.append(ge.detach().float().cpu())
        ans=torch.stack(g).mean(0)
    finally:
        hp.remove();ho.remove()
    del eps,captured
    return ans
# P adet giriş probe × P adet sabit output probe yerine maliyeti sınırlamak için her giriş probe
# kendi eş indeksli output probe'u ile dual-pair edilir. Hiçbir probe veri/gold ile seçilmez.
def one_layer_dual_response(input_ids,li,probe_idx):
    pin=P[probe_idx].to(DEVICE,dtype=DTYPE)
    pout=P[probe_idx].to(DEVICE,dtype=DTYPE)
    alpha=torch.zeros((),device=DEVICE,dtype=torch.float32,requires_grad=True)
    captured={}
    def pre_hook(module,args):
        x=args[0]
        add=torch.zeros_like(x)
        add[:,-1,:]=alpha.to(x.dtype)*pin
        return (x+add,)+args[1:]
    def out_hook(module,args,out):
        y=out[0] if isinstance(out,(tuple,list)) else out
        captured["y"]=y[:,-1,:]
    hp=layers[li].register_forward_pre_hook(pre_hook);ho=layers[li].register_forward_hook(out_hook)
    try:
        with torch.enable_grad():model(input_ids=input_ids,use_cache=False,return_dict=True)
        if "y" not in captured:raise RuntimeError(f"L{li:02d} output hook çalışmadı.")
        # scalar directional derivative <pout, J pin>; ayrıca response yönü için ∂y/∂alpha gerekir.
        y=captured["y"]
        # Tek scalar alpha → tüm H output türevini torch.func.jvp olmadan 4096 backward gerektirir.
        # Bunun yerine native forward-mode JVP kullanmak için torch.func.jvp model wrapper kullanılır.
    finally:
        hp.remove();ho.remove()
    del alpha,captured
# Sağlam ve sürümden bağımsız çözüm: layer direct-call yerine full-model finite local directional derivative.
# Sabit merkezi fark; yalnız X-ray. Epsilon sabittir, hiçbir veri/gold ile ayarlanmaz.
# BF16 çözünürlüğü nedeniyle 1e-2 kullanılır; retrieval/scorer parametresi değildir.
FD_EPS=1e-2
@torch.no_grad()
def full_model_layer_response(input_ids,li,probe):
    p=probe.to(DEVICE,dtype=DTYPE)
    def run(sign):
        cap={}
        def pre_hook(module,args):
            x=args[0];add=torch.zeros_like(x);add[:,-1,:]=sign*FD_EPS*p
            return (x+add,)+args[1:]
        def out_hook(module,args,out):
            y=out[0] if isinstance(out,(tuple,list)) else out
            cap["y"]=y[0,-1].detach().float().cpu()
        hp=layers[li].register_forward_pre_hook(pre_hook);ho=layers[li].register_forward_hook(out_hook)
        try:model(input_ids=input_ids,use_cache=False,return_dict=True)
        finally:hp.remove();ho.remove()
        if "y" not in cap:raise RuntimeError(f"L{li:02d} response hook eksik.")
        return cap["y"]
    yp=run(+1.0);ym=run(-1.0)
    return (yp-ym)/(2.0*FD_EPS)
def response_trace(text):
    x=ids(text);OUT=[]
    for li in range(NL):
        row=[]
        for pi in range(NPROBE):row.append(full_model_layer_response(x,li,P[pi]))
        OUT.append(torch.stack(row))
    ans=torch.stack(OUT)
    if ans.shape!=(NL,NPROBE,H):raise RuntimeError(f"Response shape hatası: {ans.shape}")
    del x;gc.collect();torch.cuda.empty_cache()
    return ans
def rsp_sim(a,b):return cos(a,b,-1).mean(-1)
def traj_sim(a,b):
    da=a[1:]-a[:-1];db=b[1:]-b[:-1]
    return cos(da,db,-1).mean(-1)
print("\n[2/8] STATIC izleri...")
STATIC=[]
for i,w in enumerate(W):
    print(f"      {i+1:02d}/{len(W)}")
    STATIC.append({"W":static_trace(w["write"]),"N":static_trace(w["near"]),"Q":static_trace(w["query"]),"QP":static_trace(w["para"]),"QT":static_trace(w["tr"])})
print("\n[3/8] LOCAL RESPONSE izleri · full-model native forward · sabit probe bank...")
RESP=[]
for i,w in enumerate(W):
    print(f"      {i+1:02d}/{len(W)} WRITE/NEAR/QUERY/PARA/TR")
    RESP.append({"W":response_trace(w["write"]),"N":response_trace(w["near"]),"Q":response_trace(w["query"]),"QP":response_trace(w["para"]),"QT":response_trace(w["tr"])})
print("\n[4/8] TRUE / NEAR / FAR / CROSS karşılaştırması...")
ST={k:[] for k in ["TRUE","NEAR","FAR","CROSS","PARA","TR"]};RS={k:[] for k in ST};TJ={k:[] for k in ST}
for i in range(len(W)):
    j=(i+1)%len(W);k=(i+7)%len(W)
    ST["TRUE"].append(cos(STATIC[i]["Q"],STATIC[i]["W"],-1).numpy());ST["NEAR"].append(cos(STATIC[i]["Q"],STATIC[i]["N"],-1).numpy());ST["CROSS"].append(cos(STATIC[i]["Q"],STATIC[j]["W"],-1).numpy());ST["FAR"].append(cos(STATIC[i]["Q"],STATIC[k]["N"],-1).numpy());ST["PARA"].append(cos(STATIC[i]["QP"],STATIC[i]["W"],-1).numpy());ST["TR"].append(cos(STATIC[i]["QT"],STATIC[i]["W"],-1).numpy())
    RS["TRUE"].append(rsp_sim(RESP[i]["Q"],RESP[i]["W"]).numpy());RS["NEAR"].append(rsp_sim(RESP[i]["Q"],RESP[i]["N"]).numpy());RS["CROSS"].append(rsp_sim(RESP[i]["Q"],RESP[j]["W"]).numpy());RS["FAR"].append(rsp_sim(RESP[i]["Q"],RESP[k]["N"]).numpy());RS["PARA"].append(rsp_sim(RESP[i]["QP"],RESP[i]["W"]).numpy());RS["TR"].append(rsp_sim(RESP[i]["QT"],RESP[i]["W"]).numpy())
    TJ["TRUE"].append(traj_sim(RESP[i]["Q"],RESP[i]["W"]).numpy());TJ["NEAR"].append(traj_sim(RESP[i]["Q"],RESP[i]["N"]).numpy());TJ["CROSS"].append(traj_sim(RESP[i]["Q"],RESP[j]["W"]).numpy());TJ["FAR"].append(traj_sim(RESP[i]["Q"],RESP[k]["N"]).numpy());TJ["PARA"].append(traj_sim(RESP[i]["QP"],RESP[i]["W"]).numpy());TJ["TR"].append(traj_sim(RESP[i]["QT"],RESP[i]["W"]).numpy())
def am(d,k):return np.mean(np.stack(d[k]),axis=0)
for name,d in [("STATIC",ST),("RESPONSE",RS),("TRAJECTORY",TJ)]:
    print(f"\n      {name}")
    for k in ["TRUE","NEAR","CROSS","FAR","PARA","TR"]:print(f"      {k:6s} {am(d,k).mean():+.6f}")
    print(f"      TRUE-NEAR  {(am(d,'TRUE')-am(d,'NEAR')).mean():+.6f}")
    print(f"      TRUE-CROSS {(am(d,'TRUE')-am(d,'CROSS')).mean():+.6f}")
    print(f"      TRUE-FAR   {(am(d,'TRUE')-am(d,'FAR')).mean():+.6f}")
print("\n[5/8] Kritik tersine-dönüş testi · STATIC NEAR>TRUE vakaları...")
PER=[]
for i in range(len(W)):
    sd=float(np.mean(ST["TRUE"][i])-np.mean(ST["NEAR"][i]));rd=float(np.mean(RS["TRUE"][i])-np.mean(RS["NEAR"][i]));td=float(np.mean(TJ["TRUE"][i])-np.mean(TJ["NEAR"][i]))
    trap=sd<0;rf=trap and rd>0;tf=trap and td>0;PER.append((i,W[i]["s"],sd,rd,td,trap,rf,tf))
    print(f"      {i:02d} {W[i]['s']:8s} STATICΔ={sd:+.5f} RESPONSEΔ={rd:+.5f} TRAJΔ={td:+.5f} trap={int(trap)} Rflip={int(rf)} Tflip={int(tf)}")
NTRAP=sum(int(x[5]) for x in PER);RFLIP=sum(int(x[6]) for x in PER);TFLIP=sum(int(x[7]) for x in PER)
print(f"\n      STATIC NEAR>TRUE traps : {NTRAP}/{len(W)}")
print(f"      RESPONSE reversals     : {RFLIP}/{NTRAP}" if NTRAP else "      RESPONSE reversals     : 0/0")
print(f"      TRAJECTORY reversals   : {TFLIP}/{NTRAP}" if NTRAP else "      TRAJECTORY reversals   : 0/0")
print("\n[6/8] Katman profilleri — hiçbir layer seçilmiyor...")
ST_D=am(ST,"TRUE")-am(ST,"NEAR");RS_D=am(RS,"TRUE")-am(RS,"NEAR");TJ_D=am(TJ,"TRUE")-am(TJ,"NEAR")
for l in range(NL):
    tv=f"{TJ_D[l]:+.6f}" if l<NL-1 else "n/a"
    print(f"      L{l:02d} STATIC TRUE-NEAR={ST_D[l]:+.6f} RESPONSE TRUE-NEAR={RS_D[l]:+.6f} TRAJECTORY={tv}")
print("\n[7/8] Dağıtık bütün-hat özeti...")
NUM={"STATIC_TRUE":float(am(ST,"TRUE").mean()),"STATIC_NEAR":float(am(ST,"NEAR").mean()),"STATIC_TRUE_NEAR":float(ST_D.mean()),"STATIC_TRUE_CROSS":float((am(ST,"TRUE")-am(ST,"CROSS")).mean()),"STATIC_TRUE_FAR":float((am(ST,"TRUE")-am(ST,"FAR")).mean()),"RESPONSE_TRUE":float(am(RS,"TRUE").mean()),"RESPONSE_NEAR":float(am(RS,"NEAR").mean()),"RESPONSE_TRUE_NEAR":float(RS_D.mean()),"RESPONSE_TRUE_CROSS":float((am(RS,"TRUE")-am(RS,"CROSS")).mean()),"RESPONSE_TRUE_FAR":float((am(RS,"TRUE")-am(RS,"FAR")).mean()),"TRAJECTORY_TRUE":float(am(TJ,"TRUE").mean()),"TRAJECTORY_NEAR":float(am(TJ,"NEAR").mean()),"TRAJECTORY_TRUE_NEAR":float(TJ_D.mean()),"TRAJECTORY_TRUE_CROSS":float((am(TJ,"TRUE")-am(TJ,"CROSS")).mean()),"TRAJECTORY_TRUE_FAR":float((am(TJ,"TRUE")-am(TJ,"FAR")).mean()),"RESPONSE_PARA":float(am(RS,"PARA").mean()),"RESPONSE_TURKISH":float(am(RS,"TR").mean()),"TRAJECTORY_PARA":float(am(TJ,"PARA").mean()),"TRAJECTORY_TURKISH":float(am(TJ,"TR").mean()),"STATIC_TRAPS":int(NTRAP),"RESPONSE_REVERSALS":int(RFLIP),"TRAJECTORY_REVERSALS":int(TFLIP)}
for k,v in NUM.items():print(f"      {k:24s}: {v:+.8f}" if isinstance(v,float) else f"      {k:24s}: {v}")
print("\n[8/8] Sentinel / mühür...")
S1=sentinel();OK=S0==S1
SUMMARY={"test":TEST,"parent544":PARENT544,"lock_sha":LOCK_SHA,"weight_sentinel":OK,"retrieval":False,"selection":False,"training":False,"numbers":NUM,"verdict":"TEST545_TRANSFORMATION_RESPONSE_XRAY_TAMAMLANDI"}
RESULT_SHA=hashlib.sha256(json.dumps(SUMMARY,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      Weight sentinel              :","PASS" if OK else "FAIL")
print("      Trainable tensors            :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Retrieval / argmax / winner  : YOK")
print("      Threshold / router / scorer  : YOK")
print("      Memory-ID / token-ID address : YOK")
print("      Tek layer/head/probe adresi  : YOK")
print("      Gold selection               : YOK")
print("      Training/LoRA/optimizer      : YOK")
print("      Response                     : SABİT MERKEZİ FARK X-RAY")
print("\n"+"="*156);print("TEST545 SONUÇ — TRANSFORMATION-RESPONSE TRAJECTORY X-RAY");print("="*156)
print("MODEL                         :",MODEL_ID)
print("DÜNYA                         :",len(W),"| fixed probes:",NPROBE)
print("STATIC TRUE-NEAR             :",f"{NUM['STATIC_TRUE_NEAR']:+.8f}")
print("RESPONSE TRUE-NEAR           :",f"{NUM['RESPONSE_TRUE_NEAR']:+.8f}")
print("TRAJECTORY TRUE-NEAR         :",f"{NUM['TRAJECTORY_TRUE_NEAR']:+.8f}")
print("STATIC NEAR>TRUE TRAPS       :",NTRAP)
print("RESPONSE REVERSALS           :",f"{RFLIP}/{NTRAP}" if NTRAP else "0/0")
print("TRAJECTORY REVERSALS         :",f"{TFLIP}/{NTRAP}" if NTRAP else "0/0")
print("WEIGHT SENTINEL              :","PASS" if OK else "FAIL")
print("TEST544 RESULT SHA           :",PARENT544)
print("TEST545 ÖN KİLİT SHA         :",LOCK_SHA)
print("TEST545 RESULT SHA           :",RESULT_SHA)
print("TOPLAM SÜRE                  :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                        : TEST545_TRANSFORMATION_RESPONSE_XRAY_TAMAMLANDI")
print("YORUM KURALI                 : Kritik sinyal STATIC NEAR>TRUE vakalarında RESPONSE/TRAJECTORY'nin TRUE>NEAR yönüne dönmesidir; tek layer/head/probe seçilmez.")
print("="*156)
