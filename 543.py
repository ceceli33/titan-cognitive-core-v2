# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST543
# VARAN 1 — SEMANTIC SPECTRUM / ENIGMA X-RAY
#
# AMAÇ:
# Bir bilgi WRITE sırasında Transformer içinde oluşan çok boyutlu katman-akışının
# dağıtık modal/spektral yapısını taşır mı?
# Aynı bilgi tamamen farklı yüzey diliyle RECALL edildiğinde bu yapı yeniden belirir mi?
#
# "FREKANS" = fiziksel Hz değildir.
# Burada frekans/spektrum, Transformer'ın katman boyunca değişen yüksek boyutlu
# aktivasyon yörüngesinin modal yapısı için deneysel terimdir.
#
# KESİN YASAK:
# - Memory ID / cartridge ID / dosya numarası
# - Harf/rakam anahtarı
# - Token-ID adresi
# - Tek layer/head adresi
# - Tek vektör key
# - Max-cos retrieval
# - Router
# - Threshold/lambda
# - Length correction
# - Fitted scorer
# - Training/LoRA/optimizer
# - Gold ile seçim
# - "En iyi frekans" seçip adres yapmak
#
# TEST543 RETRIEVAL TESTİ DEĞİLDİR.
# Spektrum yalnız X-RAY nesnesidir.
# Hiçbir modal bileşen seçim/routing amacıyla kullanılmaz.
#
# TEST542 PARENT:
# RESULT SHA = 6e14e03cda7a28f8a49f7c0cdbd94719df020e4390167f000dfb457bae0c294c
#
# TEMEL FİKİR:
# WRITE  -> Transformer flow -> dağıtık modal spektrum
# RECALL -> Transformer flow -> dağıtık modal spektrum
#
# Soru:
# Ham metin/aktivasyon farklı olsa bile aynı bilgi gerektiğinde modal yapının
# bütününde özgül bir ortaklık yeniden ortaya çıkıyor mu?
#
# Spektral dönüşüm:
# 32 katman boyunca oluşan trajectory önce layer-axis üzerinde DC'den arındırılır.
# Ardından ortonormal DCT-II ile modal koordinatlara çevrilir.
# DCT burada öğrenilmiş değildir, semantik label kullanmaz, memory ID üretmez.
# Tek mod kullanılmaz. Tam modal dağılım korunur.
#
# Ayrıca residual yanında Q/K/V'nin bütün-head yapısı da ayrı ayrı incelenir.
# ==================================================================================================

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

TEST="543"
SEED=543543
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT542="6e14e03cda7a28f8a49f7c0cdbd94719df020e4390167f000dfb457bae0c294c"
DEVICE=torch.device("cuda")
DTYPE=torch.bfloat16
os.environ["TOKENIZERS_PARALLELISM"]="false"

if not torch.cuda.is_available(): raise RuntimeError("CUDA gerekli.")
torch.set_grad_enabled(False)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*156)
print("TEST543 — AKBASCORE MAM · VARAN 1 · SEMANTIC SPECTRUM / ENIGMA X-RAY")
print("WRITE → RELATE → RECALL · DAĞITIK MODAL HARİTA · TEK FREKANS YOK · ADRES YOK · ROUTER YOK · RETRIEVAL YOK")
print("="*156)
T0=time.perf_counter()

print("\n[1/12] Donmuş Mistral yükleniyor...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters(): p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
NL=len(layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128): raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")
print(f"      Model      : {MODEL_ID}")
print(f"      GPU        : {torch.cuda.get_device_name(0)}")
print(f"      Mimari     : {NL}L · H={H} · QH={QH} · KVH={KVH} · HD={HD}")
print("      Frozen     : EVET")
print("      Trainable  :",sum(int(p.requires_grad) for p in model.parameters()))

FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[16].mlp.down_proj.weight,layers[24].self_attn.o_proj.weight,layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15
            h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()

# Aynı bilgi farklı yüzey diliyle soruluyor.
# Deney dünyalarının Python sırası yalnız post-hoc değerlendirme içindir.
# Modele memory ID/cartridge ID olarak verilmez.
WORLDS=[
{
"write":"A woman named Elara lives in a stone cottage beside a pine-covered hill. Her bed is made from dark walnut and stands beneath the eastern window.",
"relate":"Much later, Elara placed a small reading chair beside the eastern window, close to the furniture she uses when she sleeps.",
"recall":[
"When Elara becomes tired at night, what piece of furniture does she lie down on?",
"What object in the cottage supports Elara while she is asleep?",
"After a long day, which furnishing would Elara use for sleeping?",
"If Elara says she is going to turn in for the night, what object is relevant?"
],
"near":[
"What did Elara later place beside the eastern window?",
"Which object near Elara's sleeping furniture is intended for sitting?"
],
"far":[
"What causes ocean tides?",
"Why does iron rust in moist air?"
]
},
{
"write":"Kalen keeps a dark touring bicycle in a wooden shelter behind his house. Every morning he uses it for the first part of his journey toward the railway station.",
"relate":"Later, Kalen fixed a weatherproof tool pouch beneath the seat so that repair equipment could travel with him.",
"recall":[
"What carries Kalen from home toward the station before he boards a train?",
"Which form of personal transport does Kalen ride during his morning journey?",
"If Kalen begins his usual trip to the railway, what does he travel on first?",
"What object stored behind the house is used by Kalen to get toward the station?"
],
"near":[
"What did Kalen later attach beneath the seat?",
"Where does Kalen keep repair equipment while travelling?"
],
"far":[
"Why does ice melt when heated?",
"How do roots absorb water from soil?"
]
},
{
"write":"Nara produces botanical illustrations at a long table in her workroom. She draws them on a pressure-sensitive digital drawing surface connected to her computer.",
"relate":"Later, Nara positioned a compact color calibration instrument beside the drawing equipment so printed colors would remain consistent.",
"recall":[
"What device does Nara actually draw on when she creates her plant illustrations?",
"Which piece of equipment receives Nara's hand movements while she makes digital artwork?",
"If Nara begins sketching a new botanical picture, what working surface is central to the task?",
"What device on the table is directly used to create Nara's digital drawings?"
],
"near":[
"What later device helps Nara keep printed colors consistent?",
"Which object beside the drawing equipment is used for calibration?"
],
"far":[
"Why do eclipses occur?",
"How does evaporation cool a surface?"
]
},
{
"write":"Soren prepares his morning coffee with a glass brewing vessel kept on the kitchen counter near the window.",
"relate":"Later, Soren placed a sealed ceramic container beside the brewer to protect freshly ground coffee from moisture.",
"recall":[
"What kitchen object does Soren use to make his coffee in the morning?",
"Which piece of equipment is involved when Soren actually brews his usual drink?",
"If Soren starts preparing coffee after waking up, what object does he use for brewing?",
"What item near the kitchen window performs the preparation rather than storing the ingredients?"
],
"near":[
"What later object protects Soren's ground coffee from moisture?",
"Which container beside the brewer is used for storage?"
],
"far":[
"How does a rainbow form?",
"Why do magnets attract some metals?"
]
},
{
"write":"Ilyra practises music in the quiet rear room of her apartment on a large acoustic piano with a dark wooden case.",
"relate":"Later, Ilyra placed a small electronic metronome above the keys to help keep her practice rhythm steady.",
"recall":[
"What instrument does Ilyra play when she practises at home?",
"Which object actually produces the music during Ilyra's practice sessions?",
"If Ilyra enters the rear room to practise, what instrument is she using?",
"What large object with keys is central to Ilyra's musical practice?"
],
"near":[
"What later device helps Ilyra maintain steady timing?",
"Which small electronic object did Ilyra place above the keys?"
],
"far":[
"Why do shadows change length during the day?",
"How does a battery produce electric current?"
]
},
{
"write":"Marek waters the vegetables behind his workshop with a flexible garden hose connected to an outdoor tap.",
"relate":"Later, Marek installed a programmable timer at the connection so watering could begin automatically before sunrise.",
"recall":[
"What carries water from Marek's outdoor tap toward the vegetable beds?",
"Which flexible object does Marek use when he waters the garden?",
"If Marek waters the plants by hand, what equipment delivers the water?",
"What item connected to the tap physically moves water toward Marek's vegetables?"
],
"near":[
"What later device allows Marek's watering to start automatically?",
"Which object at the connection controls the watering time?"
],
"far":[
"Why does thunder follow lightning?",
"How do fish obtain oxygen from water?"
]
},
{
"write":"Tavian records spoken interviews using a compact silver microphone mounted on a short stand beside his computer.",
"relate":"Later, Tavian added a circular acoustic screen behind the microphone to reduce reflections from the wall.",
"recall":[
"What device captures the voices during Tavian's interviews?",
"Which piece of equipment converts the interview speech into the recorded signal?",
"If Tavian begins recording a conversation, what object near his computer receives the sound?",
"What silver device on the short stand is central to Tavian's spoken recordings?"
],
"near":[
"What did Tavian later place behind the recording device?",
"Which later addition reduces sound reflections from the wall?"
],
"far":[
"Why does a balloon rise in warm air?",
"How do tree rings form?"
]
},
{
"write":"Orin observes distant birds from his balcony through a pair of black binoculars stored in a padded case.",
"relate":"Later, Orin attached a narrow carrying strap to the binoculars so he could keep his hands free while walking.",
"recall":[
"What optical equipment does Orin use to see birds that are far away?",
"Which object helps Orin enlarge his view of distant birds?",
"If Orin spots a bird far across the valley, what does he look through?",
"What black optical object from the padded case is used for distant observation?"
],
"near":[
"What later addition lets Orin carry the optical equipment hands-free?",
"Which narrow object did Orin attach for carrying?"
],
"far":[
"Why does bread become stale?",
"How does a thermometer measure temperature?"
]
}
]

LOCK={
"test":TEST,"parent542":PARENT542,"model":MODEL_ID,"seed":SEED,
"purpose":"SEMANTIC_SPECTRUM_ENIGMA_XRAY",
"physical_hz_claim":False,"retrieval":False,"human_memory_id":False,"token_id_address":False,
"single_head_address":False,"single_layer_address":False,"single_vector_key":False,
"router":False,"threshold":False,"fitted_score":False,"length_correction":False,"training":False,
"transform":"ORTHONORMAL_DCT_II_OVER_LAYER_AXIS_AFTER_DC_REMOVAL",
"selection":"NONE","gold_use":"POST_HOC_EVALUATION_ONLY",
"modal_policy":"FULL_DISTRIBUTION_NO_SINGLE_MODE_SELECTION"
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST543 kilit SHA:",LOCK_SHA)

# Ortonormal DCT-II matrisi. Öğrenilmiş/fitted değildir.
def dct_matrix(n):
    k=torch.arange(n,dtype=torch.float64).view(-1,1)
    j=torch.arange(n,dtype=torch.float64).view(1,-1)
    C=torch.cos(math.pi/n*(j+0.5)*k)
    C[0]*=math.sqrt(1/n)
    C[1:]*=math.sqrt(2/n)
    return C.float()
DCT=dct_matrix(NL)
DCT_MOTION=dct_matrix(NL-1)

def ids(text):
    return tok(text,add_special_tokens=True,return_tensors="pt").input_ids.to(DEVICE)

@torch.inference_mode()
def trace(text):
    x=ids(text)
    o=model(input_ids=x,use_cache=False,output_hidden_states=True,return_dict=True)
    R=[];Q=[];K=[];V=[]
    for l,layer in enumerate(layers):
        hout=o.hidden_states[l+1][0,-1].float().cpu()
        hin=o.hidden_states[l][0,-1]
        hn=layer.input_layernorm(hin.to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype))
        q=layer.self_attn.q_proj(hn).view(QH,HD).float().cpu()
        k=layer.self_attn.k_proj(hn).view(KVH,HD).float().cpu()
        v=layer.self_attn.v_proj(hn).view(KVH,HD).float().cpu()
        R.append(hout);Q.append(q);K.append(k);V.append(v)
    del o
    return {"R":torch.stack(R),"Q":torch.stack(Q),"K":torch.stack(K),"V":torch.stack(V)}

# Katman eksenindeki sabit/DC bileşeni kaldırılır.
# Böylece 542'de baskın çıkan genel Transformer iskeletinin etkisini azaltmayı deniyoruz.
def remove_dc(x):
    return x-x.mean(dim=0,keepdim=True)

def spectral_transform(x,C):
    shape=x.shape
    y=remove_dc(x).reshape(shape[0],-1)
    return C@y

def spectrum_energy(x,C):
    s=spectral_transform(x,C)
    e=(s*s).sum(dim=1)
    return e/(e.sum()+1e-12)

def spectral_shape_similarity(a,b,C):
    # Tam modal enerji dağılımının korelasyonu. Retrieval değildir.
    ea=spectrum_energy(a,C);eb=spectrum_energy(b,C)
    ea=ea-ea.mean();eb=eb-eb.mean()
    den=ea.norm()*eb.norm()
    return float(torch.dot(ea,eb)/den) if float(den)>0 else 0.0

def spectral_vector_alignment(a,b,C):
    # Her moddaki yüksek-boyutlu yönün karşılığı; bütün modlar raporlanır.
    A=spectral_transform(a,C);B=spectral_transform(b,C)
    A=torch.nn.functional.normalize(A,dim=1)
    B=torch.nn.functional.normalize(B,dim=1)
    return (A*B).sum(dim=1)

def full_modal_alignment(a,b,C):
    # Tek mod seçilmez. Tüm modların ortalama yönsel uyumu yalnız X-ray özetidir.
    return float(spectral_vector_alignment(a,b,C).mean())

def motion(x):
    return x[1:]-x[:-1]

def flatten_qkv(t,key):
    return t[key].reshape(NL,-1)

def stats(x):
    a=np.asarray(x,dtype=np.float64)
    return float(a.mean()),float(np.median(a)),float(a.std()),float(a.min()),float(a.max())

print("\n[2/12] WRITE / RELATE / RECALL / yakın-negatif / uzak-negatif izleri çıkarılıyor...")
DATA=[]
for wi,w in enumerate(WORLDS):
    print(f"      Dünya {wi+1}/{len(WORLDS)}")
    W=trace(w["write"])
    L=trace(w["write"]+"\n\nLater:\n"+w["relate"])
    R=[trace(q) for q in w["recall"]]
    N=[trace(q) for q in w["near"]]
    F=[trace(q) for q in w["far"]]
    DATA.append({"W":W,"L":L,"R":R,"N":N,"F":F})
gc.collect();torch.cuda.empty_cache()

def compare_family(extractor,C):
    out={k:[] for k in ["WR","LR","WN","WF","CROSS"]}
    out_vec={k:[] for k in out}
    for i,d in enumerate(DATA):
        W=extractor(d["W"]);L=extractor(d["L"])
        for r in d["R"]:
            X=extractor(r)
            out["WR"].append(spectral_shape_similarity(W,X,C))
            out["LR"].append(spectral_shape_similarity(L,X,C))
            out_vec["WR"].append(full_modal_alignment(W,X,C))
            out_vec["LR"].append(full_modal_alignment(L,X,C))
        for n in d["N"]:
            X=extractor(n)
            out["WN"].append(spectral_shape_similarity(W,X,C))
            out_vec["WN"].append(full_modal_alignment(W,X,C))
        for f in d["F"]:
            X=extractor(f)
            out["WF"].append(spectral_shape_similarity(W,X,C))
            out_vec["WF"].append(full_modal_alignment(W,X,C))
        for j,e in enumerate(DATA):
            if i==j: continue
            for r in e["R"]:
                X=extractor(r)
                out["CROSS"].append(spectral_shape_similarity(W,X,C))
                out_vec["CROSS"].append(full_modal_alignment(W,X,C))
    return out,out_vec

print("\n[3/12] RESIDUAL anlamsal spektrum...")
RES_E,RES_V=compare_family(lambda t:t["R"],DCT)
for k in RES_E:
    m,md,s,lo,hi=stats(RES_E[k]);mv=np.mean(RES_V[k])
    print(f"      {k:5s} ENERGY={m:+.6f} median={md:+.6f} std={s:.6f} | MODAL-DIR={mv:+.6f}")

print("\n[4/12] RESIDUAL hareket/yörünge spektrumu...")
MOT_E,MOT_V=compare_family(lambda t:motion(t["R"]),DCT_MOTION)
for k in MOT_E:
    m,md,s,lo,hi=stats(MOT_E[k]);mv=np.mean(MOT_V[k])
    print(f"      {k:5s} ENERGY={m:+.6f} median={md:+.6f} std={s:.6f} | MODAL-DIR={mv:+.6f}")

print("\n[5/12] Q bütün-head anlamsal spektrumu...")
Q_E,Q_V=compare_family(lambda t:flatten_qkv(t,"Q"),DCT)
for k in Q_E:
    m,md,s,lo,hi=stats(Q_E[k]);mv=np.mean(Q_V[k])
    print(f"      {k:5s} ENERGY={m:+.6f} median={md:+.6f} std={s:.6f} | MODAL-DIR={mv:+.6f}")

print("\n[6/12] K bütün-head anlamsal spektrumu...")
K_E,K_V=compare_family(lambda t:flatten_qkv(t,"K"),DCT)
for k in K_E:
    m,md,s,lo,hi=stats(K_E[k]);mv=np.mean(K_V[k])
    print(f"      {k:5s} ENERGY={m:+.6f} median={md:+.6f} std={s:.6f} | MODAL-DIR={mv:+.6f}")

print("\n[7/12] V bütün-head anlamsal spektrumu...")
V_E,V_V=compare_family(lambda t:flatten_qkv(t,"V"),DCT)
for k in V_E:
    m,md,s,lo,hi=stats(V_E[k]);mv=np.mean(V_V[k])
    print(f"      {k:5s} ENERGY={m:+.6f} median={md:+.6f} std={s:.6f} | MODAL-DIR={mv:+.6f}")

print("\n[8/12] Tek tek frekans seçmeden tam modal profil...")
# Gold yalnız post-hoc karşılaştırma içindir.
# Hiçbir mod retrieval için seçilmez.
MODAL={"R":{"WR":[],"WN":[],"CROSS":[]},"MOTION":{"WR":[],"WN":[],"CROSS":[]},"Q":{"WR":[],"WN":[],"CROSS":[]},"K":{"WR":[],"WN":[],"CROSS":[]},"V":{"WR":[],"WN":[],"CROSS":[]}}
EXTRACT={
"R":(lambda t:t["R"],DCT),
"MOTION":(lambda t:motion(t["R"]),DCT_MOTION),
"Q":(lambda t:flatten_qkv(t,"Q"),DCT),
"K":(lambda t:flatten_qkv(t,"K"),DCT),
"V":(lambda t:flatten_qkv(t,"V"),DCT)
}
for name,(fn,C) in EXTRACT.items():
    for i,d in enumerate(DATA):
        W=fn(d["W"])
        for r in d["R"]: MODAL[name]["WR"].append(spectral_vector_alignment(W,fn(r),C).numpy())
        for n in d["N"]: MODAL[name]["WN"].append(spectral_vector_alignment(W,fn(n),C).numpy())
        for j,e in enumerate(DATA):
            if i==j: continue
            for r in e["R"]: MODAL[name]["CROSS"].append(spectral_vector_alignment(W,fn(r),C).numpy())
    wr=np.mean(np.stack(MODAL[name]["WR"]),axis=0)
    wn=np.mean(np.stack(MODAL[name]["WN"]),axis=0)
    cr=np.mean(np.stack(MODAL[name]["CROSS"]),axis=0)
    print(f"\n      {name} — bütün modlar; seçim YOK")
    for m in range(len(wr)):
        print(f"      M{m:02d} WR={wr[m]:+.5f} NEAR={wn[m]:+.5f} CROSS={cr[m]:+.5f} ΔWR-CROSS={wr[m]-cr[m]:+.5f}")

print("\n[9/12] Dünya-içi özgüllük testi — retrieval yapmadan post-hoc matris...")
# Her WRITE ile her dünyanın RECALL spektrumu ölçülür.
# Argmax kullanılmaz. Accuracy üretilmez. Matris yalnız X-ray'dir.
def world_matrix(extractor,C):
    M=np.zeros((len(DATA),len(DATA)),dtype=np.float64)
    for i,d in enumerate(DATA):
        W=extractor(d["W"])
        for j,e in enumerate(DATA):
            vals=[full_modal_alignment(W,extractor(r),C) for r in e["R"]]
            M[i,j]=np.mean(vals)
    return M

WM={}
for name,(fn,C) in EXTRACT.items():
    M=world_matrix(fn,C);WM[name]=M
    diag=float(np.diag(M).mean())
    off=float(M[~np.eye(len(M),dtype=bool)].mean())
    print(f"\n      {name} MATRİSİ — diagonal={diag:+.6f} cross={off:+.6f} Δ={diag-off:+.6f}")
    for row in M:
        print("      "+" ".join(f"{x:+.4f}" for x in row))

print("\n[10/12] WRITE → RELATE spektral devamlılık...")
REL={}
for name,(fn,C) in EXTRACT.items():
    vals=[]
    for d in DATA:
        vals.append(full_modal_alignment(fn(d["W"]),fn(d["L"]),C))
    REL[name]=vals
    m,md,s,lo,hi=stats(vals)
    print(f"      {name:7s} mean={m:+.6f} median={md:+.6f} min={lo:+.6f} max={hi:+.6f}")

print("\n[11/12] Kör ayırma özeti — yeni yasa üretmeden...")
# Burada yalnız deney grubunun ortalama farkı raporlanır.
# Bu fark threshold, score veya retrieval mekanizması değildir.
PANELS={
"RES_ENERGY":RES_E,"RES_MODAL":RES_V,
"MOTION_ENERGY":MOT_E,"MOTION_MODAL":MOT_V,
"Q_ENERGY":Q_E,"Q_MODAL":Q_V,
"K_ENERGY":K_E,"K_MODAL":K_V,
"V_ENERGY":V_E,"V_MODAL":V_V
}
DELTAS={}
for name,p in PANELS.items():
    wr=float(np.mean(p["WR"]));near=float(np.mean(p["WN"]));far=float(np.mean(p["WF"]));cross=float(np.mean(p["CROSS"]))
    DELTAS[name]={"WR":wr,"NEAR":near,"FAR":far,"CROSS":cross,"D_NEAR":wr-near,"D_CROSS":wr-cross}
    print(f"      {name:14s} WR={wr:+.6f} NEAR={near:+.6f} FAR={far:+.6f} CROSS={cross:+.6f} ΔNEAR={wr-near:+.6f} ΔCROSS={wr-cross:+.6f}")

print("\n[12/12] Protokol / sentinel / mühür...")
S1=sentinel();WEIGHT_OK=S0==S1
print("      Weight sentinel                    :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensor                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Eğitim / LoRA / optimizer          : YOK")
print("      Retrieval                          : YOK")
print("      Memory ID                          : YASAK / KULLANILMADI")
print("      Harf-rakam adres                   : YASAK / KULLANILMADI")
print("      Token-ID adresi                    : YASAK / KULLANILMADI")
print("      Tek layer/head adresi              : YASAK / KULLANILMADI")
print("      Tek frekans/mod adresi             : YASAK / KULLANILMADI")
print("      Tek vektör key                     : YASAK / KULLANILMADI")
print("      Router / threshold / lambda        : YOK")
print("      Fitted scorer                      : YOK")
print("      Gold ile seçim                     : YOK")
print("      DCT                                : SABİT / ORTONORMAL / ÖĞRENİLMEMİŞ")
print("      Tam modal dağılım                  : X-RAY")
print("      Model ağırlıkları                  :","DEĞİŞMEDİ" if WEIGHT_OK else "HATA")

SUMMARY={
"test":TEST,"parent542":PARENT542,"lock_sha":LOCK_SHA,"weight_sentinel":WEIGHT_OK,
"retrieval":False,"human_id":False,"single_frequency_address":False,"router":False,"training":False,
"deltas":DELTAS,
"world_matrix":{
    k:{
        "diag":float(np.diag(v).mean()),
        "cross":float(v[~np.eye(len(v),dtype=bool)].mean()),
        "delta":float(np.diag(v).mean()-v[~np.eye(len(v),dtype=bool)].mean())
    } for k,v in WM.items()
},
"write_relate":{k:float(np.mean(v)) for k,v in REL.items()},
"verdict":"TEST543_SEMANTIC_SPECTRUM_ENIGMA_XRAY_TAMAMLANDI"
}
RESULT_SHA=hashlib.sha256(json.dumps(SUMMARY,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*156)
print("TEST543 SONUÇ — SEMANTIC SPECTRUM / ENIGMA X-RAY")
print("="*156)
print("MODEL                         :",MODEL_ID)
print("DÜNYA SAYISI                  :",len(WORLDS))
print("MOD                           : KEŞİF / X-RAY")
print("FREKANS TANIMI                : FİZİKSEL Hz DEĞİL · TRANSFORMER LAYER-AXIS MODAL YAPI")
print("RETRIEVAL                     : YOK")
print("MEMORY ID                     : YOK")
print("TEK FREKANS / MOD ADRESİ      : YOK")
print("ROUTER                        : YOK")
print("-"*156)
for name,d in DELTAS.items():
    print(f"{name:29s}: WR={d['WR']:+.6f} CROSS={d['CROSS']:+.6f} Δ={d['D_CROSS']:+.6f}")
print("-"*156)
print("DÜNYA-İÇİ MODAL MATRİS:")
for name,M in WM.items():
    diag=float(np.diag(M).mean());off=float(M[~np.eye(len(M),dtype=bool)].mean())
    print(f"{name:29s}: SAME={diag:+.6f} OTHER={off:+.6f} Δ={diag-off:+.6f}")
print("-"*156)
print("WRITE→RELATE MODAL DEVAMLILIK:")
for name,v in REL.items(): print(f"{name:29s}: {np.mean(v):+.6f}")
print("-"*156)
print("WEIGHT SENTINEL               :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE TENSORS             :",sum(int(p.requires_grad) for p in model.parameters()))
print("TEST542 RESULT SHA            :",PARENT542)
print("TEST543 ÖN KİLİT SHA          :",LOCK_SHA)
print("TEST543 RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                         : TEST543_SEMANTIC_SPECTRUM_ENIGMA_XRAY_TAMAMLANDI")
print("YORUM KURALI                  : Tek mod/head/layer seçme. Önce SAME-vs-OTHER dağıtık ayrışmanın var olup olmadığına bak.")
print("="*156)
