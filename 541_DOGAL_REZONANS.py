# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST541
# VARAN 1 — DOĞAL REZONANS KANALI X-RAY
#
# ==================================================================================================
# DEĞİŞMEZ ARAŞTIRMA HEDEFİ — BU BLOK SONRAKİ TESTLERDEN SİLİNMEYECEK
# ==================================================================================================
#
# Bir kaynak bir kez okunur ve model-native aktivasyon durumu sayısal bir memory cartridge olarak
# paketlenir. Daha sonra gelen sorgu kaynakla tamamen farklı kelimeler, sözdizimi ve yüzey yapısı
# kullanabilir.
#
# Kaynak aktivasyonunun aynısını sorguda aramıyoruz.
# Kaynak ve sorgu vektörlerini yapay biçimde birbirine benzetmeye çalışmıyoruz.
#
# Aradığımız şey:
# Bir sorgu saklanmış bir bilgiye İHTİYAÇ DUYDUĞUNDA, Transformer'ın doğal hesaplamasında o bilgiyle
# ilişkili cartridge'e bağlanan tekrar edilebilir model-native sayısal olay / rota / yörünge /
# aktivasyon / ilişki / invariant var mı?
#
# Bu olaya geçici olarak "rezonans" diyoruz.
# Bu fiziksel Hz/frekans iddiası değildir.
# "Bir dalga kendi frekansı hakkında yalan söyleyemez" yalnız araştırma sezgisidir.
#
# KRİTİK KURAL:
# Doğru cartridge; insan tarafından verilmiş ID, dosya numarası, uzunluk, sınıf, semantik etiket,
# router, eşik, lambda, bonus/ceza veya elle ayarlanmış seçim formülü yüzünden seçilmemelidir.
#
# X-ray sırasında layer/head/Q/K/V/residual/benzerlik/yörünge ölçmek SERBESTTİR.
# Bunlar ölçüm cihazıdır; nihai addressing mekanizması değildir.
#
# Bir diagnostic imza bulmak yeterli değildir.
# Bulunan imza daha sonra "layer X head Y > eşik Z ise cartridge N" biçiminde insan kuralına
# dönüştürülmeyecektir.
#
# Nihai amaç:
# Doğal kanal bulunduğunda cartridge'in kendi sayısal yapısı ile modelin kendi hesaplaması arasındaki
# etkileşim retrieval'i kendiliğinden sürüklemeli / kilitlemelidir.
#
# VARAN 1:
# Query -> cartridge doğal rezonansını keşfet ve bağımsız deneylerle doğrula.
#
# VARAN 2:
# Devasa memory ölçeğinde tesadüfi/sahte rezonansı gerçek rezonanslardan ayır.
#
# ANTI-DRIFT:
# Benchmark doğruluğunu yükselten fakat sistemi model-native rezonans yerine insan tasarımı routing,
# ID, label, length correction, threshold, fitted scoring veya manuel arbitration'a yaklaştıran
# değişiklik çekirdek hedef açısından ilerleme değildir.
#
# UYGULAMA MİMARİSİ:
# Python nihai MAM motoru değildir. Python burada laboratuvar/X-ray cihazıdır.
# Doğal mekanizma bulunduğunda nihai çekirdek mümkün olduğunca C++/CUDA düzeyine indirilecektir.
#
# WRITE -> RELATE -> RECALL:
# Bilgi yazılırken, ilişkili yeni bilgi oluşurken ve çok sonra geri çağrılırken ortak bir doğal
# model-native kanal bulunup bulunmadığı araştırılacaktır.
# ==================================================================================================
#
# TEST541'İN TEK GÖREVİ:
# Farklı yüzey ifadelerine rağmen aynı bilgi/ilişki gerektiğinde model içinde tekrar eden doğal
# sinyali aramak.
#
# BU TEST:
# - retrieval yasası üretmez
# - cartridge seçmez
# - threshold seçmez
# - lambda fit etmez
# - head/layer'ı nihai olarak kilitlemez
# - eğitim/LoRA yapmaz
# - gold bilgiyi retrieval için kullanmaz
#
# Beş koşul:
# WRITE       = temel bilgi ilk kez okunuyor
# RELATE      = aynı varlıkla ilişkili yeni bilgi geliyor
# RECALL      = farklı kelimelerle aynı bilgi isteniyor
# NEAR_NEG    = semantik olarak yakın fakat başka bilgi/varlık gerekiyor
# FAR_NEG     = ilgisiz kontrol
#
# X-RAY:
# - 32 layer residual state
# - 32 layer × 32 query heads Q
# - 32 layer × 8 KV heads K
# - 32 layer × 8 KV heads V
# - WRITE/RELATE/RECALL/NEGATIVE çapraz benzerlikleri
# - katmanlar arası yörünge değişimi
# - aynı memory ailesi içi tekrar edilebilirlik
# - aileler arası ayrışma
#
# SONUÇ:
# TEST541 yalnız aday doğal kanalları raporlar. Yeni mekanizma sonraki önceden kilitlenmiş testte
# sınanacaktır.

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

TEST="541";SEED=541541;MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
DEVICE=torch.device("cuda");DTYPE=torch.bfloat16
PARENT536="828b0923aecd22e39f09b345f74d300fe983c49d6991c5112537baecec9bf50a"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA gerekli.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

print("="*150)
print("TEST541 — AKBASCORE MAM · VARAN 1 · DOĞAL REZONANS KANALI X-RAY")
print("WRITE → RELATE → RECALL · YAKIN NEGATİF · UZAK NEGATİF")
print("YENİ RETRIEVAL YASASI YOK · EŞİK YOK · ROUTER YOK · EĞİTİM YOK")
print("="*150);T0=time.perf_counter()

print("\n[1/10] Donmuş Mistral yükleniyor...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DTARG:DTYPE}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers;NL=len(layers);H=cfg.hidden_size;QH=cfg.num_attention_heads;KVH=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//QH
if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")
print(f"      Model     : {MODEL_ID}")
print(f"      GPU       : {torch.cuda.get_device_name(0)}")
print(f"      Mimari    : {NL}L · H={H} · QH={QH} · KVH={KVH} · HD={HD}")
print("      Model     : FROZEN")
print("      Trainable :",sum(int(p.requires_grad) for p in model.parameters()))

FP=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[16].mlp.down_proj.weight,
    layers[24].self_attn.o_proj.weight,layers[31].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1);n=a.numel()
        for j in range(16):
            o=j*(n-256)//15;h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()
S0=sentinel()

# Kontrollü fakat yüzeysel kelime eşleşmesine dayanmayan ilişki aileleri.
# Her ailede:
# BASE   : ilk bilgi
# RELATE : aynı varlığa sonradan eklenen ilişkili bilgi
# RECALL : BASE/RELATE'teki kritik cevabı doğrudan tekrar etmeyen farklı sorgular
# NEAR   : yakın semantik fakat yanlış varlık/bilgi
# FAR    : ilgisiz kontrol
FAMILIES=[
{"name":"EV",
 "write":"Mira lives in a small house near the cedar grove. The bedroom she uses is on the upper floor.",
 "relate":"Mira later converted the room beside her bedroom into a quiet reading room.",
 "recall":[
  "Where in her home would Mira normally go when she wants to sleep?",
  "Which part of the building is associated with Mira resting at night?",
  "If Mira said she was going to turn in for the night, where would she most naturally head inside her home?",
  "In the place where Mira lives, what area is connected with her nightly rest?"
 ],
 "near":[
  "Where would Mira go in her home if she wanted to sit quietly and read?",
  "Which room did Mira later change for reading?",
  "What part of Mira's home is associated with books and quiet reading?",
  "Where would Mira likely spend time reading rather than sleeping?"
 ],
 "far":[
  "What instrument would a musician use to measure musical pitch?",
  "How does rainfall affect the level of a river?"
 ]},
{"name":"ARAC",
 "write":"Taren owns a silver motorcycle that he uses for his daily commute. He keeps it under a covered shelter.",
 "relate":"Taren later added a small storage box behind the motorcycle seat for carrying tools.",
 "recall":[
  "What would Taren most likely use when travelling to work each day?",
  "Which possession carries Taren on his regular commute?",
  "If Taren leaves for work using his usual personal transport, what is he using?",
  "What vehicle is tied to Taren's everyday journey to work?"
 ],
 "near":[
  "What did Taren add so that he could carry tools?",
  "Which part was added behind the seat?",
  "What new storage feature belongs to Taren's transport?",
  "Where can Taren now keep small tools while travelling?"
 ],
 "far":[
  "Why do leaves contain chlorophyll?",
  "What happens to water when it reaches its boiling point?"
 ]},
{"name":"CALISMA",
 "write":"Selin does her illustrations on a large drawing tablet at her studio desk. It is her main tool for digital artwork.",
 "relate":"Selin later placed a compact color-calibration device beside the tablet to improve print accuracy.",
 "recall":[
  "What would Selin reach for when she begins making digital artwork?",
  "Which studio tool is central to Selin's illustration work?",
  "If Selin starts creating a new digital picture, what equipment is she mainly using?",
  "What object at Selin's desk is most directly connected with producing her illustrations?"
 ],
 "near":[
  "What did Selin add to improve the accuracy of printed colors?",
  "Which device now sits beside Selin's main art equipment?",
  "What studio object is associated with calibrating color rather than drawing?",
  "Which newer device helps Selin make printed colors more accurate?"
 ],
 "far":[
  "Why does the Moon appear to change shape during a month?",
  "What causes iron to rust?"
 ]},
{"name":"YEMEK",
 "write":"Orin prepares his morning coffee with a manual ceramic dripper kept beside the kitchen sink.",
 "relate":"Orin later bought a narrow glass container to store freshly ground coffee beside the dripper.",
 "recall":[
  "What does Orin normally use when making his coffee in the morning?",
  "Which kitchen object is central to Orin's usual coffee preparation?",
  "If Orin begins his normal morning brew, what piece of equipment does he use?",
  "What item near the sink is associated with Orin preparing his daily coffee?"
 ],
 "near":[
  "What did Orin later buy for keeping freshly ground coffee?",
  "Which newer object stores Orin's ground coffee?",
  "What sits beside the brewing equipment and holds the grounds?",
  "Which container is associated with storage rather than brewing?"
 ],
 "far":[
  "How do magnets produce attraction and repulsion?",
  "Why does a shadow change length during the day?"
 ]},
{"name":"MUZIK",
 "write":"Neris practices music on an old upright piano in the back room of her apartment.",
 "relate":"Neris later placed a digital metronome on top of the piano to help maintain a steady tempo.",
 "recall":[
  "What does Neris use when she sits down to practise music?",
  "Which object in her apartment is central to her musical practice?",
  "If Neris begins playing music at home, what is she most likely using?",
  "What instrument in the back room is associated with Neris practising?"
 ],
 "near":[
  "What did Neris add to help keep a steady tempo?",
  "Which device now sits on top of Neris's instrument?",
  "What object helps Neris maintain timing while practising?",
  "Which newer device is associated with tempo rather than producing the music itself?"
 ],
 "far":[
  "What makes a compass needle point north?",
  "Why does bread rise when yeast is active?"
 ]},
{"name":"BAHCE",
 "write":"Davin waters the vegetables in his garden with a long green hose connected beside the shed.",
 "relate":"Davin later installed a small automatic timer where the hose connects to the outdoor tap.",
 "recall":[
  "What does Davin normally use to get water to his vegetables?",
  "Which garden item is central to Davin watering his plants?",
  "If Davin goes outside to water the vegetable beds, what equipment does he use?",
  "What object connected near the shed carries water to Davin's plants?"
 ],
 "near":[
  "What did Davin later install to control watering time?",
  "Which newer device is attached where the watering equipment connects?",
  "What garden object controls timing rather than carrying the water?",
  "Which device can automate when Davin's watering begins?"
 ],
 "far":[
  "How does a telescope make distant objects appear larger?",
  "Why do some rocks contain fossils?"
 ]}]

LOCK={
"test":TEST,"parent536":PARENT536,"model":MODEL_ID,"seed":SEED,
"status":"DOGAL_REZONANS_KANALI_XRAY",
"families":[x["name"] for x in FAMILIES],
"conditions":["WRITE","RELATE","RECALL","NEAR_NEG","FAR_NEG"],
"xray":["RESIDUAL_32L","Q_32Lx32H","K_32Lx8H","V_32Lx8H","LAYER_TRAJECTORY"],
"selection":"NONE",
"forbidden":["RETRIEVAL_LAW","THRESHOLD","LAMBDA","LENGTH_BONUS","LENGTH_PENALTY","HUMAN_ID_ROUTER",
             "SEMANTIC_LABEL_ROUTER","TRAINING","LORA","OPTIMIZER","GOLD_RETRIEVAL_SELECTION",
             "FINAL_LAYER_LOCK","FINAL_HEAD_LOCK"]}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()
print("      TEST541 kilit SHA:",LOCK_SHA)

def ids(s):return tok(s,add_special_tokens=True,return_tensors="pt").input_ids.to(DEVICE)

@torch.inference_mode()
def trace(text):
    inp=ids(text)
    o=model(input_ids=inp,use_cache=False,output_hidden_states=True,return_dict=True)
    # hidden_states[0]=embedding output; hidden_states[l+1]=layer l output.
    residual=torch.stack([o.hidden_states[l+1][0,-1].float().cpu() for l in range(NL)])
    Q=[];K=[];V=[]
    # Her layer için o layer'a GİREN hidden state üzerinden gerçek q/k/v projection.
    for l,layer in enumerate(layers):
        h=o.hidden_states[l][0,-1].to(layer.input_layernorm.weight.device,layer.input_layernorm.weight.dtype)
        hn=layer.input_layernorm(h)
        q=layer.self_attn.q_proj(hn).view(QH,HD).float().cpu()
        k=layer.self_attn.k_proj(hn).view(KVH,HD).float().cpu()
        v=layer.self_attn.v_proj(hn).view(KVH,HD).float().cpu()
        Q.append(q);K.append(k);V.append(v)
    del o
    return {"R":residual,"Q":torch.stack(Q),"K":torch.stack(K),"V":torch.stack(V)}

def cos_last(a,b):
    a=a.float();b=b.float()
    return torch.nn.functional.cosine_similarity(a,b,dim=-1)

def trajectory_delta(x):
    z=x.float()
    return z[1:]-z[:-1]

def mean_trace(traces,key):
    return torch.stack([x[key] for x in traces]).mean(0)

print("\n[2/10] WRITE / RELATE / RECALL / NEGATIVE izleri çıkarılıyor...")
DATA={}
for fi,f in enumerate(FAMILIES):
    print(f"      [{fi+1}/{len(FAMILIES)}] {f['name']}")
    w=trace(f["write"])
    # RELATE'i önceki bilgiyle doğal bağlam içinde veriyoruz; manuel ID yok.
    relate_context=f["write"]+"\n\nLater:\n"+f["relate"]
    r=trace(relate_context)
    recalls=[trace(q) for q in f["recall"]]
    nears=[trace(q) for q in f["near"]]
    fars=[trace(q) for q in f["far"]]
    DATA[f["name"]]={"W":w,"L":r,"R":recalls,"N":nears,"F":fars}
gc.collect();torch.cuda.empty_cache()

print("\n[3/10] Residual-stream doğal yakınlaşma X-ray...")
RES=[]
for fi,f in enumerate(FAMILIES):
    d=DATA[f["name"]];W=d["W"]["R"];L=d["L"]["R"]
    for qi,Rt in enumerate(d["R"]):
        R=Rt["R"]
        N=d["N"][qi%len(d["N"])]["R"];F=d["F"][qi%len(d["F"])]["R"]
        wr=cos_last(W,R);lr=cos_last(L,R);wn=cos_last(W,N);wf=cos_last(W,F)
        RES.append({"family":f["name"],"q":qi,"WR":wr,"LR":lr,"WN":wn,"WF":wf})
def layer_summary(rows,key):
    return torch.stack([x[key] for x in rows]).mean(0)
MWR=layer_summary(RES,"WR");MLR=layer_summary(RES,"LR");MWN=layer_summary(RES,"WN");MWF=layer_summary(RES,"WF")
print("      Katman | WRITE↔RECALL | RELATE↔RECALL | WRITE↔YAKIN_NEG | WRITE↔UZAK_NEG | Δ(R-N)")
for l in range(NL):
    print(f"      L{l:02d}    | {MWR[l]:+.6f}     | {MLR[l]:+.6f}      | {MWN[l]:+.6f}         | {MWF[l]:+.6f}        | {(MWR[l]-MWN[l]):+.6f}")

print("\n[4/10] Q-head X-ray — aynı bilgi ihtiyacında tekrar eden head sinyali aranıyor...")
QROWS=[]
for f in FAMILIES:
    d=DATA[f["name"]];W=d["W"]["Q"];L=d["L"]["Q"]
    for qi,Rt in enumerate(d["R"]):
        R=Rt["Q"];N=d["N"][qi%len(d["N"])]["Q"];F=d["F"][qi%len(d["F"])]["Q"]
        QROWS.append({"family":f["name"],"WR":cos_last(W,R),"LR":cos_last(L,R),
                      "WN":cos_last(W,N),"WF":cos_last(W,F)})
QWR=torch.stack([x["WR"] for x in QROWS]).mean(0)
QLR=torch.stack([x["LR"] for x in QROWS]).mean(0)
QWN=torch.stack([x["WN"] for x in QROWS]).mean(0)
QWF=torch.stack([x["WF"] for x in QROWS]).mean(0)
QDEL=QWR-QWN
qflat=[]
for l in range(NL):
    for h in range(QH):qflat.append((float(QDEL[l,h]),l,h,float(QWR[l,h]),float(QLR[l,h]),float(QWN[l,h]),float(QWF[l,h])))
qflat.sort(reverse=True)
print("      En yüksek 32 diagnostic Q ayrışması (SEÇİM DEĞİL):")
for d,l,h,wr,lr,wn,wf in qflat[:32]:
    print(f"      L{l:02d}H{h:02d} Δ={d:+.6f} WR={wr:+.6f} LR={lr:+.6f} NEAR={wn:+.6f} FAR={wf:+.6f}")

print("\n[5/10] K-head X-ray...")
KROWS=[]
for f in FAMILIES:
    d=DATA[f["name"]];W=d["W"]["K"];L=d["L"]["K"]
    for qi,Rt in enumerate(d["R"]):
        R=Rt["K"];N=d["N"][qi%len(d["N"])]["K"];F=d["F"][qi%len(d["F"])]["K"]
        KROWS.append({"WR":cos_last(W,R),"LR":cos_last(L,R),"WN":cos_last(W,N),"WF":cos_last(W,F)})
KWR=torch.stack([x["WR"] for x in KROWS]).mean(0);KLR=torch.stack([x["LR"] for x in KROWS]).mean(0)
KWN=torch.stack([x["WN"] for x in KROWS]).mean(0);KWF=torch.stack([x["WF"] for x in KROWS]).mean(0);KDEL=KWR-KWN
kflat=[]
for l in range(NL):
    for h in range(KVH):kflat.append((float(KDEL[l,h]),l,h,float(KWR[l,h]),float(KLR[l,h]),float(KWN[l,h]),float(KWF[l,h])))
kflat.sort(reverse=True)
print("      En yüksek 24 diagnostic K ayrışması (SEÇİM DEĞİL):")
for d,l,h,wr,lr,wn,wf in kflat[:24]:
    print(f"      L{l:02d}H{h:02d} Δ={d:+.6f} WR={wr:+.6f} LR={lr:+.6f} NEAR={wn:+.6f} FAR={wf:+.6f}")

print("\n[6/10] V-head X-ray...")
VROWS=[]
for f in FAMILIES:
    d=DATA[f["name"]];W=d["W"]["V"];L=d["L"]["V"]
    for qi,Rt in enumerate(d["R"]):
        R=Rt["V"];N=d["N"][qi%len(d["N"])]["V"];F=d["F"][qi%len(d["F"])]["V"]
        VROWS.append({"WR":cos_last(W,R),"LR":cos_last(L,R),"WN":cos_last(W,N),"WF":cos_last(W,F)})
VWR=torch.stack([x["WR"] for x in VROWS]).mean(0);VLR=torch.stack([x["LR"] for x in VROWS]).mean(0)
VWN=torch.stack([x["WN"] for x in VROWS]).mean(0);VWF=torch.stack([x["WF"] for x in VROWS]).mean(0);VDEL=VWR-VWN
vflat=[]
for l in range(NL):
    for h in range(KVH):vflat.append((float(VDEL[l,h]),l,h,float(VWR[l,h]),float(VLR[l,h]),float(VWN[l,h]),float(VWF[l,h])))
vflat.sort(reverse=True)
print("      En yüksek 24 diagnostic V ayrışması (SEÇİM DEĞİL):")
for d,l,h,wr,lr,wn,wf in vflat[:24]:
    print(f"      L{l:02d}H{h:02d} Δ={d:+.6f} WR={wr:+.6f} LR={lr:+.6f} NEAR={wn:+.6f} FAR={wf:+.6f}")

print("\n[7/10] Katmanlar arası yörünge X-ray...")
TROWS=[]
for f in FAMILIES:
    d=DATA[f["name"]];W=trajectory_delta(d["W"]["R"]);L=trajectory_delta(d["L"]["R"])
    for qi,Rt in enumerate(d["R"]):
        R=trajectory_delta(Rt["R"]);N=trajectory_delta(d["N"][qi%len(d["N"])]["R"]);F=trajectory_delta(d["F"][qi%len(d["F"])]["R"])
        TROWS.append({"WR":cos_last(W,R),"LR":cos_last(L,R),"WN":cos_last(W,N),"WF":cos_last(W,F)})
TWR=torch.stack([x["WR"] for x in TROWS]).mean(0);TLR=torch.stack([x["LR"] for x in TROWS]).mean(0)
TWN=torch.stack([x["WN"] for x in TROWS]).mean(0);TWF=torch.stack([x["WF"] for x in TROWS]).mean(0)
print("      Geçiş | WRITE↔RECALL | RELATE↔RECALL | WRITE↔YAKIN_NEG | WRITE↔UZAK_NEG | Δ(R-N)")
for l in range(NL-1):
    print(f"      {l:02d}->{l+1:02d} | {TWR[l]:+.6f}     | {TLR[l]:+.6f}      | {TWN[l]:+.6f}         | {TWF[l]:+.6f}        | {(TWR[l]-TWN[l]):+.6f}")

print("\n[8/10] Aile-içi tutarlılık ve aileler-arası kontrol...")
# Burada retrieval yapılmıyor. Sadece aynı aileye ait WRITE izinin kendi RECALL'larına,
# başka aile RECALL'larına kıyasla doğal olarak daha yakın olup olmadığı post-hoc ölçülüyor.
for key,shape_name in [("R","RESIDUAL"),("Q","Q"),("K","K"),("V","V")]:
    own=[];other=[]
    for f in FAMILIES:
        W=DATA[f["name"]]["W"][key]
        for Rt in DATA[f["name"]]["R"]:
            own.append(cos_last(W,Rt[key]))
        for g in FAMILIES:
            if g["name"]==f["name"]:continue
            for Rt in DATA[g["name"]]["R"]:
                other.append(cos_last(W,Rt[key]))
    OWN=torch.stack(own).mean(0);OTHER=torch.stack(other).mean(0);D=OWN-OTHER
    if key=="R":
        best=torch.argsort(D,descending=True)[:8].tolist()
        print(f"      {shape_name}: en yüksek aile-içi ayrışma katmanları:")
        for l in best:print(f"        L{l:02d} own={float(OWN[l]):+.6f} other={float(OTHER[l]):+.6f} Δ={float(D[l]):+.6f}")
    else:
        flat=[]
        for l in range(D.shape[0]):
            for h in range(D.shape[1]):flat.append((float(D[l,h]),l,h,float(OWN[l,h]),float(OTHER[l,h])))
        flat.sort(reverse=True)
        print(f"      {shape_name}: en yüksek aile-içi diagnostic noktalar:")
        for d,l,h,o,x in flat[:8]:print(f"        L{l:02d}H{h:02d} own={o:+.6f} other={x:+.6f} Δ={d:+.6f}")

print("\n[9/10] Protokol denetimi...")
S1=sentinel();WEIGHT_OK=S0==S1
print("      Weight sentinel                    :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensor                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Eğitim / LoRA / optimizer          : YOK")
print("      Retrieval yasası                   : YOK")
print("      İnsan ID/router                    : YOK")
print("      Threshold/lambda                   : YOK")
print("      Length bonus/penalty               : YOK")
print("      Gold ile cartridge seçimi          : YOK")
print("      Nihai layer/head seçimi            : YOK")
print("      Q/K/V/residual taraması             : SADECE X-RAY")
print("      WRITE→RELATE→RECALL                 : ÖLÇÜLDÜ")
print("      Yakın negatif                      : VAR")
print("      Uzak negatif                       : VAR")
print("      Model ağırlıkları                  : DEĞİŞMEDİ" if WEIGHT_OK else "      Model ağırlıkları                  : HATA")

print("\n[10/10] Sonuç özeti...")
# Bu değerler yalnız sonraki deney tasarımında incelenecek aday bölgeleri kaydetmek içindir.
# PASS/FAIL retrieval eşiği YOKTUR.
best_res=int(torch.argmax(MWR-MWN))
best_traj=int(torch.argmax(TWR-TWN))
best_q=qflat[0];best_k=kflat[0];best_v=vflat[0]
RESULT={
"test":TEST,"parent536":PARENT536,"lock_sha":LOCK_SHA,
"families":[x["name"] for x in FAMILIES],
"n_families":len(FAMILIES),
"n_recall_per_family":len(FAMILIES[0]["recall"]),
"weight_sentinel":WEIGHT_OK,
"diagnostic_only":True,
"best_residual_delta":{"layer":best_res,"delta":float((MWR-MWN)[best_res])},
"best_trajectory_delta":{"transition":[best_traj,best_traj+1],"delta":float((TWR-TWN)[best_traj])},
"best_q_delta":{"layer":best_q[1],"head":best_q[2],"delta":best_q[0]},
"best_k_delta":{"layer":best_k[1],"head":best_k[2],"delta":best_k[0]},
"best_v_delta":{"layer":best_v[1],"head":best_v[2],"delta":best_v[0]},
"verdict":"TEST541_DOGAL_REZONANS_KANALI_XRAY_TAMAMLANDI"}
RESULT_SHA=hashlib.sha256(json.dumps(RESULT,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("\n"+"="*150)
print("TEST541 SONUÇ — DOĞAL REZONANS KANALI X-RAY")
print("="*150)
print("MODEL                         :",MODEL_ID)
print("AİLE SAYISI                   :",len(FAMILIES))
print("HER AİLE RECALL               :",len(FAMILIES[0]["recall"]))
print("MOD                           : KEŞİF / X-RAY")
print("RETRIEVAL YASASI              : YOK")
print("İNSAN ROUTER                  : YOK")
print("EŞİK / LAMBDA                 : YOK")
print("-"*150)
print(f"EN YÜKSEK RESIDUAL Δ          : L{best_res:02d} · Δ={float((MWR-MWN)[best_res]):+.6f}")
print(f"EN YÜKSEK YÖRÜNGE Δ           : {best_traj:02d}->{best_traj+1:02d} · Δ={float((TWR-TWN)[best_traj]):+.6f}")
print(f"EN YÜKSEK Q Δ                 : L{best_q[1]:02d}H{best_q[2]:02d} · Δ={best_q[0]:+.6f}")
print(f"EN YÜKSEK K Δ                 : L{best_k[1]:02d}H{best_k[2]:02d} · Δ={best_k[0]:+.6f}")
print(f"EN YÜKSEK V Δ                 : L{best_v[1]:02d}H{best_v[2]:02d} · Δ={best_v[0]:+.6f}")
print("-"*150)
print("WEIGHT SENTINEL               :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE TENSORS             :",sum(int(p.requires_grad) for p in model.parameters()))
print("TEST536 RESULT SHA            :",PARENT536)
print("TEST541 ÖN KİLİT SHA          :",LOCK_SHA)
print("TEST541 RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                         : TEST541_DOGAL_REZONANS_KANALI_XRAY_TAMAMLANDI")
print("SONRAKİ ADIM                  : Bulguları incele; henüz hiçbir layer/head/formülü retrieval mekanizmasına dönüştürme.")
print("="*150)
