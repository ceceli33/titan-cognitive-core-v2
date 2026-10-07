# Copyright © 2026 Mustafa Akbaş
# AKBASCORE MAM — TEST542
# VARAN 1 — TRANSFORMER İÇ GÖKYÜZÜ / BÜTÜNSEL GEOMETRİ X-RAY
#
# ==================================================================================================
# DEĞİŞMEZ ARAŞTIRMA ANAYASASI
# ==================================================================================================
#
# HEDEF:
# Transformer içinde bir bilgi işlenirken oluşan model-native çok boyutlu yapıyı tek bir sayı,
# harf, token, ID, dosya numarası, layer, head veya elle oluşturulmuş anahtara indirgemeden incelemek.
#
# CARTRIDGE:
# Bir "adres + veri" paketi değildir.
# Bir ID değildir.
# Bir token değildir.
# Tek vektör değildir.
# Tek layer/head değildir.
# Tek cosine değeri değildir.
#
# Cartridge için araştırılan nesne:
# Bilgi işlenirken Transformer boyunca oluşan çok katmanlı iç geometrinin / yörüngenin /
# göreli yapının model-native sayısal izi.
#
# YASAK — KESİN:
# - İnsan tarafından verilmiş memory ID
# - Harf/rakam kimliği ile retrieval
# - Dosya numarası / cartridge numarası ile retrieval
# - Token-ID'yi memory adresi yapmak
# - Tek head'i memory adresi yapmak
# - Tek layer'ı memory adresi yapmak
# - Tek vektörü memory key yapmak
# - "En yüksek cosine = memory" retrieval'i
# - Semantik label router
# - İnsan yapımı router
# - Threshold/lambda/bonus/penalty
# - Length correction
# - Gold bilgiyi seçim için kullanmak
# - Eğitim/LoRA/optimizer
#
# İZİN VERİLEN:
# Q/K/V/residual ve katman geçişlerini yalnız X-RAY cihazı olarak ölçmek.
# Ölçülen tek bir nokta retrieval adresine dönüştürülemez.
#
# YILDIZ HARİTASI İLKESİ:
# Bir yıldız tek başına gökyüzü değildir.
# Bir layer/head/koordinat tek başına memory değildir.
# Aranan şey yıldızların birbirlerine göre oluşturduğu bütünsel geometridir.
#
# WRITE:
# Bilginin ilk işlenişinde oluşan Transformer iç geometrisi.
#
# RELATE:
# Daha sonra aynı dünya/varlıkla ilgili yeni bilgi geldiğinde oluşan geometri ve eski bilgiyle
# model-native geometrik ilişkinin izi.
#
# RECALL:
# Kaynak kelimeleri tekrar etmeden aynı bilgiye ihtiyaç duyan sorgunun oluşturduğu geometri.
#
# TEST542 RETRIEVAL TESTİ DEĞİLDİR.
# TEST542 bir retrieval formülü üretmeyecektir.
# TEST542 "en iyi head" seçmeyecektir.
# TEST542 yalnız bütünsel geometrinin WRITE -> RELATE -> RECALL boyunca korunup korunmadığını arar.
#
# Python yalnız laboratuvar cihazıdır. Nihai MAM motoru değildir.
# ==================================================================================================

import os,sys,subprocess,importlib.util,random,time,hashlib,json,gc,math
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
    if importlib.util.find_spec(mod) is None:
        subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])

import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

TEST="542"
SEED=542542
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
PARENT541="d070361b8d9cc72e79e834ea8e030c764bfe73ae8faf817c6fa5a39c51207ea6"
DEVICE=torch.device("cuda")
DTYPE=torch.bfloat16
os.environ["TOKENIZERS_PARALLELISM"]="false"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA gerekli.")

torch.set_grad_enabled(False)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

print("="*154)
print("TEST542 — AKBASCORE MAM · VARAN 1 · TRANSFORMER İÇ GÖKYÜZÜ / BÜTÜNSEL GEOMETRİ X-RAY")
print("WRITE → RELATE → RECALL · TEK ADRES YOK · TEK VEKTÖR KEY YOK · ROUTER YOK · RETRIEVAL YOK")
print("="*154)

T0=time.perf_counter()

print("\n[1/10] Donmuş Mistral yükleniyor...")
_tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2])
DTARG="dtype" if _tv>=(4,56) else "torch_dtype"

tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"":0},
    attn_implementation="sdpa",
    **{DTARG:DTYPE}
).eval()

for p in model.parameters():
    p.requires_grad_(False)

cfg=model.config
layers=model.model.layers
NL=len(layers)
H=cfg.hidden_size
QH=cfg.num_attention_heads
KVH=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//QH

if (NL,H,QH,KVH,HD)!=(32,4096,32,8,128):
    raise RuntimeError(f"Mimari uyuşmazlığı: {(NL,H,QH,KVH,HD)}")

print(f"      Model      : {MODEL_ID}")
print(f"      GPU        : {torch.cuda.get_device_name(0)}")
print(f"      Mimari     : {NL}L · H={H} · QH={QH} · KVH={KVH} · HD={HD}")
print("      Frozen     : EVET")
print("      Trainable  :",sum(int(p.requires_grad) for p in model.parameters()))

FP=[
    layers[0].self_attn.q_proj.weight,
    layers[8].self_attn.o_proj.weight,
    layers[16].mlp.down_proj.weight,
    layers[24].self_attn.o_proj.weight,
    layers[31].mlp.down_proj.weight,
    model.model.norm.weight,
    model.lm_head.weight
]

def sentinel():
    h=hashlib.sha256()
    for p in FP:
        a=p.detach().reshape(-1)
        n=a.numel()
        for j in range(16):
            o=j*(n-256)//15
            h.update(a[o:o+256].float().cpu().numpy().tobytes())
    return h.hexdigest()

S0=sentinel()

# Her dünya iki ilişkili bilgi taşır.
# Ama hiçbir WORLD adı veya Python index'i modele memory adresi olarak verilmez.
# Bunlar yalnız deney kayıtlarının hangi örneğe ait olduğunu sonradan bilmek içindir.
WORLDS=[
{
"write":"A woman named Elara lives in a stone cottage beside a pine-covered hill. She sleeps in a narrow room beneath the eastern window.",
"relate":"Months later, Elara turned the unused space beside that sleeping room into a place where she restores old books.",
"recall":[
"When night is over and Elara wants to rest, which part of her home is associated with that activity?",
"Inside Elara's home, where would she normally settle down for the night?",
"If Elara says she is going to sleep, what area of the cottage is relevant?",
"Which part of Elara's living space is tied to her nightly rest?"
],
"bridge":[
"What new use did Elara give to the space next to the place where she rests at night?",
"Which later change in Elara's home occurred beside her sleeping area?"
],
"near":[
"Where does Elara work on damaged old books?",
"What part of Elara's home became associated with book restoration?"
],
"far":[
"Why does a metal bridge expand on a hot day?",
"How do bees communicate the location of food?"
]
},
{
"write":"A man named Kalen keeps a dark touring bicycle in a wooden shelter behind his house. He rides it to the railway station each morning.",
"relate":"Later, Kalen attached a weatherproof pouch beneath the bicycle seat so that he could carry repair equipment.",
"recall":[
"What does Kalen normally rely on for the first part of his journey to the station?",
"Which possession carries Kalen from his home toward the railway each morning?",
"If Kalen starts his regular trip to catch a train, what personal transport does he use?",
"What object stored behind Kalen's house is associated with his morning journey?"
],
"bridge":[
"What later addition was made to the thing Kalen uses to reach the railway station?",
"How did Kalen modify his regular transport so repair equipment could travel with him?"
],
"near":[
"Where does Kalen keep his repair equipment while travelling?",
"What newer part of Kalen's transport is meant for carrying tools?"
],
"far":[
"Why can salt lower the freezing point of water?",
"What causes the phases of the Moon?"
]
},
{
"write":"Nara creates botanical illustrations using a pressure-sensitive drawing surface at the long table in her workroom.",
"relate":"Later, Nara placed a small optical calibration instrument beside her drawing equipment to make printed colors more consistent.",
"recall":[
"What equipment is central when Nara begins creating one of her digital botanical pictures?",
"What does Nara primarily work on when producing her illustrations?",
"If Nara starts drawing a new plant image in her workroom, what device is most directly involved?",
"Which object on Nara's work table is associated with creating the artwork itself?"
],
"bridge":[
"What later device did Nara place beside the equipment she uses to create her pictures?",
"How did Nara later supplement the main equipment used for her illustration work?"
],
"near":[
"What does Nara use to improve consistency of printed colors?",
"Which newer device beside her drawing equipment is associated with calibration?"
],
"far":[
"Why does ice float on liquid water?",
"How does a seed obtain energy before its first leaves develop?"
]
},
{
"write":"Soren prepares a hot drink every morning using a glass brewing vessel kept on the counter near the kitchen window.",
"relate":"Later, Soren placed a sealed ceramic container beside the brewing vessel to keep freshly ground beans away from moisture.",
"recall":[
"What piece of kitchen equipment is central to Soren's normal morning preparation?",
"What does Soren use when beginning his usual hot drink routine?",
"If Soren starts preparing his morning brew, which object near the window is involved?",
"What kitchen item is directly associated with Soren making the drink rather than storing ingredients?"
],
"bridge":[
"What later object did Soren place beside the equipment used for his morning brew?",
"How did Soren later add storage beside the object used to prepare his drink?"
],
"near":[
"What does Soren use to protect the ground beans from moisture?",
"Which newer kitchen object is intended for ingredient storage?"
],
"far":[
"Why do planets remain in orbit around a star?",
"How does sound travel through air?"
]
},
{
"write":"Ilyra practises music on a large stringed keyboard instrument in the quiet rear room of her apartment.",
"relate":"Later, Ilyra placed a small electronic timing device above the keys to help keep her practice rhythm steady.",
"recall":[
"What instrument is central when Ilyra begins practising music at home?",
"What does Ilyra play during her regular practice sessions?",
"If Ilyra goes into the rear room to make music, what is she using?",
"Which object in Ilyra's apartment actually produces the music she practises?"
],
"bridge":[
"What later device did Ilyra add to the instrument she uses for practice?",
"How did Ilyra later supplement her musical setup to help maintain steady timing?"
],
"near":[
"What helps Ilyra maintain a steady rhythm while practising?",
"Which newer device in Ilyra's setup is associated with timing?"
],
"far":[
"Why do some volcanic rocks contain holes?",
"How does a lens focus light?"
]
},
{
"write":"Marek supplies water to the vegetables behind his workshop using a flexible line connected to the outdoor tap.",
"relate":"Later, Marek installed a programmable control unit at the connection so watering could begin automatically before sunrise.",
"recall":[
"What equipment normally carries water from Marek's tap toward the vegetables?",
"What does Marek use to move water across the garden?",
"If Marek waters the vegetable beds manually, what object actually carries the water?",
"Which item connected outside the workshop is directly responsible for delivering water to the plants?"
],
"bridge":[
"What later device did Marek add to the equipment that carries water to his vegetables?",
"How did Marek later modify the watering setup so it could begin automatically?"
],
"near":[
"What controls when Marek's watering begins automatically?",
"Which newer device in Marek's garden setup is responsible for timing?"
],
"far":[
"How does a compass respond to Earth's magnetic field?",
"Why does yeast make bread dough expand?"
]
}
]

LOCK={
"test":TEST,
"parent541":PARENT541,
"model":MODEL_ID,
"seed":SEED,
"purpose":"WHOLE_TRANSFORMER_GEOMETRY_XRAY",
"retrieval":False,
"single_address":False,
"single_vector_key":False,
"human_memory_id":False,
"token_id_address":False,
"single_head_address":False,
"single_layer_address":False,
"router":False,
"threshold":False,
"fitted_score":False,
"training":False,
"measurements":[
"RESIDUAL_LAYER_GEOMETRY",
"Q_HEAD_GEOMETRY",
"K_HEAD_GEOMETRY",
"V_HEAD_GEOMETRY",
"LAYER_TO_LAYER_MOTION",
"PAIRWISE_RELATIONAL_GEOMETRY",
"WRITE_RELATE_RECALL_GEOMETRY"
]
}

LOCK_SHA=hashlib.sha256(
    json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()
).hexdigest()

print("      TEST542 kilit SHA:",LOCK_SHA)

def ids(text):
    return tok(text,add_special_tokens=True,return_tensors="pt").input_ids.to(DEVICE)

@torch.inference_mode()
def trace(text):
    x=ids(text)
    o=model(
        input_ids=x,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True
    )

    R=[]
    Q=[]
    K=[]
    V=[]

    for l,layer in enumerate(layers):
        # Son token yalnız ölçüm örnekleme noktasıdır.
        # Retrieval adresi değildir.
        hout=o.hidden_states[l+1][0,-1].float().cpu()
        hin=o.hidden_states[l][0,-1]
        hn=layer.input_layernorm(
            hin.to(
                layer.input_layernorm.weight.device,
                layer.input_layernorm.weight.dtype
            )
        )

        q=layer.self_attn.q_proj(hn).view(QH,HD).float().cpu()
        k=layer.self_attn.k_proj(hn).view(KVH,HD).float().cpu()
        v=layer.self_attn.v_proj(hn).view(KVH,HD).float().cpu()

        R.append(hout)
        Q.append(q)
        K.append(k)
        V.append(v)

    del o

    return {
        "R":torch.stack(R),
        "Q":torch.stack(Q),
        "K":torch.stack(K),
        "V":torch.stack(V)
    }

def normalize_rows(x):
    return torch.nn.functional.normalize(x.float(),dim=-1)

def gram(x):
    """
    Tek bir koordinat/key üretmez.
    Bir yıldız kümesinin kendi içindeki göreli açı yapısını çıkarır.
    """
    z=normalize_rows(x)
    return z@z.T

def upper_triangle(g):
    n=g.shape[0]
    ix=torch.triu_indices(n,n,offset=1)
    return g[ix[0],ix[1]]

def gram_similarity(a,b):
    """
    İki yapının tek bir yıldızını eşleştirmez.
    İki bütünsel göreli-geometri matrisinin ne kadar benzer değiştiğini
    diagnostic olarak ölçer.
    Retrieval için kullanılmayacaktır.
    """
    ga=upper_triangle(gram(a))
    gb=upper_triangle(gram(b))
    ga=ga-ga.mean()
    gb=gb-gb.mean()
    den=ga.norm()*gb.norm()
    if float(den)==0:
        return 0.0
    return float(torch.dot(ga,gb)/den)

def layer_geometry_R(t):
    # 32 residual yıldızı: her layer bir yıldız.
    return t["R"]

def head_geometry(t,key,l):
    # Aynı layer içindeki bütün Q veya KV head'leri birlikte yıldız kümesi.
    return t[key][l]

def motion_geometry(t):
    # Katmanlar arası hareket vektörlerinin tümü birlikte.
    return t["R"][1:]-t["R"][:-1]

def cross_layer_distance_structure(t):
    """
    32 layer'ın birbirine göre uzaklık yapısı.
    Tek layer seçilmez.
    """
    z=normalize_rows(t["R"])
    return 1.0-(z@z.T)

def matrix_structure_similarity(A,B):
    a=A.float().reshape(-1)
    b=B.float().reshape(-1)
    a=a-a.mean()
    b=b-b.mean()
    den=a.norm()*b.norm()
    if float(den)==0:
        return 0.0
    return float(torch.dot(a,b)/den)

print("\n[2/10] WRITE → RELATE → RECALL bütünsel izleri çıkarılıyor...")

DATA=[]

for wi,w in enumerate(WORLDS):
    print(f"      Dünya {wi+1}/{len(WORLDS)}")

    W=trace(w["write"])

    # RELATE doğal bağlam içinde oluşuyor.
    L=trace(w["write"]+"\n\nLater:\n"+w["relate"])

    R=[trace(q) for q in w["recall"]]

    # Bridge sorguları doğrudan WRITE ile RELATE arasındaki bağlantıya ihtiyaç duyuyor.
    B=[trace(q) for q in w["bridge"]]

    N=[trace(q) for q in w["near"]]
    F=[trace(q) for q in w["far"]]

    DATA.append({
        "W":W,
        "L":L,
        "R":R,
        "B":B,
        "N":N,
        "F":F
    })

gc.collect()
torch.cuda.empty_cache()

print("\n[3/10] 32-layer residual gökyüzü geometrisi...")

RES={
"WR":[],
"LR":[],
"WB":[],
"LB":[],
"WN":[],
"WF":[],
"CROSS":[]
}

for i,d in enumerate(DATA):
    W=d["W"]
    L=d["L"]

    for r in d["R"]:
        RES["WR"].append(
            gram_similarity(layer_geometry_R(W),layer_geometry_R(r))
        )
        RES["LR"].append(
            gram_similarity(layer_geometry_R(L),layer_geometry_R(r))
        )

    for b in d["B"]:
        RES["WB"].append(
            gram_similarity(layer_geometry_R(W),layer_geometry_R(b))
        )
        RES["LB"].append(
            gram_similarity(layer_geometry_R(L),layer_geometry_R(b))
        )

    for n in d["N"]:
        RES["WN"].append(
            gram_similarity(layer_geometry_R(W),layer_geometry_R(n))
        )

    for f in d["F"]:
        RES["WF"].append(
            gram_similarity(layer_geometry_R(W),layer_geometry_R(f))
        )

    for j,e in enumerate(DATA):
        if i==j:
            continue
        for r in e["R"]:
            RES["CROSS"].append(
                gram_similarity(layer_geometry_R(W),layer_geometry_R(r))
            )

def stats(x):
    a=np.asarray(x,dtype=np.float64)
    return float(a.mean()),float(np.median(a)),float(a.std()),float(a.min()),float(a.max())

for k in RES:
    m,md,s,lo,hi=stats(RES[k])
    print(f"      {k:6s} mean={m:+.6f} median={md:+.6f} std={s:.6f} min={lo:+.6f} max={hi:+.6f}")

print("\n[4/10] Katmanlar-arası hareket/yörünge geometrisi...")

MOTION={k:[] for k in ["WR","LR","WB","LB","WN","WF","CROSS"]}

for i,d in enumerate(DATA):
    W=motion_geometry(d["W"])
    L=motion_geometry(d["L"])

    for r in d["R"]:
        R=motion_geometry(r)
        MOTION["WR"].append(gram_similarity(W,R))
        MOTION["LR"].append(gram_similarity(L,R))

    for b in d["B"]:
        B=motion_geometry(b)
        MOTION["WB"].append(gram_similarity(W,B))
        MOTION["LB"].append(gram_similarity(L,B))

    for n in d["N"]:
        MOTION["WN"].append(
            gram_similarity(W,motion_geometry(n))
        )

    for f in d["F"]:
        MOTION["WF"].append(
            gram_similarity(W,motion_geometry(f))
        )

    for j,e in enumerate(DATA):
        if i==j:
            continue
        for r in e["R"]:
            MOTION["CROSS"].append(
                gram_similarity(W,motion_geometry(r))
            )

for k in MOTION:
    m,md,s,lo,hi=stats(MOTION[k])
    print(f"      {k:6s} mean={m:+.6f} median={md:+.6f} std={s:.6f} min={lo:+.6f} max={hi:+.6f}")

print("\n[5/10] Bütünsel Q-head takımyıldızı...")

QRES={k:[] for k in ["WR","LR","WB","LB","WN","WF","CROSS"]}

for l in range(NL):
    bucket={k:[] for k in QRES}

    for i,d in enumerate(DATA):
        W=d["W"]["Q"][l]
        L=d["L"]["Q"][l]

        for r in d["R"]:
            bucket["WR"].append(gram_similarity(W,r["Q"][l]))
            bucket["LR"].append(gram_similarity(L,r["Q"][l]))

        for b in d["B"]:
            bucket["WB"].append(gram_similarity(W,b["Q"][l]))
            bucket["LB"].append(gram_similarity(L,b["Q"][l]))

        for n in d["N"]:
            bucket["WN"].append(gram_similarity(W,n["Q"][l]))

        for f in d["F"]:
            bucket["WF"].append(gram_similarity(W,f["Q"][l]))

        for j,e in enumerate(DATA):
            if i==j:
                continue
            for r in e["R"]:
                bucket["CROSS"].append(
                    gram_similarity(W,r["Q"][l])
                )

    for k in QRES:
        QRES[k].append(float(np.mean(bucket[k])))

    print(
        f"      L{l:02d} "
        f"WR={QRES['WR'][-1]:+.5f} "
        f"LR={QRES['LR'][-1]:+.5f} "
        f"WB={QRES['WB'][-1]:+.5f} "
        f"LB={QRES['LB'][-1]:+.5f} "
        f"NEAR={QRES['WN'][-1]:+.5f} "
        f"FAR={QRES['WF'][-1]:+.5f} "
        f"CROSS={QRES['CROSS'][-1]:+.5f}"
    )

print("\n[6/10] Bütünsel K/V-head takımyıldızları...")

KVRES={}

for key in ["K","V"]:
    KVRES[key]={k:[] for k in ["WR","LR","WB","LB","WN","WF","CROSS"]}

    print(f"      ---- {key} ----")

    for l in range(NL):
        bucket={k:[] for k in KVRES[key]}

        for i,d in enumerate(DATA):
            W=d["W"][key][l]
            L=d["L"][key][l]

            for r in d["R"]:
                bucket["WR"].append(
                    gram_similarity(W,r[key][l])
                )
                bucket["LR"].append(
                    gram_similarity(L,r[key][l])
                )

            for b in d["B"]:
                bucket["WB"].append(
                    gram_similarity(W,b[key][l])
                )
                bucket["LB"].append(
                    gram_similarity(L,b[key][l])
                )

            for n in d["N"]:
                bucket["WN"].append(
                    gram_similarity(W,n[key][l])
                )

            for f in d["F"]:
                bucket["WF"].append(
                    gram_similarity(W,f[key][l])
                )

            for j,e in enumerate(DATA):
                if i==j:
                    continue

                for r in e["R"]:
                    bucket["CROSS"].append(
                        gram_similarity(W,r[key][l])
                    )

        for k in KVRES[key]:
            KVRES[key][k].append(
                float(np.mean(bucket[k]))
            )

        print(
            f"      L{l:02d} "
            f"WR={KVRES[key]['WR'][-1]:+.5f} "
            f"LR={KVRES[key]['LR'][-1]:+.5f} "
            f"WB={KVRES[key]['WB'][-1]:+.5f} "
            f"LB={KVRES[key]['LB'][-1]:+.5f} "
            f"NEAR={KVRES[key]['WN'][-1]:+.5f} "
            f"FAR={KVRES[key]['WF'][-1]:+.5f} "
            f"CROSS={KVRES[key]['CROSS'][-1]:+.5f}"
        )

print("\n[7/10] 32×32 katman ilişki haritasının bütünsel korunumu...")

MAP={k:[] for k in ["WR","LR","WB","LB","WN","WF","CROSS"]}

for i,d in enumerate(DATA):
    W=cross_layer_distance_structure(d["W"])
    L=cross_layer_distance_structure(d["L"])

    for r in d["R"]:
        R=cross_layer_distance_structure(r)
        MAP["WR"].append(matrix_structure_similarity(W,R))
        MAP["LR"].append(matrix_structure_similarity(L,R))

    for b in d["B"]:
        B=cross_layer_distance_structure(b)
        MAP["WB"].append(matrix_structure_similarity(W,B))
        MAP["LB"].append(matrix_structure_similarity(L,B))

    for n in d["N"]:
        MAP["WN"].append(
            matrix_structure_similarity(
                W,
                cross_layer_distance_structure(n)
            )
        )

    for f in d["F"]:
        MAP["WF"].append(
            matrix_structure_similarity(
                W,
                cross_layer_distance_structure(f)
            )
        )

    for j,e in enumerate(DATA):
        if i==j:
            continue

        for r in e["R"]:
            MAP["CROSS"].append(
                matrix_structure_similarity(
                    W,
                    cross_layer_distance_structure(r)
                )
            )

for k in MAP:
    m,md,s,lo,hi=stats(MAP[k])
    print(
        f"      {k:6s} "
        f"mean={m:+.6f} "
        f"median={md:+.6f} "
        f"std={s:.6f} "
        f"min={lo:+.6f} "
        f"max={hi:+.6f}"
    )

print("\n[8/10] WRITE → RELATE → BRIDGE üçlü geometri kontrolü...")

TRI=[]

for d in DATA:
    W=d["W"]
    L=d["L"]

    wl=gram_similarity(
        layer_geometry_R(W),
        layer_geometry_R(L)
    )

    for b in d["B"]:
        wb=gram_similarity(
            layer_geometry_R(W),
            layer_geometry_R(b)
        )

        lb=gram_similarity(
            layer_geometry_R(L),
            layer_geometry_R(b)
        )

        TRI.append((wl,wb,lb))

A=np.asarray(TRI,dtype=np.float64)

print(f"      WRITE↔RELATE mean : {A[:,0].mean():+.6f}")
print(f"      WRITE↔BRIDGE mean : {A[:,1].mean():+.6f}")
print(f"      RELATE↔BRIDGE mean: {A[:,2].mean():+.6f}")

print("\n[9/10] Protokol denetimi...")

S1=sentinel()
WEIGHT_OK=S0==S1

print("      Weight sentinel                    :","PASS" if WEIGHT_OK else "FAIL")
print("      Trainable tensor                   :",sum(int(p.requires_grad) for p in model.parameters()))
print("      Eğitim / LoRA / optimizer          : YOK")
print("      Retrieval                          : YOK")
print("      İnsan memory ID                    : YASAK / KULLANILMADI")
print("      Harf-rakam memory adresi           : YASAK / KULLANILMADI")
print("      Token-ID memory adresi             : YASAK / KULLANILMADI")
print("      Tek layer adresi                   : YASAK / KULLANILMADI")
print("      Tek head adresi                    : YASAK / KULLANILMADI")
print("      Tek vektör key                     : YASAK / KULLANILMADI")
print("      Max-cosine retrieval               : YASAK / KULLANILMADI")
print("      Semantik label router              : YASAK / KULLANILMADI")
print("      Threshold/lambda                   : YOK")
print("      Gold ile seçim                     : YOK")
print("      Bütünsel geometri                  : X-RAY")
print("      WRITE→RELATE→RECALL                 : ÖLÇÜLDÜ")
print("      Model ağırlıkları                  :","DEĞİŞMEDİ" if WEIGHT_OK else "HATA")

print("\n[10/10] Sonuç...")

SUMMARY={
"test":TEST,
"parent541":PARENT541,
"lock_sha":LOCK_SHA,
"weight_sentinel":WEIGHT_OK,
"retrieval":False,
"human_id":False,
"single_address":False,
"single_vector_key":False,
"residual_geometry":{
    k:float(np.mean(v)) for k,v in RES.items()
},
"motion_geometry":{
    k:float(np.mean(v)) for k,v in MOTION.items()
},
"layer_map_geometry":{
    k:float(np.mean(v)) for k,v in MAP.items()
},
"write_relate_bridge":{
    "WL":float(A[:,0].mean()),
    "WB":float(A[:,1].mean()),
    "LB":float(A[:,2].mean())
},
"verdict":"TEST542_BUTUNSEL_TRANSFORMER_GEOMETRISI_XRAY_TAMAMLANDI"
}

RESULT_SHA=hashlib.sha256(
    json.dumps(
        SUMMARY,
        sort_keys=True,
        separators=(",",":")
    ).encode()
).hexdigest()

print("\n"+"="*154)
print("TEST542 SONUÇ — TRANSFORMER İÇ GÖKYÜZÜ / BÜTÜNSEL GEOMETRİ")
print("="*154)
print("MODEL                         :",MODEL_ID)
print("DÜNYA SAYISI                  :",len(WORLDS))
print("MOD                           : KEŞİF / X-RAY")
print("RETRIEVAL                     : YOK")
print("TEK ADRES                     : YOK")
print("TEK VEKTÖR KEY                : YOK")
print("İNSAN MEMORY ID               : YOK")
print("ROUTER                        : YOK")
print("-"*154)

print(
    "RESIDUAL GÖKYÜZÜ WR/CROSS    :",
    f"{np.mean(RES['WR']):+.6f} / {np.mean(RES['CROSS']):+.6f}"
)

print(
    "YÖRÜNGE WR/CROSS             :",
    f"{np.mean(MOTION['WR']):+.6f} / {np.mean(MOTION['CROSS']):+.6f}"
)

print(
    "LAYER-HARİTASI WR/CROSS      :",
    f"{np.mean(MAP['WR']):+.6f} / {np.mean(MAP['CROSS']):+.6f}"
)

print(
    "WRITE↔RELATE↔BRIDGE          :",
    f"{A[:,0].mean():+.6f} / {A[:,1].mean():+.6f} / {A[:,2].mean():+.6f}"
)

print("-"*154)
print("WEIGHT SENTINEL               :","PASS" if WEIGHT_OK else "FAIL")
print("TRAINABLE TENSORS             :",sum(int(p.requires_grad) for p in model.parameters()))
print("TEST541 RESULT SHA            :",PARENT541)
print("TEST542 ÖN KİLİT SHA          :",LOCK_SHA)
print("TEST542 RESULT SHA            :",RESULT_SHA)
print("TOPLAM SÜRE                   :",f"{time.perf_counter()-T0:.2f}s")
print("KARAR                         : TEST542_BUTUNSEL_TRANSFORMER_GEOMETRISI_XRAY_TAMAMLANDI")
print("SONRAKİ ADIM                  : Haritanın doğal yapısını incele; hiçbir koordinatı memory adresine dönüştürme.")
print("="*154)
