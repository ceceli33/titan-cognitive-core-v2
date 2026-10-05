# TEST485 — AKBASCORE MAM · SEMANTIC ADDRESS SIGNAL UNDER LEXICAL DECOUPLING
# Copyright © 2026 Mustafa Akbaş
# New original AKBASCORE MAM material is source-available under the repository LICENSE.
# Earlier MIT-licensed Titan Cognitive Core / AkbasCore material remains under its original license.
# See the repository LICENSE for complete licensing terms.
#
# FROZEN FOUNDATION : TEST482 Qwen2.5-7B-Instruct · K120/V128/OWN · exact PCA codebook
# DISCOVERY LOCK     : TEST484 -> HMEAN · L03 · CENTER_COS
# PURPOSE            : determine whether TEST484 addressability survives removal of literal object-name overlap.
# PRIMARY            : HMEAN L03 CENTER_COS — locked before TEST485.
# SECONDARY          : HMEAN L03 L2; QK_NATIVE L03 CENTER_COS — diagnostic only.
# CONTROL            : lexical Jaccard.
# NO config search · NO cartridge LM readout · NO router · NO ANN · NO training · NO LoRA · NO gradients.

import os,sys,subprocess,importlib.util,random,re,time,hashlib,json
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("numpy","numpy")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

TEST="485";SEED=485;MODEL_ID="Qwen/Qwen2.5-7B-Instruct";K_DIM=120;V_DIM=128;SEP="\n\n";L=3
DEVICE=torch.device("cuda");os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
torch.set_grad_enabled(False);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

CORPUS=[
"Jonas Weber carried the wooden crate across the quiet market square.",
"Priya Nair wrote a long letter to her cousin in the morning.",
"The small boat drifted slowly toward the rocky shore.",
"Omar Haddad fixed the broken clock in the village school.",
"A tired teacher closed the green door of the library.",
"Sofia Rossi baked fresh bread for the harvest festival.",
"The children watched the kites rising above the hill.",
"Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.",
"Nadia Petrova translated the ancient manuscript into French.",
"The farmer counted the sheep before sunset.",
"Hiro Sato opened a tiny bakery beside the river.",
"An old dog slept under the kitchen table all afternoon.",
"Carlos Mendes washed the windows of his grandmother's house.",
"The museum guard locked the heavy gate at midnight.",
"Fatima Zahra grew tomatoes in the backyard garden.",
"The pilot announced a short delay because of fog.",
"Anna Kowalski found a lost wallet on the bus.",
"Snow covered the mountain village during the night.",
"Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.",
"Lucas Martin cleaned the roof of the barn after the storm.",
"A young violinist practiced scales in the empty hall.",
"Mei Lin delivered the package to the wrong address.",
"Marek Novak placed the glass bottle beneath the wooden bench.",
"Sara Ibrahim carried a red notebook into the quiet classroom.",
"Noah Schmidt repaired the small radio beside the kitchen window.",
"Yuki Mori left a paper envelope near the station entrance.",
"Amira Hassan moved the ceramic bowl onto the upper shelf.",
"Peter Novak opened the metal box behind the old theater.",
"Lucia Costa placed the yellow scarf inside the travel bag.",
"Daniel Kim carried a black umbrella through the central courtyard."
]

RECORDS=[
("F1","emerald barometer","KR-214","cedar archive"),
("F1","porcelain compass","DM-763","marble annex"),
("F1","violet harmonica","FS-408","amber gallery"),
("F1","bamboo chronometer","JN-951","fern vault"),
("F1","obsidian sextant","RW-326","copper archive"),
("F1","linen telescope","AE-875","quartz annex"),
("F1","coral metronome","LP-143","ivory gallery"),
("F1","willow calculator","XM-692","silver vault"),
("F2","turquoise monocle","HC-517","indigo lighthouse"),
("F2","mahogany notebook","VK-280","pearl pavilion"),
("F2","ceramic astrolabe","SB-934","granite lodge"),
("F2","scarlet blueprint","NT-461","willow gallery"),
("F2","opal kaleidoscope","GY-708","maple tower"),
("F2","bronze hourglass","PD-352","coral pavilion"),
("F2","silk manuscript","ZU-819","birch lodge"),
("F2","crystal chime","EF-625","onyx gallery"),
("F3","jade projector","WL-407","hazel chamber"),
("F3","glass clarinet","CQ-586","lagoon studio"),
("F3","canvas diary","MR-172","acorn room"),
("F3","iron medallion","UX-943","poplar hall"),
("F3","pearl camera","BH-650","cobalt chamber"),
("F3","steel accordion","KO-238","canal studio"),
("F3","cotton almanac","YD-714","walnut room"),
("F3","onyx brooch","TG-569","elm hall"),
("F4","ivory receiver","PS-381","summit workshop"),
("F4","cedar plaque","AL-826","valley depot"),
("F4","azure kettle","RF-504","birch observatory"),
("F4","brass caliper","MW-197","shore conservatory"),
("F4","crystal phonograph","DE-648","forest workshop"),
("F4","maple slate","KI-275","delta depot"),
("F4","white lantern","OV-930","clover observatory"),
("F4","golden protractor","XC-412","island conservatory")
]

def source(fam,obj,cid,place):
    if fam=="F1":return f"The {obj} is stored in container {cid}."
    if fam=="F2":return f"Container {cid} contains the {obj}."
    if fam=="F3":return f"The {obj} can be found inside container {cid}."
    if fam=="F4":return f"Inside container {cid} there is the {obj}."
    raise ValueError(fam)

SOURCES=[source(*r) for r in RECORDS]

# TEST485 SEMANTIC QUERY PANEL
# Literal two-word object names are forbidden from the query.
# Descriptions were fixed before any TEST485 model measurement.
DESCRIPTIONS=[
"the green instrument used to measure atmospheric pressure",
"the ceramic instrument used to determine direction",
"the purple handheld instrument played by blowing and drawing air",
"the bamboo device used for precise timekeeping",
"the dark volcanic-stone instrument historically used for celestial navigation",
"the fabric-covered optical instrument for viewing distant objects",
"the reddish device used to keep a steady musical tempo",
"the wooden device used for arithmetic calculations",
"the blue-green single-eye optical lens",
"the dark wooden book used for handwritten notes",
"the pottery instrument historically used to determine celestial position",
"the bright red technical drawing",
"the gemstone-colored optical toy that forms changing symmetrical patterns",
"the metal timer in which sand falls between two glass chambers",
"the smooth-fabric document written by hand",
"the transparent decorative object that produces a ringing sound",
"the green-stone device used to display images on a surface",
"the transparent woodwind instrument played with a single reed",
"the cloth-covered personal book used for daily written entries",
"the metal ornamental disk worn or kept as a keepsake",
"the pale gemstone-colored device used to take photographs",
"the metal bellows-driven musical instrument with keys or buttons",
"the soft-fabric annual reference book containing facts and dates",
"the black gemstone ornamental pin",
"the pale handheld apparatus used to receive transmitted signals",
"the wooden flat sign or commemorative plate",
"the blue vessel used to heat water",
"the metal measuring tool with two jaws for measuring dimensions",
"the transparent historical machine that reproduces recorded sound",
"the wooden flat board used for writing",
"the pale portable light used for illumination",
"the yellow-metal instrument used to measure and draw angles"
]

QT=[
"Which container holds {d}?",
"Where was {d} stored?",
"In which container can I find {d}?",
"Which container should contain {d}?"
]
QUERIES=[QT[i%4].format(d=d) for i,d in enumerate(DESCRIPTIONS)]

# HARD GUARD: exact object name and each full constituent pair must not appear in query.
for i,(_,obj,_,_) in enumerate(RECORDS):
    if obj.casefold() in QUERIES[i].casefold():raise RuntimeError(f"Literal object leak at {i}: {obj}")

LOCK={
"test":TEST,"seed":SEED,"model":MODEL_ID,"k_dim":K_DIM,"v_dim":V_DIM,"layer":L,
"primary":"HMEAN_L03_CENTER_COS","secondary":["HMEAN_L03_L2","QK_NATIVE_L03_CENTER_COS"],
"corpus":CORPUS,"records":RECORDS,"sources":SOURCES,"descriptions":DESCRIPTIONS,"queries":QUERIES
}
LOCK_SHA=hashlib.sha256(json.dumps(LOCK,sort_keys=True,separators=(",",":")).encode()).hexdigest()

print("="*150)
print("TEST485 — AKBASCORE MAM · SEMANTIC ADDRESS SIGNAL UNDER LEXICAL DECOUPLING")
print("32 FROZEN NIRVANA CARTRIDGES · TEST484 CONFIG LOCKED · LITERAL OBJECT NAMES REMOVED")
print("="*150)
print("LOCK SHA:",LOCK_SHA);T0=time.perf_counter()

print("[1/7] Loading frozen Qwen...")
_tv=tuple(int("".join(c for c in x if c.isdigit())or 0) for x in transformers.__version__.split(".")[:2])
DT="dtype" if _tv>=(4,56) else "torch_dtype"
tok=AutoTokenizer.from_pretrained(MODEL_ID,use_fast=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16}).eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
H=cfg.hidden_size;NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads
HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD;NL=len(layers);GQA=NH//NKV
if (NL,H,NH,NKV,HD)!=(28,3584,28,4,128):raise RuntimeError(f"Architecture mismatch: {(NL,H,NH,NKV,HD)}")
if NH%NKV:raise RuntimeError("Unexpected GQA layout.")
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
enc=lambda s:tok(s,add_special_tokens=False).input_ids
print(f"      {MODEL_ID} | {torch.cuda.get_device_name(0)} | {NL}L | BF16 | frozen")

@torch.inference_mode()
def states_from_ids(ids):
    o=model(input_ids=torch.tensor([ids],device=DEVICE),output_hidden_states=True,use_cache=False,return_dict=True)
    Q=[];K=[];V=[];HS=[]
    for l in range(NL):
        h=o.hidden_states[l][0];z=layers[l].input_layernorm(h);a=layers[l].self_attn
        Q.append(a.q_proj(z).contiguous());K.append(a.k_proj(z).contiguous())
        V.append(a.v_proj(z).contiguous());HS.append(h.contiguous())
    return Q,K,V,HS

def forge_full(s):return states_from_ids([PAD]+enc(s+SEP))
def forge(s):
    _,K,V,_=forge_full(s);return K,V

print("[2/7] Building exact frozen TEST482 PCA codebook...")
CB=[];CO=[forge(s) for s in CORPUS]
for l in range(NL):
    e={}
    for j,n in enumerate(("K","V")):
        R=torch.cat([c[j][l][1:] for c in CO]).float().view(-1,NKV,HD);MU=[];BB=[]
        for h in range(NKV):
            X=R[:,h];mu=X.mean(0);_,_,Vh=torch.linalg.svd(X-mu,full_matrices=False)
            m=min(128,Vh.shape[0]);b=Vh[:m].T.contiguous()
            if m<128:b=torch.nn.functional.pad(b,(0,128-m))
            MU.append(mu);BB.append(b)
        e[n]=(torch.stack(MU),torch.stack(BB))
    CB.append(e)
del CO;torch.cuda.empty_cache()
print("      Codebook ready.")

@torch.inference_mode()
def packet_from_raw(K,V):
    out={}
    for n,X,d in (("K",K,K_DIM),("V",V,V_DIM)):
        rows=[]
        for l in range(NL):
            mu,B=CB[l][n]
            coeff=torch.einsum("thi,hid->thd",X[l][1:].float().view(-1,NKV,HD)-mu,B[:,:,:d]).to(torch.bfloat16)
            content=(mu+torch.einsum("thd,hid->thi",coeff.float(),B[:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)
            rows.append(torch.cat([X[l][:1],content]))
        out[n]=rows
    return out["K"],out["V"]

def hmean(x):return x[1:].float().mean(0).cpu()
def qnative(x):return x[1:].float().view(-1,NH,HD).mean(0).reshape(-1).cpu()
def knative(x):
    z=x[1:].float().view(-1,NKV,HD).mean(0)
    return z.repeat_interleave(GQA,dim=0).reshape(-1).cpu()

print("[3/7] Forging frozen cartridge-side address states...")
BH=[];BK=[]
for i,s in enumerate(SOURCES):
    Q,K,V,HS=forge_full(s)
    # packet is deliberately forged too: confirms TEST482 K120/V128 path remains executable/frozen.
    KC,VC=packet_from_raw(K,V)
    BH.append(hmean(HS[L]));BK.append(knative(K[L]))
    del Q,K,V,HS,KC,VC
    print(f"      [{i+1:02d}/32] {RECORDS[i][1]}")
BH=torch.stack(BH).float();BK=torch.stack(BK).float()

print("[4/7] Encoding lexical-decoupled semantic queries...")
QH=[];QQ=[]
for i,q in enumerate(QUERIES):
    Q,K,V,HS=forge_full(q)
    QH.append(hmean(HS[L]));QQ.append(qnative(Q[L]))
    del Q,K,V,HS
QH=torch.stack(QH).float();QQ=torch.stack(QQ).float()

def nr(x):return torch.nn.functional.normalize(x.float(),dim=-1)
def center_cos(q,B):
    c=B.mean(0)
    return nr(B-c)@torch.nn.functional.normalize(q.float()-c,dim=0)
def l2(q,B):return -torch.sum((B.float()-q.float())**2,dim=1)
def qk(q,B):return nr(B)@torch.nn.functional.normalize(q.float(),dim=0)
def rankof(s,gold):
    o=torch.argsort(s,descending=True)
    return int((o==gold).nonzero(as_tuple=False)[0,0])+1
def summ(r):
    a=np.asarray(r,dtype=float)
    return {"R1":float(np.mean(a<=1)),"R5":float(np.mean(a<=5)),"R16":float(np.mean(a<=16)),
            "MRR":float(np.mean(1.0/a)),"MEDR":float(np.median(a))}

def evaluate(name,Q,B,fn,verbose=True):
    ranks=[];marg=[];z=[]
    if verbose:print(f"\n      {name}")
    for i in range(32):
        s=fn(Q[i],B);w=torch.cat([s[:i],s[i+1:]])
        r=rankof(s,i);mg=float(s[i]-w.max());sd=float(w.std(unbiased=False))+1e-12
        ranks.append(r);marg.append(mg);z.append((float(s[i])-float(w.mean()))/sd)
        if verbose:print(f"        [{i:02d}] rank={r:2d} margin={mg:+.6f} | {RECORDS[i][1]}")
    return {"NAME":name,**summ(ranks),"MED_MARGIN":float(np.median(marg)),
            "P05_MARGIN":float(np.quantile(marg,.05)),"MED_Z":float(np.median(z))},ranks

print("[5/7] Running PRE-LOCKED semantic address tests...")
PRIMARY,RP=evaluate("PRIMARY · HMEAN L03 CENTER_COS",QH,BH,center_cos)
SECONDARY,RS=evaluate("SECONDARY · HMEAN L03 L2",QH,BH,l2)
QK,RQ=evaluate("DIAGNOSTIC · QK_NATIVE L03 COS",QQ,BK,qk)

print("\n[6/7] Lexical and permutation controls...")
def words(s):return set(re.findall(r"[a-z0-9]+",s.casefold()))
SW=[words(s) for s in SOURCES]
LEX_R=[];LEX_G=[];LEX_M=[]
for i,qtxt in enumerate(QUERIES):
    q=words(qtxt);scores=np.asarray([len(q&x)/max(1,len(q|x)) for x in SW],dtype=float)
    order=np.argsort(-scores,kind="stable");r=int(np.where(order==i)[0][0])+1
    wrong=np.delete(scores,i);LEX_R.append(r);LEX_G.append(float(scores[i]));LEX_M.append(float(scores[i]-wrong.max()))
    print(f"      [{i:02d}] lexical rank={r:2d} gold={scores[i]:.4f} margin={LEX_M[-1]:+.4f} | {RECORDS[i][1]}")
LEX={"NAME":"LEXICAL_JACCARD",**summ(LEX_R),"MED_GOLD":float(np.median(LEX_G)),"MED_MARGIN":float(np.median(LEX_M))}

rng=np.random.default_rng(SEED+1);perm=np.arange(32);rng.shuffle(perm)
NULL_R=[]
for i in range(32):NULL_R.append(rankof(center_cos(QH[i],BH),int(perm[i])))
NULL={"NAME":"PERMUTATION_NULL",**summ(NULL_R)}

# Stronger lexical diagnostic: remove generic question words and test whether any rare source token survives.
STOP={"the","a","an","which","where","was","is","in","can","i","find","should","container","holds","hold","stored","contain","contains",
      "used","to","for","with","of","and","or","there","be","being","kept","put","device","instrument","object","machine","tool"}
RARE_OVERLAP=[]
for i in range(32):
    q=words(QUERIES[i])-STOP;s=words(SOURCES[i])-STOP
    RARE_OVERLAP.append(sorted(q&s))
print("\n      Same-record residual lexical overlaps:")
for i,x in enumerate(RARE_OVERLAP):print(f"        [{i:02d}] {x}")

print("[7/7] Final falsification verdict...")
# Primary hypothesis is intentionally judged against both chance and lexical control.
# 32-way random R1=3.125%, R5=15.625%. Thresholds are discovery criteria, not inferential p-values.
if PRIMARY["R1"]>=0.50 and PRIMARY["R5"]>=0.75 and PRIMARY["MRR"]>=0.60 and PRIMARY["MRR"]>=LEX["MRR"]+0.15:
    VERDICT="SEMANTIC_MODEL_NATIVE_ADDRESS_SIGNAL_SURVIVES_LEXICAL_DECOUPLING"
elif PRIMARY["R5"]>=0.50 and PRIMARY["MRR"]>=0.35 and PRIMARY["MRR"]>=LEX["MRR"]+0.10:
    VERDICT="PARTIAL_SEMANTIC_ADDRESS_SIGNAL_SURVIVES"
elif PRIMARY["MRR"]>NULL["MRR"]+0.10:
    VERDICT="WEAK_SIGNAL_ABOVE_NULL_BUT_NOT_SEMANTICALLY_SEALED"
else:
    VERDICT="TEST484_SIGNAL_DOES_NOT_SURVIVE_LEXICAL_DECOUPLING"

print("\n"+"="*150)
print("TEST485 FINAL RESULT — AKBASCORE MAM")
print("="*150)
print("FOUNDATION             : TEST482 QWEN K120/V128/OWN")
print("TEST482 BLOB SHA       : ecc630144a44479cb35c432c50eb0d09bc6866d7")
print("TEST484 DISCOVERY LOCK : HMEAN L03 CENTER_COS")
print("CARTRIDGES / QUERIES   : 32 / 32")
print("LITERAL OBJECT NAMES   : REMOVED FROM QUERIES")
print("CONFIG SEARCH          : NONE")
print("LOCK SHA               :",LOCK_SHA)
print("-"*150)
print("PRIMARY                 :",PRIMARY)
print("SECONDARY               :",SECONDARY)
print("QK_NATIVE               :",QK)
print("LEXICAL CONTROL         :",LEX)
print("PERMUTATION NULL        :",NULL)
print("-"*150)
print("VERDICT                 :",VERDICT)
print("INTERPRETATION          : semantic-decoupling falsification only; no scale or sublinear-retrieval claim.")
print("CARTRIDGE LM READOUT    : NOT USED")
print("ROUTER / ANN / GRAPH    : NONE")
print("TRAINING / LoRA         : NONE")
print("MODEL WEIGHT CHANGE     : NONE")
print("NIRVANA MOTOR CHANGE    : NONE")
print("POST-HOC CONFIG TUNING  : NONE")
print(f"TOTAL TEST TIME         : {time.perf_counter()-T0:.2f}s")
print("="*150)
