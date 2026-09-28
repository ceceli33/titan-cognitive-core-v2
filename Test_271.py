# ==================================================================================================
# AKBASCORE · TEST 271
# FROZEN RESIDUAL SUBSPACE → HELD-OUT FACT TRANSFER
# SOURCE: GOLDEN GATE · TARGET: ELENA VOSS / COBALT PRISM / ARDENT OBSERVATORY
# NO STEERING · NO TRAINING · STRICT SOURCE→TARGET ISOLATION
# ==================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required.")
SEED=271;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;MAX_NEW=64
ROOT=Path("/content/AKBASCORE_TEST271") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST271");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
SOURCE_FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
TARGET_FACT="Elena Voss stored the cobalt prism beneath the Ardent Observatory."
ROLES=["SUBJECT","ACTION","OBJECT","LOCATION"];AXES=ROLES+["BINDING"]
ROLE_PAIRS={
"SUBJECT":[
(SOURCE_FACT,"Daniel Carter planted the Turkish flag at the base of the Golden Gate Bridge."),
("Mustafa Akbaş placed the silver key beside the oak tree.","Daniel Carter placed the silver key beside the oak tree."),
("Mustafa Akbaş carried the blue book into the stone house.","Daniel Carter carried the blue book into the stone house."),
("Mustafa Akbaş left the copper coin beside the wooden gate.","Daniel Carter left the copper coin beside the wooden gate.")],
"ACTION":[
(SOURCE_FACT,"Mustafa Akbaş removed the Turkish flag at the base of the Golden Gate Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed removed the silver key beside the oak tree."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks removed the blue book from the stone house."),
("Nora Hayes placed the copper coin beside the wooden gate.","Nora Hayes removed the copper coin from beside the wooden gate.")],
"OBJECT":[
(SOURCE_FACT,"Mustafa Akbaş planted the Canadian flag at the base of the Golden Gate Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed placed the copper coin beside the oak tree."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks carried the red box into the stone house."),
("Nora Hayes left the glass bottle beside the wooden gate.","Nora Hayes left the paper envelope beside the wooden gate.")],
"LOCATION":[
(SOURCE_FACT,"Mustafa Akbaş planted the Turkish flag at the base of the Brooklyn Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed placed the silver key beside the stone wall."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks carried the blue book into the railway station."),
("Nora Hayes left the copper coin beside the wooden gate.","Nora Hayes left the copper coin beside the garden fountain.")]}
BIND_PAIRS=[
("Emma Reed placed the silver key beside the oak tree. Liam Brooks placed the copper coin beside the stone wall.","Emma Reed placed the copper coin beside the stone wall. Liam Brooks placed the silver key beside the oak tree."),
("Nora Hayes carried the blue book into the stone house. Owen Clark carried the red box into the railway station.","Nora Hayes carried the red box into the railway station. Owen Clark carried the blue book into the stone house."),
("Alice Morgan left the glass bottle beside the wooden gate. Henry Cole left the paper envelope beside the garden fountain.","Alice Morgan left the paper envelope beside the garden fountain. Henry Cole left the glass bottle beside the wooden gate."),
("Sofia Grant placed the brass token near the marble arch. Ethan Blake placed the green notebook near the river bench.","Sofia Grant placed the green notebook near the river bench. Ethan Blake placed the brass token near the marble arch."),
("Maya Stone carried the white package into the north room. Lucas Dean carried the black folder into the south room.","Maya Stone carried the black folder into the south room. Lucas Dean carried the white package into the north room."),
("Clara Hill left the orange card beside the iron fence. Noah Price left the violet ribbon beside the brick column.","Clara Hill left the violet ribbon beside the brick column. Noah Price left the orange card beside the iron fence."),
("Eva Lane placed the ceramic cup near the pine tree. Adam Wells placed the metal ring near the lake shore.","Eva Lane placed the metal ring near the lake shore. Adam Wells placed the ceramic cup near the pine tree."),
("Iris Wood carried the yellow map into the east hall. Leo Hart carried the gray case into the west hall.","Iris Wood carried the gray case into the west hall. Leo Hart carried the yellow map into the east hall.")
]
SOURCE_Q={
"WHO":"Who planted the Turkish flag at the base of the Golden Gate Bridge?",
"WHAT":"What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?",
"WHERE":"Where did Mustafa Akbaş plant the Turkish flag?",
"ACTION":"What did Mustafa Akbaş do with the Turkish flag at the base of the Golden Gate Bridge?"}
TARGET_Q={
"WHO":"Who stored the cobalt prism beneath the Ardent Observatory?",
"WHAT":"What did Elena Voss store beneath the Ardent Observatory?",
"WHERE":"Where did Elena Voss store the cobalt prism?",
"ACTION":"What did Elena Voss do with the cobalt prism beneath the Ardent Observatory?"}
TARGET_EXPECTED={"WHO":"Elena Voss","WHAT":"cobalt prism","WHERE":"beneath the Ardent Observatory","ACTION":"stored"}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 271 — FROZEN RESIDUAL SUBSPACE → HELD-OUT FACT TRANSFER");print("="*110);print("START:",START)
print("[1/10] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print("[2/10] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
def enc_text(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
def enc_query(fact,q,natural=False):return enc_text(f"Information: {fact}\n\nQuestion: {q}" if natural else q)
@torch.inference_mode()
def capture(e):
    o=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def capture_text(x):return capture(enc_text(x))
@torch.inference_mode()
def generate(fact,q,natural=False):
    e=enc_query(fact,q,natural);n=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip()
print("[3/10] SOURCE ROLE AXES")
RV={}
for r in ROLES:
    ds=[]
    for a,b in ROLE_PAIRS[r]:
        A=capture_text(a);B=capture_text(b);ds.append([A[L]-B[L] for L in range(TOTAL)])
    M=[torch.stack([d[L] for d in ds]).mean(0) for L in range(TOTAL)];RV[r]=[unit(x) for x in M]
    print(f"{r:8s} raw L19={M[19].norm():.3f} L25={M[25].norm():.3f} L27={M[27].norm():.3f}")
print("[4/10] SOURCE BINDING AXIS")
BD=[]
for i,(a,b) in enumerate(BIND_PAIRS,1):
    A=capture_text(a);B=capture_text(b);BD.append([A[L]-B[L] for L in range(TOTAL)])
    print(f"[{i}/8] Δ19={BD[-1][19].norm():.3f} Δ25={BD[-1][25].norm():.3f} Δ27={BD[-1][27].norm():.3f}")
BV=[unit(torch.stack([d[L] for d in BD]).mean(0)) for L in range(TOTAL)]
LOO=[]
for L in range(TOTAL):
    z=[]
    for j in range(8):z.append(cos(BV[L],unit(torch.stack([BD[i][L] for i in range(8) if i!=j]).mean(0))))
    LOO.append(float(np.mean(z)))
print(f"LOO · L19={LOO[19]:.5f} L25={LOO[25]:.5f} L27={LOO[27]:.5f}")
AXV={**RV,"BINDING":BV}
print("[5/10] SOURCE RESIDUAL BASIS")
SRC_XR={};SRC_RES={q:[] for q in SOURCE_Q}
for q,text in SOURCE_Q.items():
    null=capture(enc_query(SOURCE_FACT,text,False));nat=capture(enc_query(SOURCE_FACT,text,True));SRC_XR[q]={"NULL":null,"NATURAL":nat}
    for L in range(TOTAL):
        N=nat[L]-null[L];X=torch.stack([AXV[a][L] for a in AXES],dim=1).float();c=torch.linalg.lstsq(X,N.unsqueeze(1)).solution[:,0]
        SRC_RES[q].append(N-X@c)
SOURCE_BASIS={};SOURCE_STATS={}
for L in range(TOTAL):
    U=torch.stack([unit(SRC_RES[q][L]) for q in SOURCE_Q],dim=0);_,S,Vh=torch.linalg.svd(U,full_matrices=False)
    var=(S*S)/(S*S).sum().clamp_min(EPS);cum=torch.cumsum(var,0);k90=int((cum>=.90).nonzero(as_tuple=False)[0].item()+1)
    SOURCE_BASIS[L]=Vh[:k90].T.detach().clone()
    SOURCE_STATS[f"L{L:02d}"]={"k90":k90,"pc1":float(var[0]),"pc12":float(var[:2].sum()),"variance":[float(x) for x in var]}
for L in [19,21,23,25,26,27]:
    x=SOURCE_STATS[f"L{L:02d}"];print(f"L{L:02d} FROZEN · k90={x['k90']} PC1={x['pc1']*100:.2f}% PC1+2={x['pc12']*100:.2f}%")
print("[6/10] FREEZE SOURCE BASIS")
SOURCE_FP={L:hashlib.sha256(SOURCE_BASIS[L].cpu().numpy().tobytes()).hexdigest() for L in range(TOTAL)}
print("SOURCE BASIS SEALED · target has not been processed")
print("[7/10] HELD-OUT TARGET NATURAL / NULL")
TGT_XR={};BEHAV={}
for q,text in TARGET_Q.items():
    null=capture(enc_query(TARGET_FACT,text,False));nat=capture(enc_query(TARGET_FACT,text,True));TGT_XR[q]={"NULL":null,"NATURAL":nat}
    BEHAV[q]={"NULL":generate(TARGET_FACT,text,False),"NATURAL":generate(TARGET_FACT,text,True)}
    print(f"\n{q} · expected={TARGET_EXPECTED[q]}");print("NATURAL:",BEHAV[q]["NATURAL"]);print("NULL   :",BEHAV[q]["NULL"])
print("\n[8/10] HELD-OUT TRANSFER")
TRANSFER={};TGT_RES={q:[] for q in TARGET_Q}
for q in TARGET_Q:
    TRANSFER[q]=[]
    for L in range(TOTAL):
        N=TGT_XR[q]["NATURAL"][L]-TGT_XR[q]["NULL"][L];X=torch.stack([AXV[a][L] for a in AXES],dim=1).float()
        c=torch.linalg.lstsq(X,N.unsqueeze(1)).solution[:,0];E=X@c;R=N-E;TGT_RES[q].append(R)
        B=SOURCE_BASIS[L];P=B@(B.T@unit(R));proj=float(P.norm());capt=proj*proj
        pc1=abs(cos(R,B[:,0]));n2=float(torch.dot(N,N));r2=float(torch.dot(R,R))
        TRANSFER[q].append({"source_subspace_projection":proj,"source_subspace_energy":capt,"source_pc1_abs_cos":pc1,
        "source_axes_r2":max(0.,min(1.,1.-r2/max(n2,EPS))),"target_residual_fraction":r2/max(n2,EPS),"source_k90":B.shape[1]})
for L in [19,21,23,25,26,27]:
    vals=[TRANSFER[q][L]["source_subspace_projection"] for q in TARGET_Q];eng=[TRANSFER[q][L]["source_subspace_energy"] for q in TARGET_Q]
    print(f"L{L:02d} frozen-source projection mean={np.mean(vals):.4f} · energy={np.mean(eng)*100:.2f}% · "+ " ".join(f"{q}={TRANSFER[q][L]['source_subspace_projection']:.3f}" for q in TARGET_Q))
print("[9/10] TARGET INTERNAL CONSISTENCY / SOURCE-vs-TARGET")
TARGET_GEOM={}
for L in range(TOTAL):
    qs=list(TARGET_Q);U=torch.stack([unit(TGT_RES[q][L]) for q in qs],dim=0);_,S,Vh=torch.linalg.svd(U,full_matrices=False)
    var=(S*S)/(S*S).sum().clamp_min(EPS);pair=[cos(U[i],U[j]) for i in range(4) for j in range(i+1,4)]
    source_proj=[TRANSFER[q][L]["source_subspace_projection"] for q in qs]
    TARGET_GEOM[f"L{L:02d}"]={"mean_pair_cos":float(np.mean(pair)),"pc1":float(var[0]),"pc12":float(var[:2].sum()),"source_projection_mean":float(np.mean(source_proj))}
for L in [19,21,23,25,26,27]:
    x=TARGET_GEOM[f"L{L:02d}"];print(f"L{L:02d} TARGET pair={x['mean_pair_cos']:+.4f} PC1={x['pc1']*100:.2f}% PC1+2={x['pc12']*100:.2f}% SOURCE→TARGET={x['source_projection_mean']:.4f}")
print("[10/10] ISOLATION / INTEGRITY / SEAL")
if any(hashlib.sha256(SOURCE_BASIS[L].cpu().numpy().tobytes()).hexdigest()!=SOURCE_FP[L] for L in range(TOTAL)):raise RuntimeError("Source basis mutated.")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
SUMMARY={}
for L in [19,21,23,25,26,27]:
    vals=[TRANSFER[q][L]["source_subspace_projection"] for q in TARGET_Q];eng=[TRANSFER[q][L]["source_subspace_energy"] for q in TARGET_Q]
    SUMMARY[f"L{L:02d}"]={"source_k90":SOURCE_STATS[f"L{L:02d}"]["k90"],"projection_mean":float(np.mean(vals)),"projection_min":float(np.min(vals)),"projection_max":float(np.max(vals)),"captured_energy_mean":float(np.mean(eng)),"target_pair_cos":TARGET_GEOM[f"L{L:02d}"]["mean_pair_cos"],"target_pc1":TARGET_GEOM[f"L{L:02d}"]["pc1"]}
    x=SUMMARY[f"L{L:02d}"];print(f"L{L:02d} · FROZEN SOURCE k={x['source_k90']} · TARGET PROJ={x['projection_mean']:.4f} · CAPTURED={x['captured_energy_mean']*100:.2f}%")
R={"schema":"akbascore.test271.v1","test":"TEST 271","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does a residual subspace discovered only from the Golden Gate source fact transfer to a completely held-out fact with different subject, action, object and location?",
"source_fact":SOURCE_FACT,"target_fact":TARGET_FACT,"source_queries":SOURCE_Q,"target_queries":TARGET_Q,"target_expected":TARGET_EXPECTED,
"axes":AXES,"role_pairs":ROLE_PAIRS,"binding_pairs":BIND_PAIRS,"binding_loo":LOO,"source_stats":SOURCE_STATS,
"method":"Build S/A/O/L/B only from source-era forge. Extract source natural-minus-null residuals, freeze each layer's k90 SVD basis, then process held-out target. Target data never participates in source basis construction. Measure target residual projection into frozen source subspace.",
"behavior":BEHAV,"transfer":TRANSFER,"target_geometry":TARGET_GEOM,"summary":SUMMARY,
"integrity":{"strict_source_target_order":True,"source_basis_frozen_before_target":True,"source_basis_unchanged":True,"steering":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
raw=canon(R);sha=hashlib.sha256(raw).hexdigest();run=f"T271-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
tp=ROOT/f"{run}.txt";o=["="*110,"TEST 271 — FROZEN RESIDUAL SUBSPACE → HELD-OUT FACT TRANSFER","="*110,f"SOURCE: {SOURCE_FACT}",f"TARGET: {TARGET_FACT}",""]
for q in TARGET_Q:o.append(f"[{q}] NATURAL: {BEHAV[q]['NATURAL']}");o.append(f"[{q}] NULL: {BEHAV[q]['NULL']}")
o.append("")
for L in [19,21,23,25,26,27]:
    x=SUMMARY[f"L{L:02d}"];o.append(f"L{L:02d} K={x['source_k90']} PROJ={x['projection_mean']:.6f} CAPTURED={x['captured_energy_mean']*100:.3f}% TARGET_PAIR={x['target_pair_cos']:+.6f} TARGET_PC1={x['target_pc1']*100:.3f}%")
o+=["","STRICT SOURCE→TARGET ISOLATION PASS","NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 271 COMPLETE")
print("FROZEN GOLDEN-GATE RESIDUAL SUBSPACE → HELD-OUT ELENA-VOSS FACT")
print("STRICT SOURCE→TARGET ISOLATION · NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)
