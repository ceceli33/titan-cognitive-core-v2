# ==================================================================================================
# AKBASCORE · TEST 272
# MULTI-DOMAIN MATCHED-COUNTERFACTUAL TRANSFER MATRIX
# FROZEN SOURCE RESIDUAL SUBSPACE → HELD-OUT DOMAINS
# CORRECT−MATCHED CONTROL · NO STEERING · NO TRAINING · STRICT SOURCE→TARGET ISOLATION
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
SEED=272;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;MAX_NEW=64
ROOT=Path("/content/AKBASCORE_TEST272") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST272");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
SOURCE_FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
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
TARGETS={
"FICTION":{
"correct":"Elena Voss stored the cobalt prism beneath the Ardent Observatory.",
"control":"Elena Voss stored the amber prism beneath the Ardent Observatory.",
"query":"What did Elena Voss store beneath the Ardent Observatory?","expected":"cobalt prism","control_expected":"amber prism"},
"SCIENCE":{
"correct":"The synthetic alloy Velorium reaches superconductivity at 173 kelvin.",
"control":"The synthetic alloy Velorium reaches superconductivity at 241 kelvin.",
"query":"At what temperature does Velorium reach superconductivity?","expected":"173 kelvin","control_expected":"241 kelvin"},
"TEMPORAL":{
"correct":"The Orpheus probe entered lunar orbit before the Selene probe entered lunar orbit.",
"control":"The Selene probe entered lunar orbit before the Orpheus probe entered lunar orbit.",
"query":"Which probe entered lunar orbit first?","expected":"Orpheus probe","control_expected":"Selene probe"},
"TECHNICAL":{
"correct":"Protocol ZX-41 assigns channel seven to the thermal calibration stream.",
"control":"Protocol ZX-41 assigns channel three to the thermal calibration stream.",
"query":"Which channel does Protocol ZX-41 assign to the thermal calibration stream?","expected":"channel seven","control_expected":"channel three"}}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 272 — MULTI-DOMAIN MATCHED-COUNTERFACTUAL TRANSFER MATRIX");print("="*110);print("START:",START)
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
def enc_info(fact,q):return enc_text(f"Information: {fact}\n\nQuestion: {q}")
@torch.inference_mode()
def capture(e):
    o=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def capture_text(x):return capture(enc_text(x))
@torch.inference_mode()
def generate(fact,q):
    e=enc_info(fact,q);n=e.input_ids.shape[1]
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
    z=[cos(BV[L],unit(torch.stack([BD[i][L] for i in range(8) if i!=j]).mean(0))) for j in range(8)]
    LOO.append(float(np.mean(z)))
print(f"LOO · L19={LOO[19]:.5f} L25={LOO[25]:.5f} L27={LOO[27]:.5f}")
AXV={**RV,"BINDING":BV}
print("[5/10] SOURCE RESIDUAL BASIS")
SRC_RES={q:[] for q in SOURCE_Q}
for q,text in SOURCE_Q.items():
    null=capture(enc_text(text));nat=capture(enc_info(SOURCE_FACT,text))
    for L in range(TOTAL):
        N=nat[L]-null[L];X=torch.stack([AXV[a][L] for a in AXES],dim=1).float();c=torch.linalg.lstsq(X,N.unsqueeze(1)).solution[:,0]
        SRC_RES[q].append(N-X@c)
SOURCE_BASIS={};SOURCE_STATS={}
for L in range(TOTAL):
    U=torch.stack([unit(SRC_RES[q][L]) for q in SOURCE_Q],dim=0);_,S,Vh=torch.linalg.svd(U,full_matrices=False)
    var=(S*S)/(S*S).sum().clamp_min(EPS);cum=torch.cumsum(var,0);k90=int((cum>=.90).nonzero(as_tuple=False)[0].item()+1)
    SOURCE_BASIS[L]=Vh[:k90].T.detach().clone()
    SOURCE_STATS[f"L{L:02d}"]={"k90":k90,"pc1":float(var[0]),"pc12":float(var[:2].sum())}
for L in [19,21,23,25,26,27]:
    x=SOURCE_STATS[f"L{L:02d}"];print(f"L{L:02d} FROZEN · k90={x['k90']} PC1={x['pc1']*100:.2f}% PC1+2={x['pc12']*100:.2f}%")
SOURCE_FP={L:hashlib.sha256(SOURCE_BASIS[L].cpu().numpy().tobytes()).hexdigest() for L in range(TOTAL)}
print("[6/10] SOURCE BASIS SEALED · TARGETS UNSEEN")
print("[7/10] MATCHED TARGET CAPTURE + BEHAVIOR")
XR={};BEHAV={}
for name,d in TARGETS.items():
    C=capture(enc_info(d["correct"],d["query"]));M=capture(enc_info(d["control"],d["query"]));XR[name]={"CORRECT":C,"CONTROL":M}
    BEHAV[name]={"CORRECT":generate(d["correct"],d["query"]),"CONTROL":generate(d["control"],d["query"])}
    print(f"\n{name} · expected={d['expected']} / control={d['control_expected']}")
    print("CORRECT:",BEHAV[name]["CORRECT"]);print("CONTROL:",BEHAV[name]["CONTROL"])
print("\n[8/10] FROZEN SOURCE → MATCHED-DELTA TRANSFER")
TRANSFER={}
for name in TARGETS:
    TRANSFER[name]=[]
    for L in range(TOTAL):
        D=XR[name]["CORRECT"][L]-XR[name]["CONTROL"][L];B=SOURCE_BASIS[L];u=unit(D);P=B@(B.T@u)
        TRANSFER[name].append({"delta_norm":float(D.norm()),"projection":float(P.norm()),"captured_energy":float(torch.dot(P,P)),"pc1_abs_cos":abs(cos(D,B[:,0])),"source_k90":B.shape[1]})
for L in [0,5,10,15,19,21,23,25,26,27]:
    v=[TRANSFER[n][L]["projection"] for n in TARGETS];e=[TRANSFER[n][L]["captured_energy"] for n in TARGETS]
    print(f"L{L:02d} PROJ={np.mean(v):.4f} ENERGY={np.mean(e)*100:.2f}% · "+" ".join(f"{n}={TRANSFER[n][L]['projection']:.3f}" for n in TARGETS))
print("[9/10] CROSS-DOMAIN MATCHED-DELTA GEOMETRY")
GEOM={}
names=list(TARGETS)
for L in range(TOTAL):
    U=torch.stack([unit(XR[n]["CORRECT"][L]-XR[n]["CONTROL"][L]) for n in names],dim=0);_,S,Vh=torch.linalg.svd(U,full_matrices=False)
    var=(S*S)/(S*S).sum().clamp_min(EPS);pair=[cos(U[i],U[j]) for i in range(4) for j in range(i+1,4)]
    GEOM[f"L{L:02d}"]={"mean_pair_cos":float(np.mean(pair)),"pc1":float(var[0]),"pc12":float(var[:2].sum()),"pc123":float(var[:3].sum())}
for L in [19,21,23,25,26,27]:
    g=GEOM[f"L{L:02d}"];print(f"L{L:02d} PAIR={g['mean_pair_cos']:+.4f} PC1={g['pc1']*100:.2f}% PC1+2={g['pc12']*100:.2f}% PC1+2+3={g['pc123']*100:.2f}%")
print("[10/10] ISOLATION / INTEGRITY / SEAL")
if any(hashlib.sha256(SOURCE_BASIS[L].cpu().numpy().tobytes()).hexdigest()!=SOURCE_FP[L] for L in range(TOTAL)):raise RuntimeError("Source basis mutated.")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
SUMMARY={}
for L in [19,21,23,25,26,27]:
    v=[TRANSFER[n][L]["projection"] for n in TARGETS];e=[TRANSFER[n][L]["captured_energy"] for n in TARGETS]
    SUMMARY[f"L{L:02d}"]={"k":SOURCE_STATS[f"L{L:02d}"]["k90"],"projection_mean":float(np.mean(v)),"projection_min":float(np.min(v)),"projection_max":float(np.max(v)),"captured_energy_mean":float(np.mean(e)),"cross_domain_pair":GEOM[f"L{L:02d}"]["mean_pair_cos"]}
    x=SUMMARY[f"L{L:02d}"];print(f"L{L:02d} · K={x['k']} · PROJ={x['projection_mean']:.4f} · CAPTURED={x['captured_energy_mean']*100:.2f}% · PAIR={x['cross_domain_pair']:+.4f}")
R={"schema":"akbascore.test272.v1","test":"TEST 272","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does the frozen Golden-Gate residual subspace transfer to relation-sensitive matched counterfactual deltas across unrelated semantic domains?",
"source_fact":SOURCE_FACT,"source_queries":SOURCE_Q,"targets":TARGETS,"axes":AXES,"role_pairs":ROLE_PAIRS,"binding_pairs":BIND_PAIRS,"binding_loo":LOO,
"method":"Source basis is built and frozen before target processing exactly as in TEST271. Each held-out domain uses CORRECT and lexically/structurally matched CONTROL information with the same query. Target delta=CORRECT-CONTROL, reducing generic Information/context effects. No target data updates the source basis.",
"source_stats":SOURCE_STATS,"behavior":BEHAV,"transfer":TRANSFER,"cross_domain_geometry":GEOM,"summary":SUMMARY,
"integrity":{"strict_source_target_order":True,"source_basis_frozen_before_targets":True,"source_basis_unchanged":True,"steering":False,"training":False,"weight_update":False,"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T272-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
o=["="*110,"TEST 272 — MULTI-DOMAIN MATCHED-COUNTERFACTUAL TRANSFER MATRIX","="*110]
for n in TARGETS:o+=["",f"[{n}] CORRECT: {BEHAV[n]['CORRECT']}",f"[{n}] CONTROL: {BEHAV[n]['CONTROL']}"]
o.append("")
for L in [19,21,23,25,26,27]:
    x=SUMMARY[f"L{L:02d}"];o.append(f"L{L:02d} K={x['k']} PROJ={x['projection_mean']:.6f} CAPTURED={x['captured_energy_mean']*100:.3f}% PAIR={x['cross_domain_pair']:+.6f}")
o+=["","STRICT SOURCE→TARGET ISOLATION PASS","MATCHED CORRECT−CONTROL · NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 272 COMPLETE")
print("FROZEN SOURCE SUBSPACE → MULTI-DOMAIN MATCHED COUNTERFACTUALS")
print("STRICT ISOLATION · NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)
