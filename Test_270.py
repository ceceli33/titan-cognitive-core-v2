# ==================================================================================================
# AKBASCORE · TEST 270
# RESIDUAL CONSENSUS DISCOVERY
# NATURAL−NULL → REMOVE span{SUBJECT,ACTION,OBJECT,LOCATION,BINDING}
# RESIDUAL CONSENSUS + SVD SUBSPACE + LEAVE-ONE-QUERY-OUT · NO STEERING · NO TRAINING
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
SEED=270;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct";SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;MAX_NEW=64
ROOT=Path("/content/AKBASCORE_TEST270") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST270");ROOT.mkdir(parents=True,exist_ok=True)
START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
ROLES=["SUBJECT","ACTION","OBJECT","LOCATION"];AXES=ROLES+["BINDING"]
ROLE_PAIRS={
"SUBJECT":[
(FACT,"Daniel Carter planted the Turkish flag at the base of the Golden Gate Bridge."),
("Mustafa Akbaş placed the silver key beside the oak tree.","Daniel Carter placed the silver key beside the oak tree."),
("Mustafa Akbaş carried the blue book into the stone house.","Daniel Carter carried the blue book into the stone house."),
("Mustafa Akbaş left the copper coin beside the wooden gate.","Daniel Carter left the copper coin beside the wooden gate.")],
"ACTION":[
(FACT,"Mustafa Akbaş removed the Turkish flag at the base of the Golden Gate Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed removed the silver key beside the oak tree."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks removed the blue book from the stone house."),
("Nora Hayes placed the copper coin beside the wooden gate.","Nora Hayes removed the copper coin from beside the wooden gate.")],
"OBJECT":[
(FACT,"Mustafa Akbaş planted the Canadian flag at the base of the Golden Gate Bridge."),
("Emma Reed placed the silver key beside the oak tree.","Emma Reed placed the copper coin beside the oak tree."),
("Liam Brooks carried the blue book into the stone house.","Liam Brooks carried the red box into the stone house."),
("Nora Hayes left the glass bottle beside the wooden gate.","Nora Hayes left the paper envelope beside the wooden gate.")],
"LOCATION":[
(FACT,"Mustafa Akbaş planted the Turkish flag at the base of the Brooklyn Bridge."),
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
QUERIES={
"WHO":"Who planted the Turkish flag at the base of the Golden Gate Bridge?",
"WHAT":"What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?",
"WHERE":"Where did Mustafa Akbaş plant the Turkish flag?",
"ACTION":"What did Mustafa Akbaş do with the Turkish flag at the base of the Golden Gate Bridge?"}
EXPECTED={"WHO":"Mustafa Akbaş","WHAT":"Turkish flag","WHERE":"base of the Golden Gate Bridge","ACTION":"planted"}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
print("="*110);print("TEST 270 — RESIDUAL CONSENSUS DISCOVERY");print("="*110);print("START:",START)
print("[1/9] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print("[2/9] WEIGHT SENTINEL")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();print("FP:",[f"{x:.4f}" for x in FP0])
def enc_text(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
def enc_query(q,natural=False):return enc_text(f"Information: {FACT}\n\nQuestion: {q}" if natural else q)
@torch.inference_mode()
def capture(e):
    o=model(**e,use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def capture_text(x):return capture(enc_text(x))
@torch.inference_mode()
def generate(q,natural=False):
    e=enc_query(q,natural);n=e.input_ids.shape[1]
    o=model.generate(**e,max_new_tokens=MAX_NEW,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id,eos_token_id=tok.eos_token_id)
    return tok.decode(o[0,n:],skip_special_tokens=True).strip()
print("[3/9] ROLE AXES")
RV={}
for r in ROLES:
    ds=[]
    for a,b in ROLE_PAIRS[r]:
        A=capture_text(a);B=capture_text(b);ds.append([A[L]-B[L] for L in range(TOTAL)])
    M=[torch.stack([d[L] for d in ds]).mean(0) for L in range(TOTAL)];RV[r]=[unit(x) for x in M]
    print(f"{r:8s} raw L19={M[19].norm():.3f} L25={M[25].norm():.3f} L27={M[27].norm():.3f}")
print("[4/9] BINDING AXIS")
BD=[]
for i,(a,b) in enumerate(BIND_PAIRS,1):
    A=capture_text(a);B=capture_text(b);BD.append([A[L]-B[L] for L in range(TOTAL)])
    print(f"[{i}/8] Δ19={BD[-1][19].norm():.3f} Δ25={BD[-1][25].norm():.3f} Δ27={BD[-1][27].norm():.3f}")
BV=[unit(torch.stack([d[L] for d in BD]).mean(0)) for L in range(TOTAL)]
LOO=[]
for L in range(TOTAL):
    z=[]
    for j in range(len(BD)):z.append(cos(BV[L],unit(torch.stack([BD[i][L] for i in range(len(BD)) if i!=j]).mean(0))))
    LOO.append(float(np.mean(z)))
print(f"LOO · L19={LOO[19]:.5f} L25={LOO[25]:.5f} L27={LOO[27]:.5f}")
AXV={**RV,"BINDING":BV}
print("[5/9] NATURAL / NULL CAPTURE")
XR={};BEHAV={}
for q,text in QUERIES.items():
    XR[q]={"NULL":capture(enc_query(text,False)),"NATURAL":capture(enc_query(text,True))}
    BEHAV[q]={"NULL":generate(text,False),"NATURAL":generate(text,True)}
    print(f"{q:6s} NATURAL: {BEHAV[q]['NATURAL']}")
print("[6/9] RESIDUAL EXTRACTION")
RES={q:[] for q in QUERIES};DEC={q:[] for q in QUERIES}
for q in QUERIES:
    for L in range(TOTAL):
        N=XR[q]["NATURAL"][L]-XR[q]["NULL"][L];X=torch.stack([AXV[a][L] for a in AXES],dim=1).float()
        c=torch.linalg.lstsq(X,N.unsqueeze(1)).solution[:,0];E=X@c;R=N-E
        n2=float(torch.dot(N,N));r2=float(torch.dot(R,R));RES[q].append(R)
        DEC[q].append({"r2":max(0.,min(1.,1.-r2/max(n2,EPS))),"residual_energy":r2/max(n2,EPS),"natural_norm":float(N.norm()),"residual_norm":float(R.norm()),"orthogonality_max_abs_dot":float(torch.max(torch.abs(X.T@R)).item())})
for L in [19,21,23,25,26,27]:
    print(f"L{L:02d} R² WHO={DEC['WHO'][L]['r2']:.4f} WHAT={DEC['WHAT'][L]['r2']:.4f} WHERE={DEC['WHERE'][L]['r2']:.4f} ACTION={DEC['ACTION'][L]['r2']:.4f}")
print("[7/9] CONSENSUS + SVD SUBSPACE")
QS=list(QUERIES);CONS=[];DISC={}
for L in range(TOTAL):
    U=torch.stack([unit(RES[q][L]) for q in QS],dim=0)
    mean=U.mean(0);cons=unit(mean);CONS.append(cons)
    _,S,Vh=torch.linalg.svd(U,full_matrices=False)
    var=(S*S)/(S*S).sum().clamp_min(EPS);cum=torch.cumsum(var,0)
    k90=int((cum>=.90).nonzero(as_tuple=False)[0].item()+1);k95=int((cum>=.95).nonzero(as_tuple=False)[0].item()+1)
    pair=[cos(U[i],U[j]) for i in range(4) for j in range(i+1,4)]
    qcos={q:cos(RES[q][L],cons) for q in QS}
    DISC[f"L{L:02d}"]={"singular_values":[float(x) for x in S],"variance_fraction":[float(x) for x in var],"cumulative_variance":[float(x) for x in cum],"pc1_variance":float(var[0]),"pc12_variance":float(var[:2].sum()),"pc123_variance":float(var[:3].sum()),"k90":k90,"k95":k95,"mean_pair_cos":float(np.mean(pair)),"consensus_cos":qcos}
for L in [0,5,10,15,19,21,23,25,26,27]:
    d=DISC[f"L{L:02d}"];print(f"L{L:02d} pair={d['mean_pair_cos']:+.4f} PC1={d['pc1_variance']*100:6.2f}% PC1+2={d['pc12_variance']*100:6.2f}% k90={d['k90']} k95={d['k95']} | "+ " ".join(f"{q}={d['consensus_cos'][q]:+.3f}" for q in QS))
print("[8/9] LEAVE-ONE-QUERY-OUT")
LOOQ={}
for L in range(TOTAL):
    rows={}
    for held in QS:
        train=[q for q in QS if q!=held];c=unit(torch.stack([unit(RES[q][L]) for q in train]).mean(0))
        T=torch.stack([unit(RES[q][L]) for q in train],dim=0);_,S,Vh=torch.linalg.svd(T,full_matrices=False)
        h=unit(RES[held][L]);proj=Vh.T@(Vh@h);rows[held]={"consensus_cos":cos(h,c),"train_subspace_projection":float(proj.norm()),"train_pc1_cos":abs(cos(h,Vh[0]))}
    LOOQ[f"L{L:02d}"]=rows
for L in [19,21,23,25,26,27]:
    r=LOOQ[f"L{L:02d}"];print(f"L{L:02d} LOO consensus mean={np.mean([r[q]['consensus_cos'] for q in QS]):+.4f} · train-subspace projection mean={np.mean([r[q]['train_subspace_projection'] for q in QS]):.4f}")
print("[9/9] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
SUMMARY={}
for L in [19,21,23,25,26,27]:
    d=DISC[f"L{L:02d}"];r=LOOQ[f"L{L:02d}"]
    SUMMARY[f"L{L:02d}"]={"mean_pair_cos":d["mean_pair_cos"],"pc1_variance":d["pc1_variance"],"pc12_variance":d["pc12_variance"],"k90":d["k90"],"k95":d["k95"],"loo_consensus_mean":float(np.mean([r[q]["consensus_cos"] for q in QS])),"loo_subspace_projection_mean":float(np.mean([r[q]["train_subspace_projection"] for q in QS]))}
    print(f"L{L:02d} · pair={d['mean_pair_cos']:+.4f} · PC1={d['pc1_variance']*100:.2f}% · PC1+2={d['pc12_variance']*100:.2f}% · k90={d['k90']} · LOO={SUMMARY[f'L{L:02d}']['loo_consensus_mean']:+.4f}")
R={"schema":"akbascore.test270.v1","test":"TEST 270","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does the large natural-readout residual left outside span{S,A,O,L,B} contain a stable query-shared direction or low-dimensional subspace?",
"fact":FACT,"queries":QUERIES,"expected":EXPECTED,"axes":AXES,"role_pairs":ROLE_PAIRS,"binding_pairs":BIND_PAIRS,"binding_loo":LOO,
"method":"For each query/layer N=NATURAL-NULL; remove least-squares projection onto span{S,A,O,L,B}; normalize residuals; measure pairwise cosine, consensus direction, SVD variance spectrum and leave-one-query-out transfer. Discovery only; no steering.",
"behavior":BEHAV,"decomposition":DEC,"discovery":DISC,"leave_one_query_out":LOOQ,"summary":SUMMARY,
"integrity":{"fp_start":FP0,"fp_end":fp(),"steering":False,"training":False,"weight_update":False,"result":"PASS"}}
raw=canon(R);sha=hashlib.sha256(raw).hexdigest();run=f"T270-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
tp=ROOT/f"{run}.txt";o=["="*110,"TEST 270 — RESIDUAL CONSENSUS DISCOVERY","="*110,f"{MODEL_ID} · NO STEERING · NO TRAINING",""]
for L in [19,21,23,25,26,27]:
    x=SUMMARY[f"L{L:02d}"];o.append(f"L{L:02d} PAIR={x['mean_pair_cos']:+.6f} PC1={x['pc1_variance']*100:.3f}% PC1+2={x['pc12_variance']*100:.3f}% K90={x['k90']} K95={x['k95']} LOO={x['loo_consensus_mean']:+.6f} LOO_SUBSPACE={x['loo_subspace_projection_mean']:.6f}")
o+=["","WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 270 COMPLETE")
print("RESIDUAL CONSENSUS + SVD SUBSPACE + LEAVE-ONE-QUERY-OUT")
print("NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)
