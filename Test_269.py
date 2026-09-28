# ==================================================================================================
# AKBASCORE · TEST 269
# NATURAL READOUT RESIDUAL DECOMPOSITION
# N_L=NATURAL_L-NULL_L → span{SUBJECT,ACTION,OBJECT,LOCATION,BINDING}
# TRUE LEAST-SQUARES · NO STEERING · NO TRAINING · FROZEN QWEN2.5-7B-INSTRUCT
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
SEED=269;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;MAX_NEW=64
ROOT=Path("/content/AKBASCORE_TEST269") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST269");ROOT.mkdir(parents=True,exist_ok=True)
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
print("="*110);print("TEST 269 — NATURAL READOUT RESIDUAL DECOMPOSITION");print("="*110);print("START:",START)
print("[1/8] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch.")
print(f"OK · {MODEL_ID} · 28L · H={H} · {PDT} · {time.perf_counter()-t:.2f}s")
print("[2/8] WEIGHT SENTINEL")
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
print("[3/8] ROLE AXES")
RV={};ROLE_RAW={}
for r in ROLES:
    ds=[]
    for a,b in ROLE_PAIRS[r]:
        A=capture_text(a);B=capture_text(b);ds.append([A[L]-B[L] for L in range(TOTAL)])
    M=[torch.stack([d[L] for d in ds]).mean(0) for L in range(TOTAL)]
    RV[r]=[unit(x) for x in M];ROLE_RAW[r]=[float(x.norm()) for x in M]
    print(f"{r:8s} raw L19={ROLE_RAW[r][19]:.3f} L25={ROLE_RAW[r][25]:.3f} L27={ROLE_RAW[r][27]:.3f}")
print("[4/8] BINDING AXIS")
BD=[]
for i,(a,b) in enumerate(BIND_PAIRS,1):
    A=capture_text(a);B=capture_text(b);BD.append([A[L]-B[L] for L in range(TOTAL)])
    print(f"[{i}/8] Δ19={BD[-1][19].norm():.3f} Δ25={BD[-1][25].norm():.3f} Δ27={BD[-1][27].norm():.3f}")
BV=[unit(torch.stack([d[L] for d in BD]).mean(0)) for L in range(TOTAL)]
LOO=[]
for L in range(TOTAL):
    z=[]
    for j in range(len(BD)):
        loo=unit(torch.stack([BD[i][L] for i in range(len(BD)) if i!=j]).mean(0));z.append(cos(BV[L],loo))
    LOO.append(float(np.mean(z)))
print(f"LOO · L19={LOO[19]:.5f} L25={LOO[25]:.5f} L27={LOO[27]:.5f}")
AXV={**RV,"BINDING":BV}
print("[5/8] NATURAL / NULL CAPTURE + SANITY")
XR={};BEHAV={}
for q,text in QUERIES.items():
    XR[q]={"NULL":capture(enc_query(text,False)),"NATURAL":capture(enc_query(text,True))}
    BEHAV[q]={"NULL":generate(text,False),"NATURAL":generate(text,True)}
    print(f"\n{q} · expected={EXPECTED[q]}");print("NATURAL:",BEHAV[q]["NATURAL"]);print("NULL   :",BEHAV[q]["NULL"])
print("\n[6/8] TRUE LEAST-SQUARES DECOMPOSITION")
DEC={}
for q in QUERIES:
    DEC[q]=[];print("\n"+q);print("LAYER | N/NULL | RANK | COND      | R²     | EXPL%  | RESID% | cos(E,N) | cos(R,N) | R·B")
    for L in range(TOTAL):
        null=XR[q]["NULL"][L];N=XR[q]["NATURAL"][L]-null
        X=torch.stack([AXV[a][L] for a in AXES],dim=1).float()
        coef=torch.linalg.lstsq(X,N.unsqueeze(1)).solution[:,0];E=X@coef;R=N-E
        n2=float(torch.dot(N,N));e2=float(torch.dot(E,E));r2=float(torch.dot(R,R))
        rank=int(torch.linalg.matrix_rank(X).item());sv=torch.linalg.svdvals(X);cond=float((sv[0]/sv[-1].clamp_min(EPS)).item())
        rs=max(0.,min(1.,1.-r2/max(n2,EPS)))
        row={"layer":L,"natural_relative_norm":float(N.norm()/null.norm().clamp_min(EPS)),"rank":rank,"condition_number":cond,
        "coefficients":{AXES[i]:float(coef[i]) for i in range(len(AXES))},"r2":rs,"explained_energy":e2/max(n2,EPS),"residual_energy":r2/max(n2,EPS),
        "cos_explained_natural":cos(E,N),"cos_residual_natural":cos(R,N),"residual_cos_binding":cos(R,BV[L]),
        "residual_cos_axes":{a:cos(R,AXV[a][L]) for a in AXES},"orthogonality_max_abs_dot":float(torch.max(torch.abs(X.T@R)).item()),
        "natural_norm":float(N.norm()),"explained_norm":float(E.norm()),"residual_norm":float(R.norm())}
        DEC[q].append(row)
        if L in [0,5,10,15,19,21,23,25,26,27]:
            print(f"L{L:02d}   | {row['natural_relative_norm']*100:6.2f}% | {rank:4d} | {cond:9.3e} | {rs:6.4f} | {row['explained_energy']*100:6.2f}% | {row['residual_energy']*100:6.2f}% | {row['cos_explained_natural']:+.4f}   | {row['cos_residual_natural']:+.4f}   | {row['residual_cos_binding']:+.4f}")
print("\n[7/8] CROSS-QUERY RESIDUAL GEOMETRY")
CROSS={}
for L in range(TOTAL):
    rs={};qs=list(QUERIES)
    for q in qs:
        null=XR[q]["NULL"][L];N=XR[q]["NATURAL"][L]-null;X=torch.stack([AXV[a][L] for a in AXES],dim=1).float()
        coef=torch.linalg.lstsq(X,N.unsqueeze(1)).solution[:,0];rs[q]=N-X@coef
    M=np.eye(len(qs),dtype=float)
    for i in range(len(qs)):
        for j in range(i+1,len(qs)):M[i,j]=M[j,i]=cos(rs[qs[i]],rs[qs[j]])
    off=[M[i,j] for i in range(len(qs)) for j in range(i+1,len(qs))]
    CROSS[f"L{L:02d}"]={"queries":qs,"residual_cosine":M.tolist(),"mean_offdiag":float(np.mean(off))}
for L in [19,21,23,25,26,27]:print(f"L{L:02d} residual cross-query mean cosine={CROSS[f'L{L:02d}']['mean_offdiag']:+.4f}")
print("[8/8] INTEGRITY + SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure.")
SUMMARY={}
for q in QUERIES:
    SUMMARY[q]={f"L{L:02d}":{"r2":DEC[q][L]["r2"],"explained_energy":DEC[q][L]["explained_energy"],"residual_energy":DEC[q][L]["residual_energy"],"rank":DEC[q][L]["rank"],"condition_number":DEC[q][L]["condition_number"]} for L in [19,21,23,25,26,27]}
    x=DEC[q][27];print(f"{q:6s} L27 · R²={x['r2']:.4f} · explained={x['explained_energy']*100:.2f}% · residual={x['residual_energy']*100:.2f}% · rank={x['rank']} · cond={x['condition_number']:.3e}")
R={"schema":"akbascore.test269.v1","test":"TEST 269","start":START,"end":utc(),"model":MODEL_ID,
"question":"How much of the naturally readable fact-conditioned displacement lies in the existing SUBJECT/ACTION/OBJECT/LOCATION/BINDING subspace?",
"fact":FACT,"queries":QUERIES,"expected":EXPECTED,"axes":AXES,"role_pairs":ROLE_PAIRS,"binding_pairs":BIND_PAIRS,
"method":"At each query/layer N=NATURAL-NULL. X=[S,A,O,L,B]. Solve min_c ||Xc-N||_2. Explained=Xc; residual=N-explained. No orthogonality assumption.",
"binding_loo":LOO,"behavior":BEHAV,"decomposition":DEC,"cross_query_residual":CROSS,"summary":SUMMARY,
"integrity":{"fp_start":FP0,"fp_end":fp(),"steering":False,"training":False,"weight_update":False,"result":"PASS"}}
raw=canon(R);sha=hashlib.sha256(raw).hexdigest();run=f"T269-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
jp=ROOT/f"{run}.json";jp.write_bytes(json.dumps(R,ensure_ascii=False,sort_keys=True,indent=2).encode())
tp=ROOT/f"{run}.txt";o=["="*110,"TEST 269 — NATURAL READOUT RESIDUAL DECOMPOSITION","="*110,f"{MODEL_ID} · NO STEERING · NO TRAINING",""]
for q in QUERIES:
    o.append(f"[{q}] NATURAL: {BEHAV[q]['NATURAL']}");o.append(f"[{q}] NULL   : {BEHAV[q]['NULL']}")
    for L in [19,21,23,25,26,27]:
        x=DEC[q][L];o.append(f"L{L:02d} R²={x['r2']:.6f} EXPL={x['explained_energy']*100:.3f}% RESID={x['residual_energy']*100:.3f}% RANK={x['rank']} COND={x['condition_number']:.6e}")
    o.append("")
for L in [19,21,23,25,26,27]:o.append(f"L{L:02d} RESIDUAL CROSS-QUERY MEAN COS={CROSS[f'L{L:02d}']['mean_offdiag']:+.6f}")
o+=["","WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(o),encoding="utf-8")
print("="*110);print("TEST 269 COMPLETE")
print("NATURAL−NULL → span{SUBJECT,ACTION,OBJECT,LOCATION,BINDING}")
print("TRUE LEAST-SQUARES · NO STEERING · NO TRAINING · WEIGHT INTEGRITY PASS")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*110)
