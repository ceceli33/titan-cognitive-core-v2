# TEST 430 — FIRST-TOKEN DOSE SWEEP → ENGINE-OFF NATURAL COMPLETION
# TEST429 PROVEN LINEAGE | WRITE L14–L23 ONLY AT FIRST ANSWER TOKEN
# SOURCE-ABSENT WHO | GREEDY | NO CONSTRAINED DECODING | NO TARGET FORCING
import os,sys,gc,json,time,random,hashlib,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
if not torch.cuda.is_available():raise RuntimeError("CUDA REQUIRED")
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False)
SEED=392;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;SEP="\n\n";MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST430")if Path("/content").exists()else Path("/tmp/AKBASCORE_TEST430");ROOT.mkdir(parents=True,exist_ok=True);OUT=ROOT/"TEST430_SUMMARY.json"
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WHO="Who planted the Turkish flag at the base of the Golden Gate Bridge?\nAnswer:"
TARGET="Leyla Demir";SOURCE="Mustafa Akbaş"
DOSES=[0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,1.00];MAX_NEW=24
NEUTRAL=[
"Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.","The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.",
"A tired teacher closed the green door of the library.","Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.","Hiro Sato opened a tiny bakery beside the river.",
"An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.","The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.",
"The pilot announced a short delay because of fog.","Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.","Mei Lin delivered the package to the wrong address.",
"Marek Novak placed the glass bottle beneath the wooden bench.","Sara Ibrahim carried a red notebook into the quiet classroom.","Noah Schmidt repaired the small radio beside the kitchen window.","Yuki Mori left a paper envelope near the station entrance.",
"Amira Hassan moved the ceramic bowl onto the upper shelf.","Peter Novak opened the metal box behind the old theater.","Lucia Costa placed the yellow scarf inside the travel bag.","Daniel Kim carried a black umbrella through the central courtyard."]
SWAP=[
("Jonas Weber","Elena Fischer"),("Priya Nair","Daniel Brooks"),("The small boat","The large boat"),("Omar Haddad","Lucas Martin"),("A tired teacher","A young teacher"),("Sofia Rossi","Nadia Petrova"),("The children","The visitors"),("Liam O'Connor","Peter Novak"),
("Heavy rain","Strong winds"),("Nadia Petrova","Anna Kowalski"),("The farmer","The gardener"),("Hiro Sato","Ravi Kumar"),("An old dog","A young dog"),("Carlos Mendes","Daniel Kim"),("The museum guard","The night porter"),("Fatima Zahra","Amira Hassan"),
("The pilot","The captain"),("Anna Kowalski","Lucia Costa"),("Snow","Rain"),("Ravi Kumar","Omar Haddad"),("The chef","The baker"),("Lucas Martin","Marek Novak"),("A young violinist","An experienced violinist"),("Mei Lin","Yuki Mori"),
("Marek Novak","Jonas Weber"),("Sara Ibrahim","Priya Nair"),("Noah Schmidt","Hiro Sato"),("Yuki Mori","Mei Lin"),("Amira Hassan","Fatima Zahra"),("Peter Novak","Carlos Mendes"),("Lucia Costa","Sofia Rossi"),("Daniel Kim","Liam O'Connor")]
ALT=[]
for s,(a,b) in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b;ALT.append(b+s[len(a):])
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
TRAIN=list(range(24));LAYERS=list(range(28));RIDGES=[1e-4,1e-3,1e-2,.05,.1,.5,1.,5.];WRITE_LAYERS=list(range(14,24))
def clean():gc.collect();torch.cuda.empty_cache()
def sha(ps):
 h=hashlib.sha256()
 for p in ps:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   off=j*max(0,n-256)//15;h.update(a[off:off+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
def load(mid,expected):
 print("LOAD:",mid);t=time.time();tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2]);kw={"dtype"if tv>=(4,56)else"torch_dtype":DTYPE}
 tok=AutoTokenizer.from_pretrained(mid,use_fast=True);model=AutoModelForCausalLM.from_pretrained(mid,device_map={"":0},attn_implementation="sdpa",**kw).eval()
 for p in model.parameters():p.requires_grad_(False)
 c=model.config;layers=model.model.layers;dims=(len(layers),c.hidden_size,c.num_attention_heads,c.num_key_value_heads,getattr(c,"head_dim",None)or c.hidden_size//c.num_attention_heads)
 if dims!=expected:raise RuntimeError(f"Architecture mismatch: {dims}")
 print("READY:",dims,"|",round(time.time()-t,2),"s");return tok,model,layers
def ids(tok,s):return tok(s,add_special_tokens=False).input_ids
def pad(tok):return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
@torch.inference_mode()
def forge(tok,model,s):
 seq=[pad(tok)]+ids(tok,s+SEP);o=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True);pos=max(0,len(seq)-len(ids(tok,SEP))-1)
 H=[o.hidden_states[L+1][0,pos].float().cpu().contiguous()for L in range(len(model.model.layers))];del o;return H
def unit(x):return x.float()/x.float().norm().clamp_min(1e-12)
def cos(a,b):return float(F.cosine_similarity(a.float().reshape(-1),b.float().reshape(-1),dim=0,eps=1e-12))
def packet(A,B,layers):return [unit(B[L]-A[L])for L in layers]
def catpacket(p):return torch.cat([unit(x)for x in p])
def gfp(x,bank,refs):return torch.tensor([cos(catpacket(x),catpacket(bank[j]))for j in refs],dtype=torch.float32)
def ridge(X,Y,lam):
 X=X.double();Y=Y.double();mx=X.mean(0);my=Y.mean(0);A=X-mx;B=Y-my;xx=A.T@A;xy=A.T@B;s=max(1e-12,float(xx.trace()/max(1,xx.shape[0])));W=torch.linalg.solve(xx+lam*s*torch.eye(xx.shape[0],dtype=torch.float64),xy);return W.float(),mx.float(),my.float()
def pred(x,f):W,mx,my=f;return (x.float()-mx)@W+my
def fit_global(MP,QP,refs,lam):
 GM=torch.stack([gfp(MP[i],MP,refs)for i in refs]);GQ=torch.stack([gfp(QP[i],QP,refs)for i in refs]);return ridge(GM,GQ,lam)
def global_qsolve(target,QP,refs,lam):
 C=torch.stack([catpacket(QP[j])for j in refs]);G=C@C.T;a=torch.linalg.solve(G.T@G+lam*torch.eye(len(refs)),G.T@target.float())
 dirs={L:unit(sum((a[k]*QP[j][L]for k,j in enumerate(refs)),torch.zeros_like(QP[refs[0]][L])))for L in LAYERS};return dirs,a
def loo_score(MP,QP,lam):
 fc=[];lc=[]
 for hold in TRAIN:
  refs=[j for j in TRAIN if j!=hold];fit=fit_global(MP,QP,refs,lam);qh=pred(gfp(MP[hold],MP,refs),fit);qt=gfp(QP[hold],QP,refs);dirs,_=global_qsolve(qh,QP,refs,lam);fc.append(cos(qh,qt));lc.append(sum(cos(dirs[L],QP[hold][L])for L in LAYERS)/28)
 return sum(fc)/len(fc),sum(lc)/len(lc)
print("="*128);print("TEST 430 — FIRST-TOKEN DOSE SWEEP → ENGINE-OFF NATURAL COMPLETION");print("TEST429 PROVEN LINEAGE | SOURCE-ABSENT WHO | GREEDY | NO TARGET FORCING");print("="*128)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/10] MISTRAL — TEST429 EXACT PACKETS")
mt,mm,ml=load(MID,(32,4096,32,8,128));mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight];M0=sha(mp);MH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):MH.append((forge(mt,mm,a),forge(mt,mm,b)));print(f" MISTRAL {i+1}/32",end="\r")
MFINAL=(forge(mt,mm,FACT),forge(mt,mm,CF));print()
if sha(mp)!=M0:raise RuntimeError("Mistral sentinel changed")
del mt,mm,ml,mp;clean()
print("[2/10] QWEN — TEST429 EXACT PACKETS")
qt,qm,ql=load(QID,(28,3584,28,4,128));qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight];Q0=sha(qp);QH=[]
for i,(a,b) in enumerate(zip(NEUTRAL,ALT)):QH.append((forge(qt,qm,a),forge(qt,qm,b)));print(f" QWEN {i+1}/32",end="\r")
QFINAL=(forge(qt,qm,FACT),forge(qt,qm,CF));print("\n TRAIN=24 | VAL=8 | FINAL FACT/CF NEVER USED FOR FIT/SELECTION")
MP=[packet(a,b,LMAP)for a,b in MH];QP=[packet(a,b,LAYERS)for a,b in QH];MF=packet(MFINAL[0],MFINAL[1],LMAP);QF=packet(QFINAL[0],QFINAL[1],LAYERS)
print("[3/10] TRAIN-ONLY GLOBAL RIDGE")
CV=[]
for lam in RIDGES:
 fc,lc=loo_score(MP,QP,lam);CV.append({"lambda":lam,"fp_cos":fc,"local_cos":lc});print(f" λ={lam:<6g} FP={fc:+.6f} LOCAL={lc:+.6f}")
LAM=max(CV,key=lambda x:(x["local_cos"],x["fp_cos"]))["lambda"];print(" SELECTED λ=",LAM)
print("[4/10] FINAL GLOBAL SOLVE — FROZEN")
FIT=fit_global(MP,QP,TRAIN,LAM);QHAT=pred(gfp(MF,MP,TRAIN),FIT);QTRUE=gfp(QF,QP,TRAIN);DIRS,COEF=global_qsolve(QHAT,QP,TRAIN,LAM);FFP=cos(QHAT,QTRUE);FLOCAL=sum(cos(DIRS[L],QF[L])for L in LAYERS)/28
print(f" FINAL FP={FFP:+.6f} LOCAL={FLOCAL:+.6f} COEF_NORM={float(COEF.norm()):.6f}")
MAG={L:sum(float((QH[i][1][L]-QH[i][0][L]).norm())for i in TRAIN)/len(TRAIN)for L in LAYERS}
print("[5/10] SOURCE-ABSENT WHO")
PROMPT=torch.tensor([[pad(qt)]+ids(qt,WHO)],device=DEV);print(f" WHO TOKENS={PROMPT.shape[1]} | SOURCE FACT PRESENT=NO")
def first_logits(dose):
 hooks=[];pos=PROMPT.shape[1]-1
 try:
  if dose!=0:
   for L in WRITE_LAYERS:
    d=(DIRS[L]*MAG[L]*dose).to(DEV)
    def inj(m,i,o,d=d,pos=pos):
     if isinstance(o,tuple):
      h=o[0].clone();h[:,pos,:]+=d.to(h.dtype);return (h,)+o[1:]
     h=o.clone();h[:,pos,:]+=d.to(h.dtype);return h
    hooks.append(ql[L].register_forward_hook(inj))
  with torch.inference_mode():o=qm(input_ids=PROMPT,use_cache=False,return_dict=True)
  logits=o.logits[0,-1].detach().float().cpu();del o;return logits
 finally:
  for h in hooks:
   try:h.remove()
   except:pass
def complete_from_first(first_id):
 seq=torch.cat([PROMPT,torch.tensor([[first_id]],device=DEV)],dim=1);gen=[first_id]
 for _ in range(MAX_NEW-1):
  with torch.inference_mode():o=qm(input_ids=seq,use_cache=False,return_dict=True)
  nxt=int(torch.argmax(o.logits[0,-1].float()).item());del o;gen.append(nxt);seq=torch.cat([seq,torch.tensor([[nxt]],device=DEV)],dim=1)
  if nxt==qt.eos_token_id:break
 return gen,qt.decode(gen,skip_special_tokens=True)
def assay(dose):
 logits=first_logits(dose);top=torch.topk(logits,10);first=int(top.indices[0]);gen,text=complete_from_first(first)
 return {"dose":dose,"first_id":first,"first_token":qt.decode([first],skip_special_tokens=False),"text":text,"target":TARGET.lower()in text.lower(),"source":SOURCE.lower()in text.lower(),"top10":[{"id":int(i),"token":qt.decode([int(i)],skip_special_tokens=False),"logit":float(v)}for v,i in zip(top.values,top.indices)]}
print("[6/10] BASE — ENGINE OFF")
BASE=assay(0.0);print(" BASE FIRST:",repr(BASE["first_token"]));print(" BASE:",repr(BASE["text"]))
print("[7/10] FIRST-TOKEN DOSE SWEEP")
RESULTS=[]
for dose in DOSES:
 r=assay(dose);RESULTS.append(r);print(f" DOSE={dose:.2f} FIRST={r['first_token']!r} TARGET={r['target']} SOURCE={r['source']}");print("  TEXT:",repr(r["text"]))
print("[8/10] FIRST-TOKEN TRAJECTORY")
for r in RESULTS:
 print(f" {r['dose']:.2f}x → {r['first_token']!r} | TOP5:"," | ".join(f"{x['token']!r}:{x['logit']:+.3f}"for x in r["top10"][:5]))
print("[9/10] BEHAVIORAL RESULT")
HITS=[r for r in RESULTS if r["target"]]
if HITS:
 FIRST_HIT=HITS[0];VERDICT="SOURCE_ABSENT_FIRST_TOKEN_PULSE_FREE_GENERATION_RETRIEVAL_DETECTED"
 print(f" TARGET HIT: dose={FIRST_HIT['dose']:.2f} |",repr(FIRST_HIT["text"]))
else:
 FIRST_HIT=None;VERDICT="FIRST_TOKEN_PULSE_DOSE_SWEEP_NO_TARGET_RETRIEVAL"
 print(" TARGET HIT: NONE")
print("[10/10] VERDICT + INTEGRITY")
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Qwen not frozen")
REPORT={"test":430,"title":"First-Token Dose Sweep → Engine-Off Natural Completion","seed":SEED,"source_model":MID,"target_model":QID,"fact":FACT,"counterfactual":CF,"who_query":WHO,"target_answer":TARGET,"doses":DOSES,"max_new_tokens":MAX_NEW,"train_pairs":24,"layer_map":LMAP,"write_layers":WRITE_LAYERS,"selected_lambda":LAM,"cross_validation":CV,"global_final":{"fingerprint_cos":FFP,"local_mean":FLOCAL,"coef_norm":float(COEF.norm())},"base":BASE,"dose_results":RESULTS,"first_target_hit":FIRST_HIT,"verdict":VERDICT,"integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1,"pass":Q0==Q1},"notes":["TEST429 proven extraction, TRAIN-only ridge, global solve and L14-L23 write are preserved.","Each dose is an independent branch from the identical source-absent WHO prompt.","Residual steering is active only for the forward pass that selects the first answer token.","Immediately after the first token, all steering hooks are removed and completion is natural greedy Qwen generation.","No constrained decoding, target-token forcing, vocabulary masking, beam reranking or post-hoc replacement is used.","Dose sweep is fixed in advance: 0.10 through 0.80 plus 1.00.","FINAL FACT/CF remain evaluation-only. No model weights are modified."]}
OUT.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print(" VERDICT:",VERDICT);print("="*128);print("TEST 430 — COMPLETE");print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS");print("SUMMARY:",OUT);print("="*128)
