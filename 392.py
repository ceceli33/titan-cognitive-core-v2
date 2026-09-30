# TEST 392 — DUAL-MODEL END-TO-END SIGNAL X-RAY · TEST391 BASELINE
import os,sys,gc,time,json,random,hashlib,subprocess,importlib.util
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate")]:
 if importlib.util.find_spec(m)is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import torch,transformers
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as qwen_rope
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False)
SEED=392;random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEV="cuda";DTYPE=torch.bfloat16;MAX_NEW=32;SEP="\n\n";FMT="QUESTION:\n{q}\n\nANSWER:"
MID="mistralai/Mistral-7B-Instruct-v0.3";QID="Qwen/Qwen2.5-7B-Instruct"
ROOT=Path("/content/AKBASCORE_TEST392")if os.path.isdir("/content")else Path("/tmp/AKBASCORE_TEST392")
ROOT.mkdir(parents=True,exist_ok=True);LOG=ROOT/"TEST392.jsonl";SUMMARY=ROOT/"TEST392_SUMMARY.json";LOG.write_text("",encoding="utf-8")
FACT="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge."
CF="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge."
WRONG="Elena Varga stored the silver compass inside the northern archive of Tallinn."
NEUTRAL=[
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
 "Daniel Kim carried a black umbrella through the central courtyard."]
SWAP=[("Jonas Weber","Elena Fischer"),("Priya Nair","Daniel Brooks"),("The small boat","The large boat"),
 ("Omar Haddad","Lucas Martin"),("A tired teacher","A young teacher"),("Sofia Rossi","Nadia Petrova"),
 ("The children","The visitors"),("Liam O'Connor","Peter Novak"),("Heavy rain","Strong winds"),
 ("Nadia Petrova","Anna Kowalski"),("The farmer","The gardener"),("Hiro Sato","Ravi Kumar"),
 ("An old dog","A young dog"),("Carlos Mendes","Daniel Kim"),("The museum guard","The night porter"),
 ("Fatima Zahra","Amira Hassan"),("The pilot","The captain"),("Anna Kowalski","Lucia Costa"),
 ("Snow","Rain"),("Ravi Kumar","Omar Haddad"),("The chef","The baker"),
 ("Lucas Martin","Marek Novak"),("A young violinist","An experienced violinist"),
 ("Mei Lin","Yuki Mori"),("Marek Novak","Jonas Weber"),("Sara Ibrahim","Priya Nair"),
 ("Noah Schmidt","Hiro Sato"),("Yuki Mori","Mei Lin"),("Amira Hassan","Fatima Zahra"),
 ("Peter Novak","Carlos Mendes"),("Lucia Costa","Sofia Rossi"),("Daniel Kim","Liam O'Connor")]
assert len(NEUTRAL)==len(SWAP)==32
ALT=[]
for s,(a,b)in zip(NEUTRAL,SWAP):
 assert s.startswith(a)and a!=b
 ALT.append(b+s[len(a):])
QUESTIONS=[("WHO","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş","Leyla Demir"),
 ("WHAT","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","Turkish flag",None),
 ("WHERE","Where did Mustafa Akbaş plant the Turkish flag?","base of the Golden Gate Bridge",None)]
assert all(FACT not in s and CF not in s and WRONG not in s for s in NEUTRAL+ALT)
def rec(x):
 with LOG.open("a",encoding="utf-8")as f:f.write(json.dumps(x,ensure_ascii=False,default=str)+"\n")
def clean():gc.collect();torch.cuda.empty_cache()
def norm(s):return " ".join("".join(c.casefold()if c.isalnum()else" "for c in str(s)).split())
def hit(s,a):return int(bool(a)and norm(a)in norm(s))
def sha(ps):
 h=hashlib.sha256()
 for p in ps:
  a=p.detach().reshape(-1);n=a.numel()
  for j in range(16):
   off=j*max(0,n-256)//15;h.update(a[off:off+256].float().cpu().numpy().tobytes())
 return h.hexdigest()
def load(mid,expected):
 print("LOAD:",mid);t=time.time()
 tv=tuple(int("".join(c for c in x if c.isdigit())or 0)for x in transformers.__version__.split(".")[:2])
 kw={"dtype"if tv>=(4,56)else"torch_dtype":DTYPE}
 tok=AutoTokenizer.from_pretrained(mid,use_fast=True)
 model=AutoModelForCausalLM.from_pretrained(mid,device_map={"":0},attn_implementation="sdpa",**kw)
 model.eval()
 for p in model.parameters():p.requires_grad_(False)
 c=model.config;layers=model.model.layers
 dims=(len(layers),c.hidden_size,c.num_attention_heads,c.num_key_value_heads,getattr(c,"head_dim",None)or c.hidden_size//c.num_attention_heads)
 if dims!=expected:raise RuntimeError(f"Architecture mismatch {dims}")
 print("READY:",dims,"|",round(time.time()-t,2),"s")
 return tok,model,layers
def ids(tok,s):return tok(s,add_special_tokens=False).input_ids
def pad(tok):return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
@torch.inference_mode()
def forge(tok,model,layers,s):
 seq=[pad(tok)]+ids(tok,s+SEP)
 out=model(input_ids=torch.tensor([seq],device=DEV),use_cache=False,output_hidden_states=True,return_dict=True)
 K=[];V=[]
 for L,layer in enumerate(layers):
  z=layer.input_layernorm(out.hidden_states[L][0]);a=layer.self_attn
  K.append(a.k_proj(z).float().cpu().contiguous());V.append(a.v_proj(z).float().cpu().contiguous())
 del out
 return {"T":len(seq),"K":K,"V":V}
LMAP=[min(31,max(0,round((L+.5)*32/28-.5)))for L in range(28)]
def resize(x,n):
 if x.shape[0]==n:return x.contiguous()
 return F.interpolate(x.T.unsqueeze(0),size=n,mode="linear",align_corners=False)[0].T.contiguous()
def src_heads(f,L,n,kind):
 x=f[kind][LMAP[L]][1:].reshape(-1,8,128).reshape(-1,4,2,128).mean(2)
 return resize(x.reshape(-1,512),n).reshape(n,4,128)
def tgt_heads(f,L,kind):return f[kind][L][1:].reshape(-1,4,128)
def stats():
 return {k:[[{name:torch.zeros(sh,dtype=torch.float64)for name,sh in
  (("sx",(128,)),("sy",(128,)),("xx",(128,128)),("xy",(128,128)),("yy",()))}|{"n":0}
  for h in range(4)]for L in range(28)]for k in("K","V")}
ST=stats();SD=stats();SN=stats()
def add(st,X,Y):
 X=X.double();Y=Y.double()
 for h in range(4):
  a=X[:,h];b=Y[:,h];s=st[h];n=a.shape[0]
  s["sx"]+=a.sum(0);s["sy"]+=b.sum(0);s["xx"]+=a.T@a;s["xy"]+=a.T@b;s["yy"]+=(b*b).sum();s["n"]+=n
def accumulate(m,q):
 n=q["T"]-1
 for kind in("K","V"):
  for L in range(28):add(ST[kind][L],src_heads(m,L,n,kind),tgt_heads(q,L,kind))
def delta_pair(ma,mb,qa,qb,L,kind):
 n=min(qa["T"],qb["T"])-1
 X=src_heads(mb,L,n,kind)-src_heads(ma,L,n,kind)
 ya=resize(tgt_heads(qa,L,kind).reshape(-1,512),n).reshape(n,4,128)
 yb=resize(tgt_heads(qb,L,kind).reshape(-1,512),n).reshape(n,4,128)
 return X,yb-ya
def accumulate_delta(ma,mb,qa,qb,st):
 for kind in("K","V"):
  for L in range(28):
   X,Y=delta_pair(ma,mb,qa,qb,L,kind);add(st[kind][L],X,Y)
def solve(st,mode):
 B={k:[]for k in("K","V")};fit=[]
 for kind in("K","V"):
  for L in range(28):
   group=[]
   for h in range(4):
    s=st[kind][L][h];n=s["n"];mx=s["sx"]/n;my=s["sy"]/n
    xx=s["xx"]-n*torch.outer(mx,mx);xy=s["xy"]-n*torch.outer(mx,my)
    alpha=max(1e-8,float(xx.trace()/128)*.05)
    W=torch.linalg.solve(xx+alpha*torch.eye(128,dtype=torch.float64),xy)
    err=float((s["yy"]-n*(my@my)-2*(W*xy).sum()+(W*(xx@W)).sum()).clamp_min(0))
    var=float((s["yy"]-n*(my@my)).clamp_min(1e-12))
    fit.append({"mode":mode,"kind":kind,"layer":L,"head":h,"R2_train":1-err/var,"alpha":alpha,"n":n})
    group.append((W.float(),mx.float(),my.float()))
   B[kind].append(group)
 return B,fit
def bridge(m,B,n,sink):
 out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=src_heads(m,L,n,kind);Y=[]
   for h in range(4):
    W,mx,my=B[kind][L][h];Y.append((X[:,h]-mx)@W+my)
   arr.append(torch.cat([sink[kind][L],torch.stack(Y,1).reshape(n,512)],0))
  out[kind]=arr
 return out["K"],out["V"],n+1
def predict(m,B,D,n,sink,gain):
 if gain==0:return bridge(m,B,n,sink)
 out={}
 for kind in("K","V"):
  arr=[]
  for L in range(28):
   X=src_heads(m,L,n,kind);Y=[]
   for h in range(4):
    W,mx,my=B[kind][L][h];WD,_,_=D[kind][L][h]
    x=X[:,h];Y.append((x-mx)@W+my+gain*((x-mx)@(WD-W)))
   arr.append(torch.cat([sink[kind][L],torch.stack(Y,1).reshape(n,512)],0))
  out[kind]=arr
 return out["K"],out["V"],n+1
def cos(a,b):
 a=a.float().reshape(-1);b=b.float().reshape(-1)
 return float(F.cosine_similarity(a,b,dim=0,eps=1e-12))
def rel(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
def vecstats(a,b):return {"cos":cos(a,b),"rel":rel(a,b),"norm_ratio":float(a.float().norm()/b.float().norm().clamp_min(1e-12))}
def softstats(a,b):
 p=a.float().softmax(-1);q=b.float().softmax(-1)
 return {"TV":float((p-q).abs().sum(-1).mean()/2),"top1_agreement":float((p.argmax(-1)==q.argmax(-1)).float().mean()),
  "KL_native_to_bridge":float((q*(q.clamp_min(1e-12).log()-p.clamp_min(1e-12).log())).sum(-1).mean())}
def centered(a,b):
 a=a.float();b=b.float()
 return cos(a-a.mean(0,keepdim=True),b-b.mean(0,keepdim=True))
def dna(p,q):
 n=q["T"]-1
 if p[2]!=q["T"]:raise RuntimeError("DNA token length mismatch")
 rows=[]
 for L in range(28):
  for kind in("K","V"):
   x=p[0 if kind=="K"else 1][L][1:].reshape(n,4,128);y=q[kind][L][1:].reshape(n,4,128)
   for h in range(4):
    a=x[:,h];b=y[:,h];z=vecstats(a,b)
    z.update({"layer":L,"head":h,"kind":kind,"centered_cos":centered(a,b),"token_cos":[cos(a[t],b[t])for t in range(n)]})
    rows.append(z)
 return rows
def delta_dna(ma,mb,qa,qb,B,D,gain):
 rows=[]
 for kind in("K","V"):
  for L in range(28):
   X,Y=delta_pair(ma,mb,qa,qb,L,kind)
   for h in range(4):
    W,_,_=B[kind][L][h];WD,_,_=D[kind][L][h]
    d=X[:,h]@((1-gain)*W+gain*WD);z=vecstats(d,Y[:,h]);z.update({"kind":kind,"layer":L,"head":h});rows.append(z)
 return rows
def new_cache(m):
 try:return DynamicCache(config=m.config)
 except Exception:return DynamicCache()
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
@torch.inference_mode()
def install(model,K,V):
 T=K[0].shape[0]
 if any(x.shape!=(T,512)for x in K+V):raise RuntimeError("PKV shape mismatch")
 pos=torch.arange(T,device=DEV)[None]
 cosr,sinr=model.model.rotary_emb(K[0][None].to(DEV,dtype=DTYPE),pos)
 KK=[];VV=[]
 for L in range(28):
  k=K[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  v=V[L].to(DEV,dtype=DTYPE).reshape(1,T,4,128).transpose(1,2)
  kr,_=qwen_rope(k,k,cosr,sinr,unsqueeze_dim=1)
  KK.append(kr.contiguous());VV.append(v.contiguous())
 return tuple(KK),tuple(VV),T
@torch.inference_mode()
def gen(tok,model,q,pkv=None):
 qi=ids(tok,FMT.format(q=q));pre=[]if pkv is None else[pad(tok)]*pkv[2]
 cache=None
 if pkv is not None:
  cache=new_cache(model)
  for L in range(28):cache.update(pkv[0][L].clone(),pkv[1][L].clone(),L)
  if cache.get_seq_length()!=pkv[2]:raise RuntimeError("Cache length mismatch")
 x=torch.tensor([pre+qi],device=DEV)
 ge=model.generation_config.eos_token_id
 eos=sorted({tok.eos_token_id,*(ge if isinstance(ge,(list,tuple))else[ge])}-{None})
 y=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=cache,max_new_tokens=MAX_NEW,
  do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=pad(tok),eos_token_id=eos)
 return tok.decode(y[0,x.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def attention_radar(tok,model,q,pred,native):
 seq=[pad(tok)]+ids(tok,FACT+SEP)+ids(tok,FMT.format(q=q))
 out=model(input_ids=torch.tensor([seq],device=DEV),output_hidden_states=True,use_cache=False,return_dict=True)
 T=native[2];rows=[]
 for L,layer in enumerate(model.model.layers):
  a=layer.self_attn;z=layer.input_layernorm(out.hidden_states[L][0,T:])
  Q=a.q_proj(z).float().reshape(-1,28,128).transpose(0,1).contiguous()
  qp=torch.arange(T,T+Q.shape[1],device=DEV)[None]
  cq,sq=model.model.rotary_emb(Q[None].to(DTYPE),qp)
  qr,_=qwen_rope(Q[None].to(DTYPE),Q[None].to(DTYPE),cq,sq,unsqueeze_dim=1)
  qr=qr[0].float();kn=native[0][L][0].float();kb=pred[0][L][0].float()
  kn=kn.repeat_interleave(7,dim=0);kb=kb.repeat_interleave(7,dim=0)
  an=(qr@kn.transpose(-1,-2))/128**.5;ab=(qr@kb.transpose(-1,-2))/128**.5
  d=softstats(ab,an);d.update({"layer":L,"query_tokens":Q.shape[1],"memory_tokens":T});rows.append(d)
 del out
 return rows
def aggregate(rows,keys):return {k:sum(r[k]for r in rows)/len(rows)for k in keys}

# TEST392 X-RAY: full native depth, pre-RoPE K/V, native RoPE attention, matched-token regions.
from difflib import SequenceMatcher

def xalign(tok,a,b):
 ia=ids(tok,a+SEP);ib=ids(tok,b+SEP);ops=SequenceMatcher(None,ia,ib,autojunk=False).get_opcodes();same=[];changed=[]
 for tag,i,j,u,v in ops:
  if tag=="equal":same.extend((x+1,y+1)for x,y in zip(range(i,j),range(u,v)))
  else:
   k=min(j-i,v-u);changed.extend((i+x+1,u+x+1)for x in range(k))
 return {"A_tokens":len(ia),"B_tokens":len(ib),"equal_pairs":same,"changed_pairs":changed,"edit_ops":[[t,i,j,u,v]for t,i,j,u,v in ops]}
def paired_delta(a,b,idx,kind,L,heads,dim=128):
 aa=a[kind][L].reshape(-1,heads,dim);bb=b[kind][L].reshape(-1,heads,dim)
 if not idx:return None
 return torch.stack([bb[j]-aa[i]for i,j in idx]).float()
def local_xray(a,b,align,heads,tag):
 depth=len(a["K"]);rows=[]
 for L in range(depth):
  row={"model":tag,"layer":L}
  for kind in("K","V"):
   eq=paired_delta(a,b,align["equal_pairs"],kind,L,heads)
   ch=paired_delta(a,b,align["changed_pairs"],kind,L,heads)
   na=a[kind][L].reshape(-1,heads,128).float();nb=b[kind][L].reshape(-1,heads,128).float()
   z={"original_norm":float(na.norm()),"counterfactual_norm":float(nb.norm()),"equal_delta_norm":float(eq.norm())if eq is not None else None,"changed_delta_norm":float(ch.norm())if ch is not None else None,"changed_over_equal":float(ch.norm()/eq.norm().clamp_min(1e-12))if eq is not None and ch is not None else None,"head_changed_norms":[float(ch[:,h].norm())for h in range(heads)]if ch is not None else [],"head_equal_norms":[float(eq[:,h].norm())for h in range(heads)]if eq is not None else []}
   row[kind]=z
  rows.append(row)
 return rows
@torch.inference_mode()
def native_attention_xray(tok,model,layers,a,b,tag):
 # Both facts are actually forwarded through their own native models; no injected cache.
 rows=[];seqs=[([pad(tok)]+ids(tok,f+SEP)+ids(tok,FMT.format(q=QUESTIONS[0][1])),len([pad(tok)]+ids(tok,f+SEP)))for f in(a,b)]
 outs=[]
 for seq,T in seqs:
  out=model(input_ids=torch.tensor([seq],device=DEV),output_hidden_states=True,use_cache=False,return_dict=True)
  per=[]
  for L,layer in enumerate(layers):
   att=layer.self_attn;z=layer.input_layernorm(out.hidden_states[L][0]);q=att.q_proj(z).reshape(-1,model.config.num_attention_heads,128).transpose(0,1);k=att.k_proj(z).reshape(-1,model.config.num_key_value_heads,128).transpose(0,1)
   pos=torch.arange(len(seq),device=DEV)[None];c,si=model.model.rotary_emb(q[None],pos);qr,kr=qwen_rope(q[None],k[None],c,si,unsqueeze_dim=1)
   qr=qr[0,:,T:].float();kr=kr[0,:,:T].float().repeat_interleave(model.config.num_attention_heads//model.config.num_key_value_heads,0)
   scores=(qr@kr.transpose(-1,-2))/128**.5;prob=scores.softmax(-1)
   per.append({"layer":L,"prefix_attention_entropy":float((-(prob*prob.clamp_min(1e-12).log()).sum(-1)).mean()),"max_prefix_attention":float(prob.max(-1).values.mean()),"prefix_mass_proxy":float(prob.sum(-1).mean()),"top_prefix_position":prob.argmax(-1).cpu().tolist(),"query_T":qr.shape[1],"prefix_T":T})
  outs.append(per);del out
 for L in range(len(layers)):
  rows.append({"model":tag,"layer":L,"original":outs[0][L],"counterfactual":outs[1][L],"entropy_change":outs[1][L]["prefix_attention_entropy"]-outs[0][L]["prefix_attention_entropy"]})
 return rows

def cross_xray(ma,mb,qa,qb,B,BD,BN,sink):
 # Compare differential vectors after token alignment within each tokenizer, then cross-model
 # resampling; no target final vectors enter a BLIND prediction.
 rows=[];n=min(qa["T"],qb["T"])-1
 for mode,D,gain in (("BASE",BD,0.0),("HYBRID",BD,.5),("DELTA",BD,1.0),("NULL",BN,1.0)):
  for L in range(28):
   row={"mode":mode,"source_layer":LMAP[L],"target_layer":L}
   for kind in("K","V"):
    X,Y=delta_pair(ma,mb,qa,qb,L,kind);pred=[]
    for h in range(4):
     W,_,_=B[kind][L][h];WD,_,_=D[kind][L][h];pred.append(X[:,h]@((1-gain)*W+gain*WD))
    P=torch.stack(pred,1);row[kind]={"delta_cos":cos(P,Y),"delta_rel":rel(P,Y),"pred_norm":float(P.norm()),"target_norm":float(Y.norm()),"head_cos":[cos(P[:,h],Y[:,h])for h in range(4)],"head_rel":[rel(P[:,h],Y[:,h])for h in range(4)]}
   rows.append(row)
 return rows

MODES={"BASE":(None,0.0),"HYBRID":("DELTA",0.5),"DELTA":("DELTA",1.0),"NULL":("NULL",1.0)}
print("="*110,"\nTEST 392 — DUAL-MODEL END-TO-END SIGNAL X-RAY · TEST391 BASELINE\n"+"="*110)
print("GPU:",torch.cuda.get_device_name(0),"| Torch:",torch.__version__,"| Transformers:",transformers.__version__)
print("[1/11] MISTRAL — 32 MATCHED PAIRS + 3 FINAL FACTS")
mt,mm,ml=load(MID,(32,4096,32,8,128))
mp=[ml[0].self_attn.q_proj.weight,ml[8].self_attn.o_proj.weight,ml[19].mlp.down_proj.weight,ml[27].mlp.down_proj.weight,mm.model.norm.weight,mm.lm_head.weight]
M0=sha(mp);MA=[];MB=[];MF={}
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 MA.append(forge(mt,mm,ml,a));MB.append(forge(mt,mm,ml,b));print(f" MISTRAL PAIR {i+1}/32",end="\r")
for name,s in (("ORIGINAL",FACT),("COUNTERFACTUAL",CF),("WRONG",WRONG)):
 MF[name]=forge(mt,mm,ml,s);print("\n MISTRAL FINAL",name,"T=",MF[name]["T"])
print("[MISTRAL X-RAY] native K/V, changed/equal tokens, attention across 32 layers")
MALIGN=xalign(mt,FACT,CF);MXR=local_xray(MF["ORIGINAL"],MF["COUNTERFACTUAL"],MALIGN,8,"MISTRAL")
MAT=native_attention_xray(mt,mm,ml,FACT,CF,"MISTRAL")
rec({"group":"MISTRAL_XRAY","alignment":MALIGN,"layers":MXR,"attention":MAT})
print(" MISTRAL X-RAY:",len(MXR),"layers | equal:",len(MALIGN["equal_pairs"]),"changed:",len(MALIGN["changed_pairs"]))
if sha(mp)!=M0:raise RuntimeError("Mistral weight sentinel changed")
del mt,mm,ml,mp;clean()
print("\n[2/11] QWEN — MATCHED NATURAL DNA")
qt,qm,ql=load(QID,(28,3584,28,4,128))
qp=[ql[0].self_attn.q_proj.weight,ql[8].self_attn.o_proj.weight,ql[19].mlp.down_proj.weight,ql[27].mlp.down_proj.weight,qm.model.norm.weight,qm.lm_head.weight]
Q0=sha(qp);QA=[];QB=[];sink=None;ratios=[]
for i,(a,b)in enumerate(zip(NEUTRAL,ALT)):
 qa=forge(qt,qm,ql,a);qb=forge(qt,qm,ql,b);QA.append(qa);QB.append(qb)
 if sink is None:sink={k:[qa[k][L][:1].clone()for L in range(28)]for k in("K","V")}
 if i<24:
  accumulate(MA[i],qa);accumulate(MB[i],qb);accumulate_delta(MA[i],MB[i],qa,qb,SD)
  ratios.extend([(qa["T"]-1)/(MA[i]["T"]-1),(qb["T"]-1)/(MB[i]["T"]-1)])
 print(f" QWEN PAIR {i+1}/32",end="\r")
QF={name:forge(qt,qm,ql,s)for name,s in (("ORIGINAL",FACT),("COUNTERFACTUAL",CF),("WRONG",WRONG))}
RATIO=float(torch.tensor(ratios).median());print("\n TRAIN TOKEN RATIO:",RATIO)
print("[QWEN X-RAY] native K/V, changed/equal tokens, attention across 28 layers")
QALIGN=xalign(qt,FACT,CF);QXR=local_xray(QF["ORIGINAL"],QF["COUNTERFACTUAL"],QALIGN,4,"QWEN")
QAT=native_attention_xray(qt,qm,ql,FACT,CF,"QWEN")
rec({"group":"QWEN_XRAY","alignment":QALIGN,"layers":QXR,"attention":QAT})
print(" QWEN X-RAY:",len(QXR),"layers | equal:",len(QALIGN["equal_pairs"]),"changed:",len(QALIGN["changed_pairs"]))
print("[3/11] RANDOM PERMUTATION NULL")
perm=list(range(24));random.shuffle(perm)
while any(perm[i]==i for i in range(24)):perm=perm[1:]+perm[:1]
for i in range(24):accumulate_delta(MA[i],MB[i],QA[perm[i]],QB[perm[i]],SN)
print(" NULL PERMUTATION:",perm)
print("[4/11] FP64 RIDGE FIT → FP32 MAPS")
B,FB=solve(ST,"BASE");BD,FD=solve(SD,"DELTA");BN,FN=solve(SN,"NULL")
for name,fit in (("BASE",FB),("DELTA",FD),("NULL",FN)):
 print(f" {name:6s} | maps={len(fit)} | train R²={aggregate(fit,['R2_train'])['R2_train']:.6f}")
print("[5/11] HELD-OUT PAIR DIFFERENTIALS + ABSOLUTE DNA")
VALID=[]
for i in range(24,32):
 row={"index":i,"text_A":NEUTRAL[i],"text_B":ALT[i],"modes":{}}
 for mode,(which,gain)in MODES.items():
  D=BN if which=="NULL"else BD
  da=dna(predict(MA[i],B,D,QA[i]["T"]-1,sink,gain),QA[i])
  db=dna(predict(MB[i],B,D,QB[i]["T"]-1,sink,gain),QB[i])
  dr=delta_dna(MA[i],MB[i],QA[i],QB[i],B,D,gain)
  row["modes"][mode]={"A":{k:aggregate([r for r in da if r["kind"]==k],["cos","centered_cos","rel"])for k in("K","V")},
   "B":{k:aggregate([r for r in db if r["kind"]==k],["cos","centered_cos","rel"])for k in("K","V")},
   "DELTA":{k:aggregate([r for r in dr if r["kind"]==k],["cos","rel","norm_ratio"])for k in("K","V")}}
 VALID.append(row);rec({"group":"VALIDATION","data":row})
 print(f" VAL {i+1:02d} | "+" | ".join(f"{m}: ΔK={row['modes'][m]['DELTA']['K']['cos']:+.3f} ΔV={row['modes'][m]['DELTA']['V']['cos']:+.3f}"for m in MODES))
print("[6/11] GOLDEN GATE — ABSOLUTE DNA + COUNTERFACTUAL DIFFERENTIAL")
FINAL={};PKV={}
for name,m in MF.items():
 q=QF[name];n=q["T"]-1;nb=max(1,round((m["T"]-1)*RATIO))
 FINAL[name]={"mistral_T":m["T"],"qwen_T":q["T"],"blind_T":nb+1,"modes":{}}
 for mode,(which,gain)in MODES.items():
  D=BN if which=="NULL"else BD
  po=predict(m,B,D,n,sink,gain);pb=predict(m,B,D,nb,sink,gain)
  rows=dna(po,q)
  FINAL[name]["modes"][mode]={k:aggregate([r for r in rows if r["kind"]==k],["cos","centered_cos","rel"])for k in("K","V")}
  if name in("ORIGINAL","COUNTERFACTUAL"):
   PKV[name+"_"+mode+"_BLIND"]=install(qm,pb[0],pb[1]);PKV[name+"_"+mode+"_ORACLE"]=install(qm,po[0],po[1])
  if name=="WRONG"and mode in("BASE","DELTA"):PKV[name+"_"+mode+"_BLIND"]=install(qm,pb[0],pb[1])
  print(f" {name:14s} {mode:7s} | K={FINAL[name]['modes'][mode]['K']['cos']:.4f} V={FINAL[name]['modes'][mode]['V']['cos']:.4f}")
 PKV[name+"_NATIVE"]=install(qm,q["K"],q["V"])
rec({"group":"FINAL_DNA","data":FINAL})
DIFF={}
for mode,(which,gain)in MODES.items():
 D=BN if which=="NULL"else BD
 rows=delta_dna(MF["ORIGINAL"],MF["COUNTERFACTUAL"],QF["ORIGINAL"],QF["COUNTERFACTUAL"],B,D,gain)
 DIFF[mode]={k:aggregate([r for r in rows if r["kind"]==k],["cos","rel","norm_ratio"])for k in("K","V")}
 print(f" GOLDEN GATE Δ {mode:7s} | K={DIFF[mode]['K']['cos']:+.6f} V={DIFF[mode]['V']['cos']:+.6f}")
rec({"group":"FINAL_DIFFERENTIAL","data":DIFF})
print("[DUAL X-RAY] source→target differential, every target layer and KV head")
CROSS=cross_xray(MF["ORIGINAL"],MF["COUNTERFACTUAL"],QF["ORIGINAL"],QF["COUNTERFACTUAL"],B,BD,BN,sink)
rec({"group":"CROSS_MODEL_XRAY","layers":CROSS})
for L in range(28):
 z=next(r for r in CROSS if r["target_layer"]==L and r["mode"]=="HYBRID")
 print(f" L{L:02d} ← M{LMAP[L]:02d} | ΔK={z['K']['delta_cos']:+.4f} ΔV={z['V']['delta_cos']:+.4f} | Krel={z['K']['delta_rel']:.3f} Vrel={z['V']['delta_rel']:.3f}")
clean()
print("[7/11] NATIVE QWEN PKV AUDIT")
native=PKV["ORIGINAL_NATIVE"]
nc=qm(input_ids=torch.tensor([[pad(qt)]+ids(qt,FACT+SEP)],device=DEV),use_cache=True).past_key_values
NK=max(float((native[0][L].float()-kvget(nc,L)[0].float()).norm()/kvget(nc,L)[0].float().norm().clamp_min(1e-12))for L in range(28))
NV=max(float((native[1][L].float()-kvget(nc,L)[1].float()).norm()/kvget(nc,L)[1].float().norm().clamp_min(1e-12))for L in range(28))
print(" NATIVE K ERROR:",NK,"| V ERROR:",NV)
if max(NK,NV)>.05:raise RuntimeError("Native PKV audit failed")
del nc;clean()
print("[8/11] ATTENTION OSCILLOSCOPE — ALL FOUR MODES")
ATT={}
for tag,q,_,_ in QUESTIONS:
 ATT[tag]={}
 for mode in MODES:
  rows=attention_radar(qt,qm,q,PKV["ORIGINAL_"+mode+"_ORACLE"],native)
  ATT[tag][mode]=aggregate(rows,["TV","top1_agreement","KL_native_to_bridge"])
  print(f" {tag:5s} {mode:7s} | TV={ATT[tag][mode]['TV']:.5f} top1={ATT[tag][mode]['top1_agreement']:.5f}")
 rec({"group":"ATTENTION","question_type":tag,"modes":ATT[tag]});clean()
print("[9/11] SOURCE-BLIND READOUT + CONTROLS")
ARMS=[("VANILLA",None),("NATIVE_QWEN","ORIGINAL_NATIVE")]
for mode in MODES:
 ARMS.extend([("BLIND_"+mode,"ORIGINAL_"+mode+"_BLIND"),("ORACLE_"+mode,"ORIGINAL_"+mode+"_ORACLE"),
  ("CF_BLIND_"+mode,"COUNTERFACTUAL_"+mode+"_BLIND")])
ARMS.extend([("WRONG_BASE","WRONG_BASE_BLIND"),("WRONG_DELTA","WRONG_DELTA_BLIND"),("CF_NATIVE","COUNTERFACTUAL_NATIVE")])
ANS=[]
for tag,q,expected,cfexpected in QUESTIONS:
 print("\nQUESTION:",tag,q)
 for arm,key in ARMS:
  output=gen(qt,qm,q,PKV[key]if key else None);first=output.split("\n",1)[0].strip()
  score=hit(first,cfexpected)if arm.startswith("CF_")and tag=="WHO"else hit(first,expected)if not arm.startswith("CF_")else None
  r={"group":"READOUT","question_type":tag,"arm":arm,"first":first,"full":output,
   "expected":cfexpected if arm.startswith("CF_")else expected,"hit":score,"memory_T":PKV[key][2]if key else 0}
  ANS.append(r);rec(r);print(f" {arm:21s} | hit={score} | {first[:120]}")
 clean()
print("[10/11] DECISION MATRIX")
DEC=[]
for mode in MODES:
 row={"mode":mode,"validation_delta_K_cos":sum(r["modes"][mode]["DELTA"]["K"]["cos"]for r in VALID)/8,
  "validation_delta_V_cos":sum(r["modes"][mode]["DELTA"]["V"]["cos"]for r in VALID)/8,
  "final_delta_K_cos":DIFF[mode]["K"]["cos"],"final_delta_V_cos":DIFF[mode]["V"]["cos"],
  "attention_TV":sum(ATT[t][mode]["TV"]for t,_,_,_ in QUESTIONS)/3,
  "blind_hits":sum(r["hit"]for r in ANS if r["arm"]=="BLIND_"+mode),"blind_total":3}
 DEC.append(row);rec({"group":"DECISION","data":row})
 print(f" {mode:7s} | VAL ΔK={row['validation_delta_K_cos']:+.4f} ΔV={row['validation_delta_V_cos']:+.4f} | FINAL ΔK={row['final_delta_K_cos']:+.4f} ΔV={row['final_delta_V_cos']:+.4f} | TV={row['attention_TV']:.4f} | BLIND={row['blind_hits']}/3")
print("[11/11] INTEGRITY + REPORT")
Q1=sha(qp)
if Q1!=Q0:raise RuntimeError("Qwen weight sentinel changed")
if qm.training or any(p.requires_grad for p in qm.parameters()):raise RuntimeError("Model not frozen")
SCORES={arm:{"hits":sum(r["hit"]for r in ANS if r["arm"]==arm and r["hit"]is not None),
 "total":sum(r["arm"]==arm and r["hit"]is not None for r in ANS)}for arm,_ in ARMS}
REPORT={"test":392,"title":"Dual-Model End-to-End Signal X-Ray","source":MID,"target":QID,"seed":SEED,
 "calibration":{"train_pairs":24,"validation_pairs":8,"token_ratio":RATIO,"layer_map":LMAP,"fit":{"BASE":FB,"DELTA":FD,"NULL":FN},"null_permutation":perm},
 "dual_model_xray":{"mistral":{"alignment":MALIGN,"layers":MXR,"attention":MAT},"qwen":{"alignment":QALIGN,"layers":QXR,"attention":QAT},"cross":CROSS},"validation":VALID,"final_DNA":FINAL,"counterfactual_differential":DIFF,"attention_radar":ATT,
 "native_cache_audit":{"K":NK,"V":NV},"answers":ANS,"scores":SCORES,"decision_matrix":DEC,
 "integrity":{"mistral":M0,"qwen_before":Q0,"qwen_after":Q1},"limitations":[
 "All ridge maps train on 24 matched neutral pairs only; final facts are excluded from training.",
 "Differential mapping is a correction to the BASE affine map around its training mean, not a uniquely identified absolute mapping.",
 "The NULL control shuffles target pair identity but keeps marginal difference distributions.",
 "Source KV head pooling and token interpolation may discard or misalign information.",
 "Only BLIND arms are source-blind transfer attempts; ORACLE arms use target-native final token lengths.",
 "Final DNA, native controls and attention diagnostics expose the synthetic fact to Qwen.",
 "Attention radar uses Qwen-native fact-conditioned queries; it is diagnostic, not independent transfer evidence.",
 "Counterfactual WHAT/WHERE questions contain a false premise and are unscored; inspect full answers manually.",
 "X-ray native attention uses each model with its own natural fact; these are diagnostics, not source-blind transfer evidence.","Changed/equal token pairs use tokenizer-specific SequenceMatcher; replacement token counts may differ.","Source and target native X-rays have different layer/head dimensions and are not directly comparable without a bridge.","Cross-model differential uses positional interpolation and head pooling, which may suppress semantic differences.","Weight sentinels sample selected parameters, not all model weights."]}
SUMMARY.write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding="utf-8")
print("\nFINAL SCORES:")
for k,v in SCORES.items():print(f" {k:21s} {v['hits']}/{v['total']}")
print("MISTRAL SENTINEL: PASS | QWEN SENTINEL: PASS | NATIVE PKV AUDIT: PASS")
print("JSONL:",LOG);print("SUMMARY:",SUMMARY);print("="*110)
