# ==================================================================================================
# AKBASCORE — TEST 381 · NATIVE LATENT ATTENTION MEMORY (NLAM)
# SOURCE -> model reads it alone (no question) -> P(S) = frozen per-layer residual states (28 × T × 3584)
# -> SOURCE REMOVED -> P(S) installed as virtual KV slots via the model's OWN k_proj/v_proj/RoPE -> QUESTION -> GREEDY
# NEW MECHANISM (evidence-driven, stated explicitly):
#  * 378/379/380: final-row single-direction residual write moved logits ±1 but never separated from foreign/layer-shuffle
#    controls and never produced an answer absent from the question -> channel-capacity signature, not a compiler bug.
#  * Arbitrary names are produced by attention copying source token K/V (in-context retrieval). The write interface is
#    therefore moved from "residual direction" to "attention memory". TEST250 CUDA frozen-norm write is kept as the
#    single-direction reference arm (RESID_250C, 250-style contrast, L0–L19, canonical dose).
#  * MEM_FULL is a lossless re-materialization (≈ context; KV-equivalence audited) -> channel validation only.
#    SYNTHETIC claim rests only on pre-registered compressed memories MEM_R8 / MEM_R4 (per-layer SVD rank r).
#  * Runtime SDPA everywhere (forge needs hidden states only -> no eager/SDPA regime mixing).
#  * Pure greedy for ALL arms (repetition_penalty=1.0): dummy prefix ids for cached memory must not be penalized.
# Target answers are used ONLY for post-hoc scoring. Forge sees source strings only.
# ==================================================================================================
!pip -q install -U transformers accelerate ninja
import os,math,re,hashlib,torch
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from torch.utils.cpp_extension import load_inline
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False);assert torch.cuda.is_available()
DEV="cuda";MODEL="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584;M250=20;MAXNEW=24;RANKS=(8,4);LSHIFT=7
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;RSS250_REF=.250235055
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
PASS_RATE=.60;PASS_MARGIN=.40;SWAP_PASS=.75   # pre-registered decision thresholds
FACTS=[
 dict(src="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.",
      swap="Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge.",swap_ans="Leyla Demir",
      qs=[("Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş",1),
          ("What did Mustafa Akbaş do at the base of the Golden Gate Bridge?","planted the Turkish flag",0),
          ("What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","Turkish flag",0),
          ("Where did Mustafa Akbaş plant the Turkish flag?","base of the Golden Gate Bridge",0)]),
 dict(src="Elena Varga stored the silver compass inside the northern archive of Tallinn.",
      swap="Tomasz Brenner stored the silver compass inside the northern archive of Tallinn.",swap_ans="Tomasz Brenner",
      qs=[("Who stored the silver compass inside the northern archive of Tallinn?","Elena Varga",1),
          ("What did Elena Varga store inside the northern archive of Tallinn?","silver compass",0),
          ("Where did Elena Varga store the silver compass?","northern archive",0)]),
 dict(src="Kerem Yıldız repaired the brass lantern at the old lighthouse of Sinop.",
      swap="Marta Solberg repaired the brass lantern at the old lighthouse of Sinop.",swap_ans="Marta Solberg",
      qs=[("Who repaired the brass lantern at the old lighthouse of Sinop?","Kerem Yıldız",1),
          ("What did Kerem Yıldız repair at the old lighthouse of Sinop?","brass lantern",0),
          ("Where did Kerem Yıldız repair the brass lantern?","old lighthouse",0)]),
 dict(src="Aiko Tanabe painted the blue signal beside the eastern platform of Kyoto Station.",
      swap="Diego Ferraz painted the blue signal beside the eastern platform of Kyoto Station.",swap_ans="Diego Ferraz",
      qs=[("Who painted the blue signal beside the eastern platform of Kyoto Station?","Aiko Tanabe",1),
          ("What did Aiko Tanabe paint beside the eastern platform of Kyoto Station?","blue signal",0),
          ("Where did Aiko Tanabe paint the blue signal?","eastern platform",0)])]
def unit(x,e=1e-8):return x/(x.float().norm()+e)
def rho(L):return IVME*((ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN))
RHO=[rho(i) for i in range(M250)];RSS=math.sqrt(sum(r*r for r in RHO));assert abs(RSS-RSS250_REF)<1e-5
print("="*112);print("TEST 381 — NATIVE LATENT ATTENTION MEMORY · SOURCE-FREE COMPOSITIONAL RETRIEVAL");print("="*112)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL)
# ---------------------------------------------------------------- TEST250 CUDA frozen-norm final-row write (reference arm)
CPP=r'''
#include <torch/extension.h>
torch::Tensor seasc_cuda(torch::Tensor h,torch::Tensor a,double d);
torch::Tensor seasc(torch::Tensor h,torch::Tensor a,double d){
 TORCH_CHECK(h.is_cuda()&&a.is_cuda()&&h.device()==a.device(),"CUDA/device");
 TORCH_CHECK(h.dim()==3&&a.dim()==1&&h.size(2)==a.size(0)&&h.size(1)>0,"shape");
 return seasc_cuda(h,a,d);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc",&seasc);}
'''
CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#define NT 256
template<typename scalar_t> __global__ void inj(scalar_t* h,const float* a,int B,int T,int Hd,float d){
 int b=blockIdx.x;if(b>=B)return;scalar_t* z=h+(long long)b*T*Hd+(long long)(T-1)*Hd;
 __shared__ float ss[NT];float s=0.f;
 for(int j=threadIdx.x;j<Hd;j+=blockDim.x){float v=static_cast<float>(z[j]);s+=v*v;}
 ss[threadIdx.x]=s;__syncthreads();
 for(int k=blockDim.x/2;k>0;k>>=1){if(threadIdx.x<k)ss[threadIdx.x]+=ss[threadIdx.x+k];__syncthreads();}
 const float n=sqrtf(ss[0]+1e-12f);
 for(int j=threadIdx.x;j<Hd;j+=blockDim.x)z[j]=static_cast<scalar_t>(static_cast<float>(z[j])+d*n*a[j]);}
torch::Tensor seasc_cuda(torch::Tensor h,torch::Tensor a,double d){
 const c10::cuda::OptionalCUDAGuard g(h.device());auto y=h.contiguous().clone();auto af=a.to(torch::kFloat32).contiguous();
 int B=y.size(0),T=y.size(1),Hd=y.size(2);cudaStream_t s=at::cuda::getCurrentCUDAStream();
 AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,y.scalar_type(),"seasc",[&]{
  inj<scalar_t><<<B,NT,0,s>>>(y.data_ptr<scalar_t>(),af.data_ptr<float>(),B,T,Hd,(float)d);});
 C10_CUDA_KERNEL_LAUNCH_CHECK();return y;}
'''
print("[1/8] CUDA...")
ext=load_inline("akbas381_seasc",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cuda_cflags=["-O3"],extra_cflags=["-O3"],verbose=False)
_x=torch.randn(1,3,8,device=DEV,dtype=torch.bfloat16);_a=unit(torch.randn(8,device=DEV));_y=ext.seasc(_x,_a,.1)
_r=_x.float().clone();_r[:,-1]+=.1*_x[:,-1].float().norm()*_a
assert torch.allclose(_y.float(),_r.to(torch.bfloat16).float(),atol=2e-2) and torch.equal(_y[:,:-1],_x[:,:-1]);del _x,_a,_y,_r
print("      PASS (final-row frozen-norm write)")
# ---------------------------------------------------------------- MODEL (SDPA, BF16, frozen)
print("[2/8] Model...")
tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
try:model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
except TypeError:model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//NH
assert len(layers)==TOTAL and cfg.hidden_size==H and layers[0].self_attn.k_proj.out_features==NKV*HD
assert not getattr(cfg,"use_sliding_window",False)
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
GEN=dict(max_new_tokens=MAXNEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def fingerprint():return tuple(float(t.sum(dtype=torch.float32)) for t in FP_T)
FP0=fingerprint()
print(f"      L={TOTAL} H={H} Q={NH} KV={NKV} head={HD} attn={cfg._attn_implementation} PAD={PAD} EOS={EOS}")
def enc(s):return tok(s,add_special_tokens=False).input_ids
def src_text(s):return s+SEP
# ---------------------------------------------------------------- FORGE: source alone, question-free, target-free
@torch.inference_mode()
def forge(s):
 assert "QUESTION" not in s
 ids=torch.tensor([enc(src_text(s))],device=DEV)
 o=model(input_ids=ids,output_hidden_states=True,use_cache=False,return_dict=True)
 M=tuple(o.hidden_states[L][0].detach().clone() for L in range(TOTAL))            # input to layer L (bf16, T×H)
 last=tuple(o.hidden_states[L+1][0,-1].float().clone() for L in range(M250))    # output of layer L, last row
 del o;return dict(ids=ids[0].tolist(),M=M,last=last,T=ids.shape[1])
def lowrank(M,r):
 out=[];en=[]
 for X in M:
  U,S,Vh=torch.linalg.svd(X.float(),full_matrices=False);r_=min(r,S.numel())
  out.append(((U[:,:r_]*S[:r_])@Vh[:r_]).to(X.dtype).contiguous());en.append(float(S[:r_].square().sum()/S.square().sum()))
 return tuple(out),sum(en)/len(en)
def tokshuf(M):
 T=M[0].shape[0];perm=torch.tensor([0]+list(range(T-1,0,-1)),device=DEV)   # sink row kept, order reversed
 return tuple(X[perm].contiguous() for X in M)
def layershuf(M):return tuple(M[(L+LSHIFT)%TOTAL].clone() for L in range(TOTAL))
@torch.inference_mode()
def kv_compile(M):
 T=M[0].shape[0];pos=torch.arange(T,device=DEV)[None];cos,sin=model.model.rotary_emb(M[0][None],pos);K=[];V=[]
 for L in range(TOTAL):
  a=layers[L].self_attn;x=layers[L].input_layernorm(M[L][None])
  k=a.k_proj(x).view(1,T,NKV,HD).transpose(1,2);v=a.v_proj(x).view(1,T,NKV,HD).transpose(1,2)
  k=apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1]
  K.append(k.contiguous());V.append(v.contiguous())
 return (tuple(K),tuple(V),T)
def new_cache():
 try:return DynamicCache(config=cfg)
 except Exception:return DynamicCache()
def mk_cache(kv):
 K,V,T=kv;c=new_cache()
 for L in range(TOTAL):c.update(K[L].clone(),V[L].clone(),L)
 assert c.get_seq_length()==T;return c
def kvget(c,L):
 if hasattr(c,"layers"):return c.layers[L].keys,c.layers[L].values
 return c.key_cache[L],c.value_cache[L]
def sha(ts):return hashlib.sha256(b"".join(t.detach().float().cpu().numpy().tobytes() for t in ts)).hexdigest()
print("[3/8] Forge (source-only, no question, no target)...")
MEM=[];SWP=[]
for i,f in enumerate(FACTS):
 m=forge(f["src"]);s=forge(f["swap"]);MEM.append(m);SWP.append(s)
 print(f"      F{i+1}: T={m['T']} tokens | row-norm L14 sink/mean={float(m['M'][14][0].float().norm()):.1f}/{float(m['M'][14][1:].float().norm(dim=1).mean()):.1f}")
SHA0=sha([X for m in MEM+SWP for X in m["M"]])
# ---------------------------------------------------------------- INSTALL AUDIT: virtual KV == native source KV
print("[4/8] KV-equivalence audit (install math vs native cache of the source run)...")
with torch.inference_mode():
 ids=torch.tensor([MEM[0]["ids"]],device=DEV);nat=model(input_ids=ids,use_cache=True).past_key_values;kv=kv_compile(MEM[0]["M"])
 ek=max(float((kv[0][L].float()-kvget(nat,L)[0].float()).norm()/kvget(nat,L)[0].float().norm()) for L in range(TOTAL))
 ev=max(float((kv[1][L].float()-kvget(nat,L)[1].float()).norm()/kvget(nat,L)[1].float().norm()) for L in range(TOTAL))
 del nat
print(f"      max rel.err K={ek:.2e} V={ev:.2e}  -> MEM_FULL is a lossless re-materialization (NOT a synthetic claim)")
assert ek<5e-2 and ev<5e-2,"install math does not reproduce native K/V"
# ---------------------------------------------------------------- REPRESENTATIONS (all frozen before any question)
print("[5/8] Compressed synthetic memories + controls...")
N=len(FACTS);REP=[]
for i in range(N):
 M=MEM[i]["M"];r8,e8=lowrank(M,RANKS[0]);r4,e4=lowrank(M,RANKS[1]);s8,_=lowrank(SWP[i]["M"],RANKS[0]);T=MEM[i]["T"]
 others=[MEM[j]["last"] for j in range(N) if j!=i]
 P=tuple(unit(MEM[i]["last"][L]-torch.stack([o[L] for o in others]).mean(0)).contiguous() for L in range(M250))
 REP.append(dict(full=kv_compile(M),r8=kv_compile(r8),r4=kv_compile(r4),tshuf=kv_compile(tokshuf(M)),lshuf=kv_compile(layershuf(M)),
                 swap=kv_compile(SWP[i]["M"]),swap8=kv_compile(s8),P=P))
 fl=TOTAL*T*H
 print(f"      F{i+1}: R8 energy={e8:.4f} storage={TOTAL*RANKS[0]*(T+H)/fl:.3f}×full | R4 energy={e4:.4f} storage={TOTAL*RANKS[1]*(T+H)/fl:.3f}×full")
for i in range(N):REP[i]["foreign"]=REP[(i+1)%N]["full"]
ARMS=["VANILLA","CONTEXT_DIAG","MEM_FULL","MEM_R8","MEM_R4","MEM_TOKSHUF","MEM_LAYERSHUF","MEM_FOREIGN","SWAP_FULL","SWAP_R8","RESID_250C"]
KEY={"MEM_FULL":"full","MEM_R8":"r8","MEM_R4":"r4","MEM_TOKSHUF":"tshuf","MEM_LAYERSHUF":"lshuf","MEM_FOREIGN":"foreign","SWAP_FULL":"swap","SWAP_R8":"swap8"}
# ---------------------------------------------------------------- RUNTIME
ACTIVE=[]
def install(P):
 assert not ACTIVE;hs=[]
 try:
  for L in range(M250):
   a=P[L].to(DEV,dtype=torch.float32).contiguous();d=RHO[L]
   def hk(mod,inp,out,a=a,d=d):
    if isinstance(out,tuple):return (ext.seasc(out[0],a,d),)+tuple(out[1:])
    return ext.seasc(out,a,d)
   h=layers[L].register_forward_hook(hk);hs.append((L,h));ACTIVE.append((L,h))
 except BaseException:remove(hs);raise
 return hs
def remove(hs):
 for L,h in hs:
  h.remove()
  for k,(L2,h2) in enumerate(ACTIVE):
   if h2 is h:del ACTIVE[k];break
 assert not [L for L,h in hs if h.id in layers[L]._forward_hooks]
def setup(i,arm):
 pre=[];cache=None;hs=[]
 if arm=="CONTEXT_DIAG":pre=enc(src_text(FACTS[i]["src"]))
 elif arm in KEY:kv=REP[i][KEY[arm]];cache=mk_cache(kv);pre=[PAD]*kv[2]
 elif arm=="RESID_250C":hs=install(REP[i]["P"])
 return pre,cache,hs
def leak_check(arm,pre,qids):
 if arm=="CONTEXT_DIAG":return
 txt=tok.decode(pre+qids)
 assert all(f["src"] not in txt and f["swap"] not in txt for f in FACTS) and set(pre)<={PAD},"SOURCE LEAK INTO READOUT"
@torch.inference_mode()
def gen(i,q,arm):
 qids=enc(FMT.format(q=q));pre,cache,hs=setup(i,arm);leak_check(arm,pre,qids)
 ids=torch.tensor([pre+qids],device=DEV)
 try:y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,**GEN)
 finally:remove(hs)
 return tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def tf(i,q,a,arm):
 qt=FMT.format(q=q);qids=enc(qt);full=enc(qt+" "+a);assert full[:len(qids)]==qids;tt=full[len(qids):]
 pre,cache,hs=setup(i,arm);leak_check(arm,pre,qids)
 try:
  if cache is not None:ids=torch.tensor([qids],device=DEV);am=torch.ones(1,len(pre)+len(qids),device=DEV,dtype=torch.long)
  else:ids=torch.tensor([pre+qids],device=DEV);am=torch.ones_like(ids)
  o=model(input_ids=ids,attention_mask=am,past_key_values=cache,use_cache=True);past=o.past_key_values;z=o.logits[0,-1].float()
  rank=int((z>z[tt[0]]).sum())+1;lp=0.
  for k,t in enumerate(tt):
   lp+=float(torch.log_softmax(z,-1)[t])
   if k==len(tt)-1:break
   am=torch.cat([am,am.new_ones(1,1)],1)
   o=model(input_ids=torch.tensor([[t]],device=DEV),attention_mask=am,past_key_values=past,use_cache=True);past=o.past_key_values;z=o.logits[0,-1].float()
 finally:remove(hs)
 return lp/len(tt),rank
def norm(s):return " "+re.sub(r"[^\w]+"," ",s.casefold()).strip()+" "
def hit(x,t):return int(norm(t) in norm(x))
# ---------------------------------------------------------------- equivalence of MEM_FULL vs CONTEXT (install fidelity)
with torch.inference_mode():
 q0=FACTS[0]["qs"][0][0];qids=enc(FMT.format(q=q0))
 pre,c,_=setup(0,"MEM_FULL");zm=model(input_ids=torch.tensor([qids],device=DEV),attention_mask=torch.ones(1,len(pre)+len(qids),device=DEV,dtype=torch.long),past_key_values=c).logits[0,-1].float()
 pre2,_,_=setup(0,"CONTEXT_DIAG");zc=model(input_ids=torch.tensor([pre2+qids],device=DEV)).logits[0,-1].float()
 GAPMC=float((zm-zc).abs().max());AGREE=bool(zm.argmax()==zc.argmax())
print(f"      MEM_FULL vs CONTEXT first-step logits: max|Δ|={GAPMC:.4f} argmax_same={AGREE}")
# ---------------------------------------------------------------- EVALUATION
print("[6/8] Evaluation (source removed; question only + installed representation)...")
R=[]
for i,f in enumerate(FACTS):
 print("\n"+"-"*112+f"\nF{i+1} memory frozen from: [source text not shown to readout] | T={MEM[i]['T']}")
 for q,a,who in f["qs"]:
  crit=int(norm(a) not in norm(q));print(f"\nQ: {q}\nTARGET(post-hoc only): {a} | critical(answer absent from question)={crit} | who={who}")
  tv,rv=tf(i,q,a,"VANILLA")
  for arm in ARMS:
   text=gen(i,q,arm);lp,rk=(tv,rv) if arm=="VANILLA" else tf(i,q,a,arm)
   h=hit(text,a);sh=hit(text,f["swap_ans"]) if arm.startswith("SWAP") else 0
   R.append(dict(f=i,q=q,a=a,who=who,crit=crit,arm=arm,text=text,hit=h,swap_hit=sh,dtf=lp-tv,rank=rk))
   print(f"{arm:14s} hit={h}"+(f" swap_hit={sh}" if arm.startswith("SWAP") else "        ")+f" rank={rk:6d} ΔTF={lp-tv:+7.3f} | {text}")
# ---------------------------------------------------------------- SUMMARY / DECISION
print("\n"+"="*112);print("[7/8] SUMMARY (critical questions only; hit = normalized answer substring)")
def rate(arm,key="hit",who_only=False):
 rs=[r for r in R if r["arm"]==arm and r["crit"] and (r["who"] or not who_only)];return sum(r[key] for r in rs)/max(1,len(rs)),len(rs)
for arm in ARMS:
 hr,n=rate(arm);wr,wn=rate(arm,who_only=True);dt=sum(r["dtf"] for r in R if r["arm"]==arm)/sum(r["arm"]==arm for r in R)
 extra=""
 if arm.startswith("SWAP"):
  fs=[r for r in R if r["arm"]==arm and r["who"]];follow=sum(r["swap_hit"] and not r["hit"] for r in fs)/max(1,len(fs));extra=f" swap_follow(who)={follow:.2f}"
 print(f"{arm:14s} retrieval={hr:.2f} ({int(round(hr*n))}/{n}) who-names={wr:.2f} mean_ΔTF={dt:+.3f}{extra}")
def follow(arm):
 fs=[r for r in R if r["arm"]==arm and r["who"]];return sum(r["swap_hit"] and not r["hit"] for r in fs)/max(1,len(fs))
CTRL=max(rate(a)[0] for a in ["VANILLA","MEM_TOKSHUF","MEM_LAYERSHUF","MEM_FOREIGN","RESID_250C"])
ok=lambda a:rate(a)[0]>=PASS_RATE and rate(a)[0]-CTRL>=PASS_MARGIN
ctx=rate("CONTEXT_DIAG")[0];full=rate("MEM_FULL")[0]
print(f"\ncontrol ceiling (max of VANILLA/TOKSHUF/LAYERSHUF/FOREIGN/RESID_250C) = {CTRL:.2f} | context diagnostic = {ctx:.2f}")
print(f"RESID_250C (single-direction TEST250 channel, same source) = {rate('RESID_250C')[0]:.2f}  <- capacity reference")
if (ok("MEM_R8") and follow("SWAP_R8")>=SWAP_PASS) or ok("MEM_R4"):
 dec="COMPRESSED_SYNTHETIC_MEMORY_SOURCE_SPECIFIC_RETRIEVAL"+(" (R4 passes)" if ok("MEM_R4") else " (R8 passes)")
elif ok("MEM_FULL") and follow("SWAP_FULL")>=SWAP_PASS:dec="NATIVE_KV_MEMORY_CHANNEL_VALIDATED__COMPRESSION_NOT_SURVIVED"
elif ctx>=PASS_RATE and full<PASS_RATE:dec="INSTALL_FAILURE__CONTEXT_WORKS_MEMORY_DOES_NOT"
elif ctx<PASS_RATE:dec="READOUT_FRAME_FAILS_EVEN_WITH_CONTEXT__NOT_A_MEMORY_RESULT"
else:dec="NO_RELIABLE_SOURCE_SPECIFIC_RETRIEVAL"
# ---------------------------------------------------------------- INTEGRITY
print("\n[8/8] INTEGRITY")
SHA1=sha([X for m in MEM+SWP for X in m["M"]])
print("forge inputs                  : source strings only | questions in forge: 0 | targets in forge: 0")
print("readout prompt                : QUESTION/ANSWER frame only (source absent; asserted every call; dummy prefix = PAD ids, never embedded)")
print("memory frozen (SHA pre==post) :","PASS" if SHA0==SHA1 else "FAIL",SHA0[:24])
print(f"KV install audit              : K={ek:.2e} V={ev:.2e} | MEM_FULL vs CONTEXT max|Δlogit|={GAPMC:.4f} argmax_same={AGREE}")
print(f"TEST250 reference arm         : L0–L19 CUDA frozen-norm final-row | RSS={RSS:.9f} | active hooks after eval={len(ACTIVE)}")
print("decoding                      : greedy, repetition_penalty=1.0 (all arms) | runtime attn:",cfg._attn_implementation)
print("weights frozen / fingerprint  :",all(not p.requires_grad for p in model.parameters()),"/","PASS" if fingerprint()==FP0 else "FAIL")
assert SHA0==SHA1 and not ACTIVE and fingerprint()==FP0
print("NOTE: MEM_FULL ≈ context by construction; only MEM_R8/MEM_R4 (+SWAP_R8) can support a synthetic-representation claim.")
print("DECISION:",dec);print("="*112)
