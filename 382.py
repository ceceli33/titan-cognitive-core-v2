# ==================================================================================================
# AKBASCORE — TEST 382 · READ-SPACE ROW-WISE LATENT MEMORY CODEC (RS-LMC)
# SOURCE -> read alone (fixed <|endoftext|> sink prefix, no question) -> per-row pre-RoPE K/V
# -> ENCODE with a FACT-INDEPENDENT model-derived codebook (PCA per layer×KV-head on 24 neutral sentences)
# -> FROZEN CODES -> SOURCE REMOVED -> decode + RoPE -> virtual KV slots -> QUESTION -> GREEDY
# Evidence -> design (vs TEST381):
#  * R8 "96% energy" was the sink row (‖sink‖≈15000 vs content≈70 at L14): energy metric was content-blind.
#  * Rank-r over TOKENS forces 17 distinct rows into r shared components -> rare-name blending ("Akbaa","Yildirim").
#    RS-LMC compresses along FEATURES per row (rows never mix) in the space attention actually reads (K/V = 1024 of 3584).
#  * Sink moved onto a fixed EOT prefix: position-0 state is causally identical for every source -> shared slot, 0 bytes/fact.
#  * Controls split: NULL(VANILLA, FOREIGN) | STRUCTURAL(TOKSHUF, LAYERSHUF, reported only) | CAUSAL(per-question
#    counterfactual source changing only the answer filler) | REFERENCE(CONTEXT_DIAG, MEM_FULL, R8_RES, R8_SINK, RESID_250C).
# Forge sees source strings only. Targets used only post-hoc. MEM_FULL ≈ context (channel reference, not a result).
# ==================================================================================================
!pip -q install -U transformers accelerate ninja
import os,math,re,hashlib,difflib,torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from torch.utils.cpp_extension import load_inline
os.environ["TOKENIZERS_PARALLELISM"]="false";torch.set_grad_enabled(False);assert torch.cuda.is_available()
DEV="cuda";MODEL="Qwen/Qwen2.5-7B-Instruct";TOTAL=28;H=3584;M250=20;MAXNEW=24;R_SVD=8;DS=(64,32,16);DMAX=64;LSHIFT=7
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;RSS250_REF=.250235055
FMT="QUESTION:\n{q}\n\nANSWER:";SEP="\n\n"
SUCC_E=.85;SUCC_CF=.75;SUCC_M=.50;PART_E=.40;PART_M=.30;NEAR=.80   # pre-registered
FACTS=[
 dict(src="Mustafa Akbaş planted the Turkish flag at the base of the Golden Gate Bridge.",qs=[
  ("who","Who planted the Turkish flag at the base of the Golden Gate Bridge?","Mustafa Akbaş","Leyla Demir planted the Turkish flag at the base of the Golden Gate Bridge.","Leyla Demir"),
  ("do","What did Mustafa Akbaş do at the base of the Golden Gate Bridge?","planted the Turkish flag","Mustafa Akbaş painted the iron railing at the base of the Golden Gate Bridge.","painted the iron railing"),
  ("what","What did Mustafa Akbaş plant at the base of the Golden Gate Bridge?","Turkish flag","Mustafa Akbaş planted the olive tree at the base of the Golden Gate Bridge.","olive tree"),
  ("where","Where did Mustafa Akbaş plant the Turkish flag?","base of the Golden Gate Bridge","Mustafa Akbaş planted the Turkish flag at the summit of the Brooklyn Bridge.","summit of the Brooklyn Bridge")]),
 dict(src="Elena Varga stored the silver compass inside the northern archive of Tallinn.",qs=[
  ("who","Who stored the silver compass inside the northern archive of Tallinn?","Elena Varga","Tomasz Brenner stored the silver compass inside the northern archive of Tallinn.","Tomasz Brenner"),
  ("what","What did Elena Varga store inside the northern archive of Tallinn?","silver compass","Elena Varga stored the copper sextant inside the northern archive of Tallinn.","copper sextant"),
  ("where","Where did Elena Varga store the silver compass?","northern archive","Elena Varga stored the silver compass inside the southern vault of Tallinn.","southern vault")]),
 dict(src="Kerem Yıldız repaired the brass lantern at the old lighthouse of Sinop.",qs=[
  ("who","Who repaired the brass lantern at the old lighthouse of Sinop?","Kerem Yıldız","Marta Solberg repaired the brass lantern at the old lighthouse of Sinop.","Marta Solberg"),
  ("what","What did Kerem Yıldız repair at the old lighthouse of Sinop?","brass lantern","Kerem Yıldız repaired the wooden rudder at the old lighthouse of Sinop.","wooden rudder"),
  ("where","Where did Kerem Yıldız repair the brass lantern?","old lighthouse","Kerem Yıldız repaired the brass lantern at the fishing harbor of Sinop.","fishing harbor")]),
 dict(src="Aiko Tanabe painted the blue signal beside the eastern platform of Kyoto Station.",qs=[
  ("who","Who painted the blue signal beside the eastern platform of Kyoto Station?","Aiko Tanabe","Diego Ferraz painted the blue signal beside the eastern platform of Kyoto Station.","Diego Ferraz"),
  ("what","What did Aiko Tanabe paint beside the eastern platform of Kyoto Station?","blue signal","Aiko Tanabe painted the red bench beside the eastern platform of Kyoto Station.","red bench"),
  ("where","Where did Aiko Tanabe paint the blue signal?","eastern platform","Aiko Tanabe painted the blue signal beside the western gate of Kyoto Station.","western gate")])]
CORPUS=["Jonas Weber carried the wooden crate across the quiet market square.","Priya Nair wrote a long letter to her cousin in the morning.",
"The small boat drifted slowly toward the rocky shore.","Omar Haddad fixed the broken clock in the village school.","A tired teacher closed the green door of the library.",
"Sofia Rossi baked fresh bread for the harvest festival.","The children watched the kites rising above the hill.","Liam O'Connor sold his old bicycle to a neighbor.",
"Heavy rain flooded the narrow street near the market.","Nadia Petrova translated the ancient manuscript into French.","The farmer counted the sheep before sunset.",
"Hiro Sato opened a tiny bakery beside the river.","An old dog slept under the kitchen table all afternoon.","Carlos Mendes washed the windows of his grandmother's house.",
"The museum guard locked the heavy gate at midnight.","Fatima Zahra grew tomatoes in the backyard garden.","The pilot announced a short delay because of fog.",
"Anna Kowalski found a lost wallet on the bus.","Snow covered the mountain village during the night.","Ravi Kumar taught his brother how to play chess.",
"The chef sharpened every knife before dinner service.","Lucas Martin cleaned the roof of the barn after the storm.","A young violinist practiced scales in the empty hall.",
"Mei Lin delivered the package to the wrong address."]
ALL_SRC=[f["src"] for f in FACTS]+[x[3] for f in FACTS for x in f["qs"]]
assert not any(c in s or s in c for c in CORPUS for s in ALL_SRC)
def unit(x,e=1e-8):return x/(x.float().norm()+e)
def rho(L):return IVME*((ZIRVE*math.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN))
RHO=[rho(i) for i in range(M250)];RSS=math.sqrt(sum(r*r for r in RHO));assert abs(RSS-RSS250_REF)<1e-5
print("="*112);print("TEST 382 — READ-SPACE ROW-WISE LATENT MEMORY CODEC · SOURCE-FREE COMPOSITIONAL RETRIEVAL");print("="*112)
print("GPU:",torch.cuda.get_device_name(0),"| Model:",MODEL)
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
print("[1/9] CUDA (TEST250 reference motor)...")
ext=load_inline("akbas382_seasc",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cuda_cflags=["-O3"],extra_cflags=["-O3"],verbose=False)
_x=torch.randn(1,3,8,device=DEV,dtype=torch.bfloat16);_a=unit(torch.randn(8,device=DEV));_y=ext.seasc(_x,_a,.1)
_r=_x.float().clone();_r[:,-1]+=.1*_x[:,-1].float().norm()*_a
assert torch.allclose(_y.float(),_r.to(torch.bfloat16).float(),atol=2e-2) and torch.equal(_y[:,:-1],_x[:,:-1]);del _x,_a,_y,_r;print("      PASS")
print("[2/9] Model...")
tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True)
try:model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
except TypeError:model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa").eval()
for p in model.parameters():p.requires_grad_(False)
cfg=model.config;layers=model.model.layers
NH=cfg.num_attention_heads;NKV=cfg.num_key_value_heads;HD=getattr(cfg,"head_dim",None) or H//NH;KVD=NKV*HD
assert len(layers)==TOTAL and cfg.hidden_size==H and NH==28 and NKV==4 and HD==128 and not getattr(cfg,"use_sliding_window",False)
PAD=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_ge=model.generation_config.eos_token_id;EOS=sorted({tok.eos_token_id,*(_ge if isinstance(_ge,(list,tuple)) else [_ge])}-{None})
GEN=dict(max_new_tokens=MAXNEW,do_sample=False,repetition_penalty=1.0,use_cache=True,pad_token_id=PAD,eos_token_id=EOS)
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
def fingerprint():return tuple(float(t.sum(dtype=torch.float32)) for t in FP_T)
FP0=fingerprint();print(f"      L={TOTAL} H={H} Q={NH} KV={NKV} head={HD} attn={cfg._attn_implementation} sink/PAD id={PAD}")
def enc(s):return tok(s,add_special_tokens=False).input_ids
# ---------------------------------------------------------------- FORGE (source only)
@torch.inference_mode()
def forge(s):
 assert "QUESTION" not in s
 ids=[PAD]+enc(s+SEP);o=model(input_ids=torch.tensor([ids],device=DEV),output_hidden_states=True,use_cache=False,return_dict=True);M=[];K=[];V=[]
 for L in range(TOTAL):
  hL=o.hidden_states[L][0];a=layers[L].self_attn;z=layers[L].input_layernorm(hL)          # hidden_states[L] = input of layer L
  M.append(hL.clone());K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())   # pre-RoPE K, V: [T,512]
 last=[o.hidden_states[L+1][0,-1].float().clone() for L in range(M250)];del o
 return dict(ids=ids,T=len(ids),M=M,K=K,V=V,last=last)
@torch.inference_mode()
def kv_from_resid(M):
 K=[];V=[]
 for L in range(TOTAL):
  a=layers[L].self_attn;z=layers[L].input_layernorm(M[L]);K.append(a.k_proj(z).contiguous());V.append(a.v_proj(z).contiguous())
 return K,V
@torch.inference_mode()
def install(K,V):
 T=K[0].shape[0];cos,sin=model.model.rotary_emb(K[0][None],torch.arange(T,device=DEV)[None]);KK=[];VV=[]
 for L in range(TOTAL):
  k=K[L].reshape(1,T,NKV,HD).transpose(1,2);v=V[L].reshape(1,T,NKV,HD).transpose(1,2)
  KK.append(apply_rotary_pos_emb(k,k,cos,sin,unsqueeze_dim=1)[1].contiguous());VV.append(v.contiguous())
 return (tuple(KK),tuple(VV),T)
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
print("[3/9] Forge: facts, per-question counterfactual sources, neutral codebook corpus (no questions, no targets)...")
FF=[forge(f["src"]) for f in FACTS];CF={(i,j):forge(x[3]) for i,f in enumerate(FACTS) for j,x in enumerate(f["qs"])};CO=[forge(s) for s in CORPUS]
SINK={"K":[CO[0]["K"][L][:1] for L in range(TOTAL)],"V":[CO[0]["V"][L][:1] for L in range(TOTAL)],"M":[CO[0]["M"][L][:1] for L in range(TOTAL)]}
sdev=max(float((f[n][L][0].float()-SINK[n][L][0].float()).abs().max()) for f in FF+list(CF.values()) for n in ("K","V") for L in range(TOTAL))
print(f"      sink slot identical across sources: max|Δ|={sdev:.2e} (causal position-0 invariance)")
# ---------------------------------------------------------------- INSTALL AUDIT
with torch.inference_mode():
 nat=model(input_ids=torch.tensor([FF[0]["ids"]],device=DEV),use_cache=True).past_key_values;kv=install(FF[0]["K"],FF[0]["V"])
 ek=max(float((kv[0][L].float()-kvget(nat,L)[0].float()).norm()/kvget(nat,L)[0].float().norm()) for L in range(TOTAL))
 ev=max(float((kv[1][L].float()-kvget(nat,L)[1].float()).norm()/kvget(nat,L)[1].float().norm()) for L in range(TOTAL));del nat
print(f"[4/9] KV install audit vs native cache: K={ek:.2e} V={ev:.2e}");assert ek<5e-2 and ev<5e-2
# ---------------------------------------------------------------- CODEBOOK (fact-independent) + CODECS
def fit_codebook():
 CB=[]
 for L in range(TOTAL):
  e={}
  for n in ("K","V"):
   R=torch.cat([c[n][L][1:] for c in CO]).float().view(-1,NKV,HD);mus=[];Bs=[];evs=[]
   for h in range(NKV):
    X=R[:,h];mu=X.mean(0);_,S,Vh=torch.linalg.svd(X-mu,full_matrices=False);mus.append(mu);Bs.append(Vh[:DMAX].T.contiguous());evs.append((S**2).cumsum(0)/(S**2).sum())
   e[n]=(torch.stack(mus),torch.stack(Bs),torch.stack([v[:DMAX] for v in evs]))
  CB.append(e)
 return CB
print("[5/9] Codebook: PCA per layer×KV-head on",sum(c["T"]-1 for c in CO),"neutral content rows...")
CB=fit_codebook()
for d in DS:print(f"      d={d:2d}: corpus variance explained K={float(torch.stack([CB[L]['K'][2][:,d-1] for L in range(TOTAL)]).mean()):.3f} V={float(torch.stack([CB[L]['V'][2][:,d-1] for L in range(TOTAL)]).mean()):.3f}")
def encode(f,d):
 return {n:[torch.einsum('thi,hid->thd',f[n][L][1:].float().view(-1,NKV,HD)-CB[L][n][0],CB[L][n][1][:,:,:d]).to(torch.bfloat16) for L in range(TOTAL)] for n in ("K","V")}
def decode(code,d):
 out={}
 for n in ("K","V"):
  out[n]=[torch.cat([SINK[n][L],(CB[L][n][0]+torch.einsum('thd,hid->thi',code[n][L].float(),CB[L][n][1][:,:,:d])).reshape(-1,KVD).to(torch.bfloat16)]) for L in range(TOTAL)]
 return out["K"],out["V"]
def lowrank(M,r,sink):
 out=[]
 for X in M:
  Y=X[1:] if sink else X;U,S,Vh=torch.linalg.svd(Y.float(),full_matrices=False);r_=min(r,S.numel());R=((U[:,:r_]*S[:r_])@Vh[:r_]).to(X.dtype)
  out.append(torch.cat([X[:1],R]) if sink else R)
 return out
def tokshuf(K,V):return [torch.cat([X[:1],X[1:].flip(0)]) for X in K],[torch.cat([X[:1],X[1:].flip(0)]) for X in V]
def layershuf(K,V):return [K[(L+LSHIFT)%TOTAL] for L in range(TOTAL)],[V[(L+LSHIFT)%TOTAL] for L in range(TOTAL)]
def relerr(A,B):return float(torch.stack([(a.float()-b.float()).norm(dim=1)/b.float().norm(dim=1).clamp_min(1e-6) for a,b in zip(A,B)]).mean())
print("[6/9] Representations (frozen) + source-only mechanism diagnostics...")
N=len(FACTS);KV={};CFKV={};CODES=[];STORE={}
for i,f in enumerate(FF):
 T=f["T"];c={d:encode(f,d) for d in DS};CODES+= [x for d in DS for n in ("K","V") for x in c[d][n]]
 dk={d:decode(c[d],d) for d in DS};r8=kv_from_resid(lowrank(f["M"],R_SVD,False));r8s=kv_from_resid(lowrank(f["M"],R_SVD,True))
 KV[i]=dict(MEM_FULL=install(f["K"],f["V"]),R8_RES=install(*r8),R8_SINK=install(*r8s),**{f"PKV{d}":install(*dk[d]) for d in DS},
            PKV32_TOKSHUF=install(*tokshuf(*dk[32])),PKV32_LAYERSHUF=install(*layershuf(*dk[32])))
 se=[float(X[0].float().norm()**2/X.float().norm()**2) for X in f["M"]];Sc=[torch.linalg.svdvals(X[1:].float()) for X in f["M"]]
 pr=sum(float(s.square().sum()**2/s.pow(4).sum()) for s in Sc)/TOTAL
 Mr=lowrank(f["M"],R_SVD,False);cerr=sum(float((a[1:].float()-b[1:].float()).norm()/b[1:].float().norm()) for a,b in zip(Mr,f["M"]))/TOTAL
 rowe=((Mr[14][1:].float()-f["M"][14][1:].float()).norm(dim=1)/f["M"][14][1:].float().norm(dim=1));worst=[tok.decode([f["ids"][1+k]]) for k in rowe.topk(3).indices.tolist()]
 print(f"      F{i+1} T={T}: sink energy share={sum(se)/TOTAL:.4f} | content eff.rank(PR)={pr:.1f}/{T-1} | R8_RES content-row rel.err={cerr:.3f} | L14 worst rows={worst}")
 print("           PKV row rel.err K/V: "+" ".join(f"d{d}={relerr([x[1:] for x in dk[d][0]],[x[1:] for x in f['K']]):.3f}/{relerr([x[1:] for x in dk[d][1]],[x[1:] for x in f['V']]):.3f}" for d in DS))
 STORE[i]=dict(T=T,RESID=TOTAL*T*H*2,KVNAT=TOTAL*T*KVD*2,R8_RES=TOTAL*R_SVD*(T+H)*2,R8_SINK=TOTAL*R_SVD*(T-1+H)*2,**{f"PKV{d}":TOTAL*(T-1)*2*NKV*d*2 for d in DS})
for i in range(N):KV[i]["FOREIGN_FULL"]=KV[(i+1)%N]["MEM_FULL"];KV[i]["FOREIGN_PKV32"]=KV[(i+1)%N]["PKV32"]
for (i,j),g in CF.items():
 CFKV[(i,j)]=dict(FULL=install(g["K"],g["V"]),**{f"PKV{d}":install(*decode(encode(g,d),d)) for d in DS})
RP=[]
for i in range(N):
 oth=[FF[j]["last"] for j in range(N) if j!=i];RP.append(tuple(unit(FF[i]["last"][L]-torch.stack([o[L] for o in oth]).mean(0)).contiguous() for L in range(M250)))
def kvsha():return hashlib.sha256(b"".join(t.float().cpu().numpy().tobytes() for d_ in list(KV.values())+list(CFKV.values()) for kv in d_.values() for t in kv[0]+kv[1])+b"".join(t.float().cpu().numpy().tobytes() for t in CODES)).hexdigest()
SHA0=kvsha()
print("      storage per fact (bf16) as fraction of residual memory | of native KV:")
for k in ["KVNAT","R8_RES","R8_SINK"]+[f"PKV{d}" for d in DS]:
 print(f"        {k:8s} {sum(STORE[i][k]/STORE[i]['RESID'] for i in range(N))/N:.4f} | {sum(STORE[i][k]/STORE[i]['KVNAT'] for i in range(N))/N:.4f}")
print(f"      shared codebook (one-time, fact-independent): {TOTAL*2*NKV*(HD*DMAX+HD)*4/1e6:.2f} MB fp32")
# ---------------------------------------------------------------- RUNTIME
ACTIVE=[]
def hk_install(P):
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
ARMS=["VANILLA","CONTEXT_DIAG","MEM_FULL","CF_FULL","FOREIGN_FULL","R8_RES","R8_SINK","PKV64","PKV32","PKV16","CF_PKV64","CF_PKV32","CF_PKV16",
      "FOREIGN_PKV32","PKV32_TOKSHUF","PKV32_LAYERSHUF","RESID_250C"]
def spec(i,j,arm):
 t,q,a,cs,ca=FACTS[i]["qs"][j]
 if arm=="VANILLA":return ("none",None,a)
 if arm=="CONTEXT_DIAG":return ("ctx",FACTS[i]["src"],a)
 if arm=="RESID_250C":return ("resid",RP[i],a)
 if arm.startswith("CF_"):return ("mem",CFKV[(i,j)][arm[3:]],ca)
 return ("mem",KV[i][arm],a)
def setup(sp):
 kind,pl,_=sp;pre=[];cache=None;hs=[]
 if kind=="ctx":pre=[PAD]+enc(pl+SEP)
 elif kind=="mem":cache=mk_cache(pl);pre=[PAD]*pl[2]
 elif kind=="resid":hs=hk_install(pl)
 return pre,cache,hs
def leak(sp,pre,qids):
 if sp[0]=="ctx":return
 txt=tok.decode(pre+qids);assert all(s not in txt for s in ALL_SRC) and set(pre)<={PAD},"SOURCE LEAK INTO READOUT"
@torch.inference_mode()
def gen(q,sp):
 qids=enc(FMT.format(q=q));pre,cache,hs=setup(sp);leak(sp,pre,qids);ids=torch.tensor([pre+qids],device=DEV)
 try:y=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,**GEN)
 finally:remove(hs)
 return tok.decode(y[0,ids.shape[1]:],skip_special_tokens=True).strip()
@torch.inference_mode()
def first_logits(q,sp):
 qids=enc(FMT.format(q=q));pre,cache,hs=setup(sp)
 try:
  ids=torch.tensor([qids if cache is not None else pre+qids],device=DEV);am=torch.ones(1,len(pre)+len(qids),device=DEV,dtype=torch.long)
  return model(input_ids=ids,attention_mask=am,past_key_values=cache,use_cache=True)
 finally:remove(hs)
@torch.inference_mode()
def tf(q,sp):
 a=sp[2];qt=FMT.format(q=q);qids=enc(qt);full=enc(qt+" "+a);assert full[:len(qids)]==qids;tt=full[len(qids):]
 pre,cache,hs=setup(sp);leak(sp,pre,qids)
 try:
  ids=torch.tensor([qids if cache is not None else pre+qids],device=DEV);am=torch.ones(1,len(pre)+len(qids),device=DEV,dtype=torch.long)
  o=model(input_ids=ids,attention_mask=am,past_key_values=cache,use_cache=True);past=o.past_key_values;z=o.logits[0,-1].float();rank=int((z>z[tt[0]]).sum())+1;lp=0.
  for k,t in enumerate(tt):
   lp+=float(torch.log_softmax(z,-1)[t])
   if k==len(tt)-1:break
   am=torch.cat([am,am.new_ones(1,1)],1);o=model(input_ids=torch.tensor([[t]],device=DEV),attention_mask=am,past_key_values=past,use_cache=True);past=o.past_key_values;z=o.logits[0,-1].float()
 finally:remove(hs)
 return lp/len(tt),rank
def norm(s):return re.sub(r"[^\w]+"," ",s.casefold()).strip()
def hit(x,t):return int(f" {norm(t)} " in f" {norm(x)} ")
def near(x,t):
 xs=norm(x).split();ts=norm(t);n=len(ts.split())
 return max([difflib.SequenceMatcher(None," ".join(xs[k:k+n]),ts).ratio() for k in range(max(1,len(xs)-n+1))] or [0.])
# ---------------------------------------------------------------- MEM_FULL vs CONTEXT equivalence (all questions)
print("[7/9] MEM_FULL vs CONTEXT first-step equivalence over all questions...")
EQ=[]
for i,f in enumerate(FACTS):
 for j,x in enumerate(f["qs"]):
  zm=first_logits(x[1],("mem",KV[i]["MEM_FULL"],None)).logits[0,-1].float();zc=first_logits(x[1],("ctx",f["src"],None)).logits[0,-1].float()
  pc=torch.log_softmax(zc,-1);pm=torch.log_softmax(zm,-1)
  EQ.append((float((zm-zc).abs().max()),float(F.cosine_similarity(zm,zc,0)),float((zm-zc).norm()/zc.norm()),float((pc.exp()*(pc-pm)).sum()),
             len(set(zm.topk(10).indices.tolist())&set(zc.topk(10).indices.tolist()))/10,int(zm.argmax()==zc.argmax())))
m=lambda k:sum(e[k] for e in EQ)/len(EQ)
print(f"      max|Δlogit| mean/max={m(0):.3f}/{max(e[0] for e in EQ):.3f} | cos={m(1):.6f} | relRMS={m(2):.4f} | KL(ctx‖mem)={m(3):.2e} | top10 overlap={m(4):.3f} | argmax agree={m(5):.3f}")
# ---------------------------------------------------------------- EVALUATION
print("[8/9] Evaluation (source removed; question + installed representation)...")
R=[]
for i,f in enumerate(FACTS):
 for j,(t,q,a,cs,ca) in enumerate(f["qs"]):
  print("\n"+"-"*112+f"\nF{i+1}.{j+1} [{t}] Q: {q}\n  target(post-hoc)={a!r} | counterfactual target={ca!r} | answer-in-question={int(f' {norm(a)} ' in f' {norm(q)} ')}/{int(f' {norm(ca)} ' in f' {norm(q)} ')}")
  for arm in ARMS:
   sp=spec(i,j,arm);text=gen(q,sp);lp,rk=tf(q,sp);tg=sp[2]
   r=dict(i=i,j=j,type=t,arm=arm,text=text,exact=hit(text,tg),near=near(text,tg),orig=hit(text,a),lp=lp,rank=rk);R.append(r)
   print(f"  {arm:16s} exact={r['exact']} near={r['near']:.2f}"+(f" orig={r['orig']}" if arm.startswith("CF_") else "       ")+f" rank={rk:6d} TF={lp:+7.3f} | {text}")
# ---------------------------------------------------------------- SUMMARY / DECISION
print("\n"+"="*112);print("[9/9] SUMMARY (13 critical questions; exact = normalized substring; near = diagnostic only)")
def rows(a):return [r for r in R if r["arm"]==a]
def rate(a):rs=rows(a);return sum(r["exact"] for r in rs)/len(rs)
def nearr(a):rs=rows(a);return sum((r["near"]>=NEAR and not r["exact"]) for r in rs)/len(rs)
def follow(a):rs=rows(a);return sum(r["exact"] and not r["orig"] for r in rs)/len(rs)
def by(a,t):rs=[r for r in rows(a) if r["type"]==t];return f"{sum(r['exact'] for r in rs)}/{len(rs)}"
GROUP={"NULL":["VANILLA","FOREIGN_FULL","FOREIGN_PKV32"],"REFERENCE":["CONTEXT_DIAG","MEM_FULL","R8_RES","R8_SINK","RESID_250C"],
       "CANDIDATE":["PKV64","PKV32","PKV16"],"CAUSAL":["CF_FULL","CF_PKV64","CF_PKV32","CF_PKV16"],"STRUCTURAL":["PKV32_TOKSHUF","PKV32_LAYERSHUF"]}
for g,arms in GROUP.items():
 print(f"\n{g}")
 for a in arms:
  x=f"  {a:16s} exact={rate(a):.2f} near-miss={nearr(a):.2f} meanTF={sum(r['lp'] for r in rows(a))/len(rows(a)):+.3f} who/do/what/where={by(a,'who')}/{by(a,'do')}/{by(a,'what')}/{by(a,'where')}"
  if g=="CAUSAL":x+=f" follow={follow(a):.2f}"
  print(x)
NULL=max(rate(a) for a in GROUP["NULL"]);CTX=rate("CONTEXT_DIAG");FULLOK=rate("MEM_FULL")>=SUCC_E and follow("CF_FULL")>=SUCC_CF
def cls(a):
 e=rate(a);cf=follow("CF_"+a) if "CF_"+a in ARMS else None
 if cf is not None and e>=SUCC_E and cf>=SUCC_CF and e-NULL>=SUCC_M:return "SUCCESS"
 if e>=PART_E and e-NULL>=PART_M:return "PARTIAL"
 return "FAIL"
print(f"\nnull ceiling={NULL:.2f} | context upper bound={CTX:.2f} | channel (MEM_FULL+CF_FULL) valid={FULLOK}")
for a in ["R8_RES","R8_SINK"]+GROUP["CANDIDATE"]:
 st=sum(STORE[i][a]/STORE[i]["RESID"] for i in range(N))/N
 print(f"  {a:8s} class={cls(a):8s} storage={st:.4f}×residual"+("" if a.startswith("PKV") else "  (no causal arm: max class PARTIAL)"))
succ=[a for a in GROUP["CANDIDATE"] if cls(a)=="SUCCESS"];part=[a for a in ["R8_RES","R8_SINK"]+GROUP["CANDIDATE"] if cls(a)=="PARTIAL"]
if CTX<SUCC_E:dec="READOUT_FRAME_LIMIT__CONTEXT_ITSELF_FAILS"
elif not FULLOK:dec="CHANNEL_NOT_VALIDATED__MEM_FULL_OR_CF_FULL_FAILS"
elif succ:
 best=min(succ,key=lambda a:int(a[3:]));dec=f"COMPACT_SOURCE_SPECIFIC_MEMORY_SUCCESS @ {best} ({sum(STORE[i][best]/STORE[i]['RESID'] for i in range(N))/N:.3f}× residual, counterfactual-following)"
elif part:dec="PARTIAL_COMPACT_MEMORY: "+",".join(part)
else:dec="COMPRESSION_BREAKS_RETRIEVAL__CHANNEL_VALID_ONLY_AT_FULL"
print("\nINTEGRITY")
SHA1=kvsha()
print("  forge inputs        : source strings only (facts, counterfactuals, neutral corpus) | questions/targets in forge: 0")
print("  readout             : QUESTION/ANSWER frame; source absent (asserted); dummy prefix = PAD ids (never embedded)")
print(f"  representations     : frozen SHA pre==post {'PASS' if SHA0==SHA1 else 'FAIL'} {SHA0[:20]}")
print(f"  install audit       : K={ek:.2e} V={ev:.2e} | sink invariance max|Δ|={sdev:.2e}")
print(f"  TEST250 motor       : RSS={RSS:.9f} | active hooks after eval={len(ACTIVE)}")
print("  decoding            : greedy, repetition_penalty=1.0 | attn",cfg._attn_implementation,"| weights frozen",all(not p.requires_grad for p in model.parameters()),"| fingerprint","PASS" if fingerprint()==FP0 else "FAIL")
assert SHA0==SHA1 and not ACTIVE and fingerprint()==FP0
print("  NOTE: MEM_FULL/CF_FULL/CONTEXT are channel references. Only PKV* with their CF arms can support a compact-memory claim.")
print("DECISION:",dec);print("="*112)
