# TEST656 — AKBASCORE MAM · NUMERIC ASSOCIATIVE ADDRESSING · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='656';SEED656=6502026;N656=8;CUT656=3;MAX_NEW656=16
ROOT656=Path('/content/AKBASCORE_TEST656');ROOT656.mkdir(parents=True,exist_ok=True)
URL656='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB656='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND656='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob656(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate656(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST656 — NATIVE CUDA DYNAMIC KV FABRIC / SOURCE-LOCKED QWEN\n'+'='*116)
gate656('CUDA_AVAILABLE',torch.cuda.is_available())
if importlib.util.find_spec('ninja') is None:subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
nvcc656=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin'/'nvcc') if CUDA_HOME else None)
gate656('NINJA',importlib.util.find_spec('ninja') is not None)
gate656('NVCC',bool(nvcc656 and Path(nvcc656).is_file()))
gate656('CXX',bool(shutil.which('c++') or shutil.which('g++')))
os.environ.setdefault('MAX_JOBS','2')
# Compile native extension BEFORE downloading model weights.
CPP656=r'''
#include <torch/extension.h>
void append_kv_cuda(torch::Tensor dst,torch::Tensor prefix,torch::Tensor body);
void append_kv(torch::Tensor dst,torch::Tensor prefix,torch::Tensor body){
 TORCH_CHECK(dst.is_cuda()&&prefix.is_cuda()&&body.is_cuda(),"CUDA required");
 TORCH_CHECK(dst.scalar_type()==torch::kBFloat16&&prefix.scalar_type()==torch::kBFloat16&&body.scalar_type()==torch::kBFloat16,"BF16 only");
 TORCH_CHECK(dst.is_contiguous()&&prefix.is_contiguous()&&body.is_contiguous(),"contiguous required");
 TORCH_CHECK(dst.dim()==4&&prefix.dim()==4&&body.dim()==4,"[B,KVH,T,HD] required");
 TORCH_CHECK(dst.size(0)==prefix.size(0)&&dst.size(0)==body.size(0)&&dst.size(1)==prefix.size(1)&&dst.size(1)==body.size(1)&&dst.size(3)==prefix.size(3)&&dst.size(3)==body.size(3),"axis mismatch");
 TORCH_CHECK(dst.size(2)==prefix.size(2)+body.size(2),"length mismatch");
 TORCH_CHECK(dst.device()==prefix.device()&&dst.device()==body.device(),"device mismatch");
 append_kv_cuda(dst,prefix,body);
}
void cosine_cuda(torch::Tensor out,torch::Tensor query,torch::Tensor candidates);
void cosine_scores(torch::Tensor out,torch::Tensor query,torch::Tensor candidates){
 TORCH_CHECK(out.is_cuda()&&query.is_cuda()&&candidates.is_cuda(),"CUDA required");
 TORCH_CHECK(out.is_contiguous()&&query.is_contiguous()&&candidates.is_contiguous(),"contiguous required");
 TORCH_CHECK(out.scalar_type()==torch::kFloat32&&query.scalar_type()==torch::kFloat32&&candidates.scalar_type()==torch::kFloat32,"FP32 required");
 TORCH_CHECK(query.dim()==1&&candidates.dim()==2&&out.dim()==1&&candidates.size(1)==query.size(0)&&out.size(0)==candidates.size(0),"score shape mismatch");
 cosine_cuda(out,query,candidates);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("append_kv",&append_kv,"Native BF16 KV append");m.def("cosine_scores",&cosine_scores,"Native latent cosine ranking");}
'''
CUDA656=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
__global__ void append_kernel(__nv_bfloat16* out,const __nv_bfloat16* pre,const __nv_bfloat16* body,
 int B,int H,int T0,int T1,int D){
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 long long n=(long long)B*H*(T0+T1)*D;
 if(i>=n)return;
 int d=i%D;long long j=i/D;int t=j%(T0+T1);long long bh=j/(T0+T1);
 long long src=((bh*(t<T0?T0:T1))+(t<T0?t:t-T0))*D+d;
 out[i]=(t<T0)?pre[src]:body[src];
}
void append_kv_cuda(torch::Tensor dst,torch::Tensor prefix,torch::Tensor body){
 int B=dst.size(0),H=dst.size(1),T0=prefix.size(2),T1=body.size(2),D=dst.size(3);
 long long n=dst.numel();
 append_kernel<<<(n+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
 reinterpret_cast<__nv_bfloat16*>(dst.data_ptr<at::BFloat16>()),
 reinterpret_cast<const __nv_bfloat16*>(prefix.data_ptr<at::BFloat16>()),
 reinterpret_cast<const __nv_bfloat16*>(body.data_ptr<at::BFloat16>()),B,H,T0,T1,D);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void cosine_kernel(float* out,const float* q,const float* c,int rows,int dim){
 int r=blockIdx.x;if(r>=rows)return;
 __shared__ float dots[256],qs[256],cs[256];int t=threadIdx.x;float d=0,a=0,b=0;
 for(int j=t;j<dim;j+=blockDim.x){float x=q[j],y=c[(long long)r*dim+j];d+=x*y;a+=x*x;b+=y*y;}
 dots[t]=d;qs[t]=a;cs[t]=b;__syncthreads();
 for(int step=128;step>0;step>>=1){if(t<step){dots[t]+=dots[t+step];qs[t]+=qs[t+step];cs[t]+=cs[t+step];}__syncthreads();}
 if(t==0)out[r]=dots[0]/sqrtf(fmaxf(qs[0]*cs[0],1e-30f));
}
void cosine_cuda(torch::Tensor out,torch::Tensor query,torch::Tensor candidates){
 cosine_kernel<<<candidates.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<float>(),query.data_ptr<float>(),candidates.data_ptr<float>(),candidates.size(0),query.size(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
print('BUILDING NATIVE KV EXTENSION',flush=True)
native656=load_inline(name='akbascore_kvfabric656_v1',cpp_sources=[CPP656],cuda_sources=[CUDA656],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
@torch.no_grad()
def splice656(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native656.append_kv(out,a,b);return out
for t0,t1 in ((29,7),(43,31),(1,1)):
 x=torch.randn((1,4,t0,128),device='cuda',dtype=torch.bfloat16);y=torch.randn((1,4,t1,128),device='cuda',dtype=torch.bfloat16)
 gate656('NATIVE_BITWISE_'+str(t0)+'_'+str(t1),torch.equal(splice656(x,y),torch.cat((x,y),dim=2)))
# Load exactly the same verified baseline as TEST650.
p=Path('/content/566.py')
if p.is_file() and blob656(p.read_bytes())==BLOB656:raw=p.read_bytes();origin='LOCAL_VERIFIED'
else:
 req=urllib.request.Request(URL656,headers={'User-Agent':'AKBASCORE-TEST656'})
 with urllib.request.urlopen(req,timeout=120) as f:raw=f.read()
 origin='GITHUB_VERIFIED'
gate656('TEST566_BLOB',blob656(raw)==BLOB656)
src=raw.decode('utf-8');gate656('TEST566_BOUNDARY',src.count(BOUND656)==1)
exec(compile(src.split(BOUND656,1)[0],'566.py','exec'),globals())
gate656('FROZEN',all(not p.requires_grad for p in model.parameters()))
gate656('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate656('PANEL_LOCK',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45')
gate656('SENTINEL',sentinel()==S0)
print('BASE',origin,'BLOB',blob656(raw),'GPU',torch.cuda.get_device_name(0),flush=True)
# Same panel generator and random seed as TEST650: identical A/A' and four competing B facts.
rng=random.Random(SEED656);alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';seen=set()
def word656():
 while True:
  s=''.join(rng.choice(alphabet) for _ in range(9)).capitalize()
  if s not in seen:seen.add(s);return s
PANEL656=[];CAR656=[]
def add656(s):
 ids=tok(source(s),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('PREFIX_MISMATCH')
 i=len(CAR656);CAR656.append(dict(fact=s,body=ids[P:]));return i
for i in range(N656):
 obj=word656();owners=[word656() for _ in range(4)];regions=[word656() for _ in range(4)]
 a=add656(f'The owner of artifact {obj} is {owners[0]}.')
 acf=add656(f'The owner of artifact {obj} is {owners[1]}.')
 bs=[add656(f'{owners[j]} lives in region {regions[j]}.') for j in range(4)]
 PANEL656.append(dict(case=i,object=obj,owners=owners,regions=regions,a=a,acf=acf,bs=bs,q=f'In which region does the owner of artifact {obj} live?',qa=f'Who owns artifact {obj}?'))
CAR=CAR656;MAX_NEW=MAX_NEW656
panel_sha=hashlib.sha256(json.dumps(PANEL656,sort_keys=True).encode()).hexdigest()
gate656('TEST650_PANEL_LOCK',panel_sha=='db8b211abed216793bf2e4010eb006c862a74202d7512ae53a8a463fe929d466')
@torch.no_grad()
def native_append656(memory,ci):
 # Preserve the locked TEST566 upper-layer computation and H3 injection exactly.
 # Only lower-layer K/V concatenation is replaced with native CUDA, after RoPE position alignment.
 cp=checkpoint(ci,CUT656);oldn=memory[0][0].shape[-2];q=cp['body']
 pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp['h'][:,P:,:]
 if body.shape!=(1,q,H):raise RuntimeError('H3_BODY_SHAPE')
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError('H3_INJECTION_SHAPE')
  return replace(out,body)
 hook=model.model.layers[CUT656].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  for li in range(CUT656+1):
   ok,ov=memory[li];k,v=cp['kv'][li]
   k=rephase(k[:,:,P:,:],cp['pos'][P:],pos);v=v[:,:,P:,:]
   grown[li]=(splice656(ok,k),splice656(ov,v))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('PREFIX_MUTATED')
  if len(result)!=NL or any(k.shape!=(1,KVH,oldn+q,HD) or v.shape!=(1,KVH,oldn+q,HD) for k,v in result):raise RuntimeError('CACHE_SHAPE')
  return result
 finally:hook.remove()
@torch.no_grad()
def native_memory656(keys):
 mem=init(keys[0],CUT656)
 for ci in keys[1:]:mem=native_append656(mem,ci)
 return mem
@torch.no_grad()
def baseline_memory656(keys):
 mem=init(keys[0],CUT656)
 for ci in keys[1:]:mem=append(mem,ci,CUT656)
 return mem
@torch.no_grad()
def independent656(keys):
 mem=None
 for ci in keys:
  cp=checkpoint(ci,CUT656);kv=cp['kv']
  if mem is None:mem=kv;continue
  oldn=mem[0][0].shape[-2];q=len(CAR[ci]['body'])
  oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
  mem=tuple((splice656(ok,rephase(k[:,:,P:,:],oldpos,newpos)),splice656(ov,v[:,:,P:,:])) for (ok,ov),(k,v) in zip(mem,kv))
 return mem
@torch.no_grad()
def joint656(keys):
 ids=list(PIDS)
 for ci in keys:ids.extend(CAR[ci]['body'])
 n=len(ids);pos=torch.arange(n,device=DEVICE)
 out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 return clone(out.past_key_values)
def validate656(mem,keys):
 n=P+sum(len(CAR[i]['body']) for i in keys)
 if len(mem)!=NL or any(k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD) or not torch.isfinite(k).all() or not torch.isfinite(v).all() for k,v in mem):raise RuntimeError('INVALID_CACHE')
 return mem
def exact656(a,b):return re.sub(r'[^a-z]','',a.casefold())==re.sub(r'[^a-z]','',b.casefold())


# TEST656 — numerical routing diagnostics; locked TEST655 native CUT3 engine unchanged.
# This is NOT learned routing. Candidate ranks use numerical tensors only; no answer or region text.
import math
T656=time.perf_counter();RESULTS656=[];ROUTES656=[];ERRORS656=[]
def extract656(ans,regions):
 hits=[v for v in regions if re.search(r'(?<![A-Za-z])'+re.escape(v)+r'(?![A-Za-z])',ans,re.IGNORECASE)]
 return hits[0] if len(hits)==1 else None
def grade656(ans,gold,regions):
 extracted=extract656(ans,regions)
 return dict(strict=exact656(ans,gold),region=extracted is not None and extracted.casefold()==gold.casefold(),extracted=extracted)
@torch.no_grad()
def h3_signature656(ci):
 cp=checkpoint(ci,CUT656);h=cp['h'][:,P:,:].float()
 if h.ndim!=3 or h.shape[-1]!=H or not h.shape[1]:raise RuntimeError('H3_SIGNATURE_SHAPE')
 return h.mean(1).squeeze(0).contiguous()
@torch.no_grad()
def native_scores656(query,candidates):
 out=torch.empty((len(candidates),),device=DEVICE,dtype=torch.float32)
 native656.cosine_scores(out,query.float().contiguous(),torch.stack([c.float() for c in candidates]).contiguous())
 return out.cpu().tolist()
@torch.no_grad()
def kv_signature656(ci):
 # Source-local unrotated K/V body, flattened over KV heads; no lexical matching.
 cp=checkpoint(ci,CUT656);k,v=cp['kv'][CUT656]
 k=k[:,:,P:,:].float();v=v[:,:,P:,:].float()
 if k.shape[2]==0:raise RuntimeError('KV_EMPTY')
 # k carries RoPE; V is position-independent. Use V for position-safe numeric address.
 return v.mean(dim=2).reshape(-1).contiguous()
@torch.no_grad()
def projected_query656(a_ci,q_ci):
 # Numeric query conditioning: project A+question H3 through the frozen CUT3 value projection.
 # This is a candidate retrieval probe, not proof of attention-based memory addressing.
 ah=h3_signature656(a_ci);qh=h3_signature656(q_ci)
 h=(qh+ah).to(device=DEVICE,dtype=DTYPE).reshape(1,1,H)
 proj=model.model.layers[CUT656].self_attn.v_proj(h).float().reshape(-1).contiguous()
 return proj
@torch.no_grad()
def conditioned_h3656(a_ci,q_ci):
 # Counterfactual-sensitive numerical query from A and question, compared to candidate H3.
 a=h3_signature656(a_ci);q=h3_signature656(q_ci)
 return (a+q).contiguous()
def ranking656(scores):return sorted(range(len(scores)),key=lambda j:(-scores[j],j))
QIDS656={}
for z in PANEL656:QIDS656[z['case']]=add656(z['q'])
# The same fixed panel, with each case tested for original A and counterfactual A'.
for z in PANEL656:
 i=z['case'];bs=z['bs'];regions=z['regions'];q=z['q'];qci=QIDS656[i]
 try:
  qh=h3_signature656(qci);bh=[h3_signature656(j) for j in bs];bv=[kv_signature656(j) for j in bs]
  h3=native_scores656(qh,bh)
  print('CASE',i,'H3_RANK',ranking656(h3),'H3_SCORES',[round(x,5) for x in h3],flush=True)
  for tag,a_ci,correct in (('A',z['a'],0),('CF',z['acf'],1)):
   qv=projected_query656(a_ci,qci)
   if qv.numel()!=bv[0].numel():raise RuntimeError(f'KV_PROJECT_DIM {qv.numel()} != {bv[0].numel()}')
   kv=native_scores656(qv,bv)
   ah=conditioned_h3656(a_ci,qci);cond=native_scores656(ah,bh)
   scoremap={'H3_COSINE':h3,'QUERY_KV':kv,'A_QUERY_H3':cond}
   selected={name:ranking656(sc)[0] for name,sc in scoremap.items()}
   random_j=random.Random(SEED656+i*17+(0 if tag=='A' else 1)).randrange(4)
   selected['RANDOM_ONE']=random_j
   ROUTES656.append(dict(case=i,source=tag,gold_B=correct,scores=scoremap,selected=selected,correct={k:(v==correct) for k,v in selected.items()}))
   print(' ',tag,'ROUTES',selected,'GOLD',correct,'KV_SCORES',[round(x,5) for x in kv],'AQH3_SCORES',[round(x,5) for x in cond],flush=True)
   arms={'ORACLE_ALL':[a_ci]+bs,'ORACLE_ONE':[a_ci,bs[correct]],'H3_COSINE':[a_ci,bs[selected['H3_COSINE']]],'QUERY_KV':[a_ci,bs[selected['QUERY_KV']]],'A_QUERY_H3':[a_ci,bs[selected['A_QUERY_H3']]],'RANDOM_ONE':[a_ci,bs[random_j]],'A_ONLY':[a_ci]}
   # Cache reuse for duplicated B selections: only one native decode per unique cartridge combination.
   memo={}
   for arm,keys in arms.items():
    key=tuple(keys)
    if key not in memo:
     try:memo[key]=(answer(validate656(native_memory656(keys),keys),q),None)
     except Exception as e:memo[key]=('',f'{type(e).__name__}: {e}')
    ans,err=memo[key]
    rec=dict(case=i,source=tag,arm=arm,selected_B=selected.get(arm),gold=regions[correct],answer=ans,error=err,**grade656(ans,regions[correct],regions))
    if err:ERRORS656.append(rec)
    RESULTS656.append(rec)
    print('   ',arm,'STRICT',int(rec['strict']),'REGION',int(rec['region']),repr(ans),err or '',flush=True)
 except Exception as e:
  ERRORS656.append(dict(case=i,error=f'{type(e).__name__}: {e}'))
  print('CASE_ERROR',i,traceback.format_exc(),flush=True)
 gc.collect();torch.cuda.empty_cache()
ARMS656=('ORACLE_ALL','ORACLE_ONE','H3_COSINE','QUERY_KV','A_QUERY_H3','RANDOM_ONE','A_ONLY')
SCORES656={arm:{metric:sum(bool(r[metric]) for r in RESULTS656 if r['arm']==arm) for metric in ('strict','region')}|{'n':sum(r['arm']==arm for r in RESULTS656)} for arm in ARMS656}
ROUTE_ACC656={arm:sum(int(r['correct'][arm]) for r in ROUTES656) for arm in ('H3_COSINE','QUERY_KV','A_QUERY_H3','RANDOM_ONE')}
report656=dict(test='656',base_blob=BLOB656,panel_sha256=panel_sha,model='Qwen/Qwen2.5-7B-Instruct',cut=CUT656,
 hypothesis='Distinguish simple H3 matching from A-conditioned numerical H3 and frozen-model V-projection addressing',
 routes=ROUTES656,scores=SCORES656,route_accuracy=ROUTE_ACC656,results=RESULTS656,errors=ERRORS656,
 weights_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,
 seconds=round(time.perf_counter()-T656,2),
 limitations='All three routers are external numerical diagnostics. QUERY_KV uses frozen V projection and stored V signatures; it is NOT true attention Q-K scoring. No response-perturbation arm, autonomous routing, disk persistence, superposition, or native end-to-end inference is demonstrated.')
out656=ROOT656/'TEST656_results.json';out656.write_text(json.dumps(report656,indent=2,ensure_ascii=False),encoding='utf-8')
print('='*116+'\nTEST656_FINAL\n'+json.dumps(SCORES656,indent=2))
print('ROUTING_ACCURACY',ROUTE_ACC656,'N',len(ROUTES656))
print('WEIGHTS_FROZEN',report656['weights_frozen'],'SENTINEL',report656['sentinel_unchanged'])
print('RESULT',out656,'SHA256',hashlib.sha256(out656.read_bytes()).hexdigest())
complete=len(RESULTS656)==N656*2*len(ARMS656) and len(ROUTES656)==N656*2 and not ERRORS656 and report656['weights_frozen'] and report656['sentinel_unchanged']
print('STATUS','EXPERIMENT_COMPLETE' if complete else 'ERRORS_PRESENT')
print('INTERPRETATION: Numerical routing comparisons only; oracle success does not prove a Transformer architectural limit.')
