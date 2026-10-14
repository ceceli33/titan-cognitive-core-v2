# TEST655 — AKBASCORE MAM · ROUTING BOTTLENECK · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='655';SEED655=6502026;N655=8;CUT655=3;MAX_NEW655=16
ROOT655=Path('/content/AKBASCORE_TEST655');ROOT655.mkdir(parents=True,exist_ok=True)
URL655='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB655='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND655='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob655(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate655(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST655 — NATIVE CUDA DYNAMIC KV FABRIC / SOURCE-LOCKED QWEN\n'+'='*116)
gate655('CUDA_AVAILABLE',torch.cuda.is_available())
if importlib.util.find_spec('ninja') is None:subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
nvcc655=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin'/'nvcc') if CUDA_HOME else None)
gate655('NINJA',importlib.util.find_spec('ninja') is not None)
gate655('NVCC',bool(nvcc655 and Path(nvcc655).is_file()))
gate655('CXX',bool(shutil.which('c++') or shutil.which('g++')))
os.environ.setdefault('MAX_JOBS','2')
# Compile native extension BEFORE downloading model weights.
CPP655=r'''
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
CUDA655=r'''
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
native655=load_inline(name='akbascore_kvfabric655_v1',cpp_sources=[CPP655],cuda_sources=[CUDA655],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
@torch.no_grad()
def splice655(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native655.append_kv(out,a,b);return out
for t0,t1 in ((29,7),(43,31),(1,1)):
 x=torch.randn((1,4,t0,128),device='cuda',dtype=torch.bfloat16);y=torch.randn((1,4,t1,128),device='cuda',dtype=torch.bfloat16)
 gate655('NATIVE_BITWISE_'+str(t0)+'_'+str(t1),torch.equal(splice655(x,y),torch.cat((x,y),dim=2)))
# Load exactly the same verified baseline as TEST650.
p=Path('/content/566.py')
if p.is_file() and blob655(p.read_bytes())==BLOB655:raw=p.read_bytes();origin='LOCAL_VERIFIED'
else:
 req=urllib.request.Request(URL655,headers={'User-Agent':'AKBASCORE-TEST655'})
 with urllib.request.urlopen(req,timeout=120) as f:raw=f.read()
 origin='GITHUB_VERIFIED'
gate655('TEST566_BLOB',blob655(raw)==BLOB655)
src=raw.decode('utf-8');gate655('TEST566_BOUNDARY',src.count(BOUND655)==1)
exec(compile(src.split(BOUND655,1)[0],'566.py','exec'),globals())
gate655('FROZEN',all(not p.requires_grad for p in model.parameters()))
gate655('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate655('PANEL_LOCK',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45')
gate655('SENTINEL',sentinel()==S0)
print('BASE',origin,'BLOB',blob655(raw),'GPU',torch.cuda.get_device_name(0),flush=True)
# Same panel generator and random seed as TEST650: identical A/A' and four competing B facts.
rng=random.Random(SEED655);alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';seen=set()
def word655():
 while True:
  s=''.join(rng.choice(alphabet) for _ in range(9)).capitalize()
  if s not in seen:seen.add(s);return s
PANEL655=[];CAR655=[]
def add655(s):
 ids=tok(source(s),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('PREFIX_MISMATCH')
 i=len(CAR655);CAR655.append(dict(fact=s,body=ids[P:]));return i
for i in range(N655):
 obj=word655();owners=[word655() for _ in range(4)];regions=[word655() for _ in range(4)]
 a=add655(f'The owner of artifact {obj} is {owners[0]}.')
 acf=add655(f'The owner of artifact {obj} is {owners[1]}.')
 bs=[add655(f'{owners[j]} lives in region {regions[j]}.') for j in range(4)]
 PANEL655.append(dict(case=i,object=obj,owners=owners,regions=regions,a=a,acf=acf,bs=bs,q=f'In which region does the owner of artifact {obj} live?',qa=f'Who owns artifact {obj}?'))
CAR=CAR655;MAX_NEW=MAX_NEW655
panel_sha=hashlib.sha256(json.dumps(PANEL655,sort_keys=True).encode()).hexdigest()
gate655('TEST650_PANEL_LOCK',panel_sha=='db8b211abed216793bf2e4010eb006c862a74202d7512ae53a8a463fe929d466')
@torch.no_grad()
def native_append655(memory,ci):
 # Preserve the locked TEST566 upper-layer computation and H3 injection exactly.
 # Only lower-layer K/V concatenation is replaced with native CUDA, after RoPE position alignment.
 cp=checkpoint(ci,CUT655);oldn=memory[0][0].shape[-2];q=cp['body']
 pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp['h'][:,P:,:]
 if body.shape!=(1,q,H):raise RuntimeError('H3_BODY_SHAPE')
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError('H3_INJECTION_SHAPE')
  return replace(out,body)
 hook=model.model.layers[CUT655].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  for li in range(CUT655+1):
   ok,ov=memory[li];k,v=cp['kv'][li]
   k=rephase(k[:,:,P:,:],cp['pos'][P:],pos);v=v[:,:,P:,:]
   grown[li]=(splice655(ok,k),splice655(ov,v))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('PREFIX_MUTATED')
  if len(result)!=NL or any(k.shape!=(1,KVH,oldn+q,HD) or v.shape!=(1,KVH,oldn+q,HD) for k,v in result):raise RuntimeError('CACHE_SHAPE')
  return result
 finally:hook.remove()
@torch.no_grad()
def native_memory655(keys):
 mem=init(keys[0],CUT655)
 for ci in keys[1:]:mem=native_append655(mem,ci)
 return mem
@torch.no_grad()
def baseline_memory655(keys):
 mem=init(keys[0],CUT655)
 for ci in keys[1:]:mem=append(mem,ci,CUT655)
 return mem
@torch.no_grad()
def independent655(keys):
 mem=None
 for ci in keys:
  cp=checkpoint(ci,CUT655);kv=cp['kv']
  if mem is None:mem=kv;continue
  oldn=mem[0][0].shape[-2];q=len(CAR[ci]['body'])
  oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
  mem=tuple((splice655(ok,rephase(k[:,:,P:,:],oldpos,newpos)),splice655(ov,v[:,:,P:,:])) for (ok,ov),(k,v) in zip(mem,kv))
 return mem
@torch.no_grad()
def joint655(keys):
 ids=list(PIDS)
 for ci in keys:ids.extend(CAR[ci]['body'])
 n=len(ids);pos=torch.arange(n,device=DEVICE)
 out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 return clone(out.past_key_values)
def validate655(mem,keys):
 n=P+sum(len(CAR[i]['body']) for i in keys)
 if len(mem)!=NL or any(k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD) or not torch.isfinite(k).all() or not torch.isfinite(v).all() for k,v in mem):raise RuntimeError('INVALID_CACHE')
 return mem
def exact655(a,b):return re.sub(r'[^a-z]','',a.casefold())==re.sub(r'[^a-z]','',b.casefold())


# TEST655 — ROUTING BOTTLENECK: same frozen model, locked panel and native CUT3.
# ORACLE_ALL = known relevant A + all four B; ORACLE_ONE = known relevant A + correct B.
# LATENT_ONE = A + B selected by native CUDA H3 cosine (no textual search).
# RANDOM_ONE = A + random B. Oracle arms are diagnostic controls, not autonomous retrieval.
# All ranking inputs are numeric checkpoint tensors. No answer/region strings enter ranking.
import itertools
T655=time.perf_counter();RESULTS655=[];ROUTES655=[];ERRORS655=[]
def extract655(ans,regions):
 hits=[v for v in regions if re.search(r'(?<![A-Za-z])'+re.escape(v)+r'(?![A-Za-z])',ans,re.IGNORECASE)]
 return hits[0] if len(hits)==1 else None
def grade655(ans,gold,regions):
 extracted=extract655(ans,regions)
 return dict(strict=exact655(ans,gold),region=extracted is not None and extracted.casefold()==gold.casefold(),extracted=extracted)
@torch.no_grad()
def h3_signature655(ci):
 cp=checkpoint(ci,CUT655);h=cp['h'][:,P:,:].float()
 if h.ndim!=3 or h.shape[-1]!=H or h.shape[1]==0:raise RuntimeError('H3_SIGNATURE_SHAPE')
 return h.mean(dim=1).squeeze(0).contiguous()
@torch.no_grad()
def native_scores655(query,candidates):
 out=torch.empty((len(candidates),),device=DEVICE,dtype=torch.float32)
 native655.cosine_scores(out,query.contiguous(),torch.stack(candidates).contiguous())
 return out.cpu().tolist()
# Each query is forged only as a numeric H3 diagnostic; never appended as a fact cartridge.
QIDS655={}
for z in PANEL655:
 QIDS655[z['case']]=add655(z['q'])
for z in PANEL655:
 i=z['case'];bs=z['bs'];regions=z['regions'];q=z['q'];qci=QIDS655[i]
 try:
  qvec=h3_signature655(qci);bvec=[h3_signature655(j) for j in bs]
  scores=native_scores655(qvec,bvec);rank=sorted(range(4),key=lambda j:(-scores[j],j));chosen=rank[0]
  ROUTES655.append(dict(case=i,scores=scores,ranking=rank,selected=chosen,selected_A_correct=chosen==0,selected_CF_correct=chosen==1))
  print('CASE',i,'H3_RANK',rank,'SCORES',[round(x,5) for x in scores],flush=True)
  for tag,source,correct in (('A',z['a'],0),('CF',z['acf'],1)):
   random_j=random.Random(SEED655+i*17+(0 if tag=='A' else 1)).randrange(4)
   arms={'ORACLE_ALL':[source]+bs,'ORACLE_ONE':[source,bs[correct]],'LATENT_ONE':[source,bs[chosen]],'RANDOM_ONE':[source,bs[random_j]],'A_ONLY':[source]}
   for arm,keys in arms.items():
    try:
     ans=answer(validate655(native_memory655(keys),keys),q)
     rec=dict(case=i,source=tag,arm=arm,selected_B=chosen if arm=='LATENT_ONE' else random_j if arm=='RANDOM_ONE' else correct if arm=='ORACLE_ONE' else None,gold=regions[correct],answer=ans,error=None,**grade655(ans,regions[correct],regions))
    except Exception as e:
     rec=dict(case=i,source=tag,arm=arm,selected_B=None,gold=regions[correct],answer='',error=f'{type(e).__name__}: {e}',strict=False,region=False,extracted=None)
     ERRORS655.append(rec)
    RESULTS655.append(rec)
    print(' ',tag,arm,'STRICT',int(rec['strict']),'REGION',int(rec['region']),repr(rec['answer']),rec['error'] or '',flush=True)
 except Exception as e:
  ERRORS655.append(dict(case=i,error=f'{type(e).__name__}: {e}'))
  print('CASE_ERROR',i,traceback.format_exc(),flush=True)
 gc.collect();torch.cuda.empty_cache()
ARMS655=('ORACLE_ALL','ORACLE_ONE','LATENT_ONE','RANDOM_ONE','A_ONLY')
SCORES655={arm:{metric:sum(bool(r[metric]) for r in RESULTS655 if r['arm']==arm) for metric in ('strict','region')}|{'n':sum(r['arm']==arm for r in RESULTS655)} for arm in ARMS655}
route_correct_A=sum(r['selected_A_correct'] for r in ROUTES655);route_correct_CF=sum(r['selected_CF_correct'] for r in ROUTES655)
report655=dict(test='655',base_blob=BLOB655,panel_sha256=panel_sha,model='Qwen/Qwen2.5-7B-Instruct',cut=CUT655,
 hypothesis='Distinguish KV integration from numeric H3 memory selection under fixed model and facts',
 scoring='strict exact or unique region extraction',routes=ROUTES655,scores=SCORES655,results=RESULTS655,errors=ERRORS655,
 route_correct_A=route_correct_A,route_correct_CF=route_correct_CF,
 weights_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,
 seconds=round(time.perf_counter()-T655,2),
 limitations='H3 cosine is an external numerical readout, not learned/internal autonomous routing. Oracle uses ground-truth selection. No disk persistence, tensor superposition or native end-to-end inference is demonstrated.')
out655=ROOT655/'TEST655_results.json';out655.write_text(json.dumps(report655,indent=2,ensure_ascii=False),encoding='utf-8')
print('='*116+'\nTEST655_FINAL\n'+json.dumps(SCORES655,indent=2))
print('NUMERIC_ROUTING_ACCURACY','A',route_correct_A,'/',N655,'CF',route_correct_CF,'/',N655)
print('WEIGHTS_FROZEN',report655['weights_frozen'],'SENTINEL',report655['sentinel_unchanged'])
print('RESULT',out655,'SHA256',hashlib.sha256(out655.read_bytes()).hexdigest())
complete=len(RESULTS655)==N655*2*len(ARMS655) and len(ROUTES655)==N655 and not ERRORS655 and report655['weights_frozen'] and report655['sentinel_unchanged']
print('STATUS','EXPERIMENT_COMPLETE' if complete else 'ERRORS_PRESENT')
print('INTERPRETATION: Oracle success with latent failure isolates selection under this readout; it does NOT prove a fundamental Transformer limit.')
