# TEST654 — AKBASCORE MAM · NATIVE CUDA KV FABRIC · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='654';SEED654=6502026;N654=8;CUT654=3;MAX_NEW654=16
ROOT654=Path('/content/AKBASCORE_TEST654');ROOT654.mkdir(parents=True,exist_ok=True)
URL654='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB654='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND654='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob654(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate654(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST654 — NATIVE CUDA DYNAMIC KV FABRIC / SOURCE-LOCKED QWEN\n'+'='*116)
gate654('CUDA_AVAILABLE',torch.cuda.is_available())
if importlib.util.find_spec('ninja') is None:subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
nvcc654=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin'/'nvcc') if CUDA_HOME else None)
gate654('NINJA',importlib.util.find_spec('ninja') is not None)
gate654('NVCC',bool(nvcc654 and Path(nvcc654).is_file()))
gate654('CXX',bool(shutil.which('c++') or shutil.which('g++')))
os.environ.setdefault('MAX_JOBS','2')
# Compile native extension BEFORE downloading model weights.
CPP654=r'''
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
CUDA654=r'''
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
native654=load_inline(name='akbascore_kvfabric654_v1',cpp_sources=[CPP654],cuda_sources=[CUDA654],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
@torch.no_grad()
def splice654(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native654.append_kv(out,a,b);return out
for t0,t1 in ((29,7),(43,31),(1,1)):
 x=torch.randn((1,4,t0,128),device='cuda',dtype=torch.bfloat16);y=torch.randn((1,4,t1,128),device='cuda',dtype=torch.bfloat16)
 gate654('NATIVE_BITWISE_'+str(t0)+'_'+str(t1),torch.equal(splice654(x,y),torch.cat((x,y),dim=2)))
# Load exactly the same verified baseline as TEST650.
p=Path('/content/566.py')
if p.is_file() and blob654(p.read_bytes())==BLOB654:raw=p.read_bytes();origin='LOCAL_VERIFIED'
else:
 req=urllib.request.Request(URL654,headers={'User-Agent':'AKBASCORE-TEST654'})
 with urllib.request.urlopen(req,timeout=120) as f:raw=f.read()
 origin='GITHUB_VERIFIED'
gate654('TEST566_BLOB',blob654(raw)==BLOB654)
src=raw.decode('utf-8');gate654('TEST566_BOUNDARY',src.count(BOUND654)==1)
exec(compile(src.split(BOUND654,1)[0],'566.py','exec'),globals())
gate654('FROZEN',all(not p.requires_grad for p in model.parameters()))
gate654('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate654('PANEL_LOCK',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45')
gate654('SENTINEL',sentinel()==S0)
print('BASE',origin,'BLOB',blob654(raw),'GPU',torch.cuda.get_device_name(0),flush=True)
# Same panel generator and random seed as TEST650: identical A/A' and four competing B facts.
rng=random.Random(SEED654);alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';seen=set()
def word654():
 while True:
  s=''.join(rng.choice(alphabet) for _ in range(9)).capitalize()
  if s not in seen:seen.add(s);return s
PANEL654=[];CAR654=[]
def add654(s):
 ids=tok(source(s),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('PREFIX_MISMATCH')
 i=len(CAR654);CAR654.append(dict(fact=s,body=ids[P:]));return i
for i in range(N654):
 obj=word654();owners=[word654() for _ in range(4)];regions=[word654() for _ in range(4)]
 a=add654(f'The owner of artifact {obj} is {owners[0]}.')
 acf=add654(f'The owner of artifact {obj} is {owners[1]}.')
 bs=[add654(f'{owners[j]} lives in region {regions[j]}.') for j in range(4)]
 PANEL654.append(dict(case=i,object=obj,owners=owners,regions=regions,a=a,acf=acf,bs=bs,q=f'In which region does the owner of artifact {obj} live?',qa=f'Who owns artifact {obj}?'))
CAR=CAR654;MAX_NEW=MAX_NEW654
panel_sha=hashlib.sha256(json.dumps(PANEL654,sort_keys=True).encode()).hexdigest()
gate654('TEST650_PANEL_LOCK',panel_sha=='db8b211abed216793bf2e4010eb006c862a74202d7512ae53a8a463fe929d466')
@torch.no_grad()
def native_append654(memory,ci):
 # Preserve the locked TEST566 upper-layer computation and H3 injection exactly.
 # Only lower-layer K/V concatenation is replaced with native CUDA, after RoPE position alignment.
 cp=checkpoint(ci,CUT654);oldn=memory[0][0].shape[-2];q=cp['body']
 pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp['h'][:,P:,:]
 if body.shape!=(1,q,H):raise RuntimeError('H3_BODY_SHAPE')
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError('H3_INJECTION_SHAPE')
  return replace(out,body)
 hook=model.model.layers[CUT654].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  for li in range(CUT654+1):
   ok,ov=memory[li];k,v=cp['kv'][li]
   k=rephase(k[:,:,P:,:],cp['pos'][P:],pos);v=v[:,:,P:,:]
   grown[li]=(splice654(ok,k),splice654(ov,v))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('PREFIX_MUTATED')
  if len(result)!=NL or any(k.shape!=(1,KVH,oldn+q,HD) or v.shape!=(1,KVH,oldn+q,HD) for k,v in result):raise RuntimeError('CACHE_SHAPE')
  return result
 finally:hook.remove()
@torch.no_grad()
def native_memory654(keys):
 mem=init(keys[0],CUT654)
 for ci in keys[1:]:mem=native_append654(mem,ci)
 return mem
@torch.no_grad()
def baseline_memory654(keys):
 mem=init(keys[0],CUT654)
 for ci in keys[1:]:mem=append(mem,ci,CUT654)
 return mem
@torch.no_grad()
def independent654(keys):
 mem=None
 for ci in keys:
  cp=checkpoint(ci,CUT654);kv=cp['kv']
  if mem is None:mem=kv;continue
  oldn=mem[0][0].shape[-2];q=len(CAR[ci]['body'])
  oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
  mem=tuple((splice654(ok,rephase(k[:,:,P:,:],oldpos,newpos)),splice654(ov,v[:,:,P:,:])) for (ok,ov),(k,v) in zip(mem,kv))
 return mem
@torch.no_grad()
def joint654(keys):
 ids=list(PIDS)
 for ci in keys:ids.extend(CAR[ci]['body'])
 n=len(ids);pos=torch.arange(n,device=DEVICE)
 out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 return clone(out.past_key_values)
def validate654(mem,keys):
 n=P+sum(len(CAR[i]['body']) for i in keys)
 if len(mem)!=NL or any(k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD) or not torch.isfinite(k).all() or not torch.isfinite(v).all() for k,v in mem):raise RuntimeError('INVALID_CACHE')
 return mem
def exact654(a,b):return re.sub(r'[^a-z]','',a.casefold())==re.sub(r'[^a-z]','',b.casefold())

# TEST654: fixed TEST653 panel, dual scoring and permutation-controlled causal deletion.
# Strict metric is unchanged; REGION_EXTRACT recognizes exactly one standalone known region.
# Two orders per case: original and deterministic nontrivial rotation (critical B indices move).
ARMS654=('B_ONLY','NATIVE_AB','NATIVE_CF','BASE_DC_AB','BASE_DC_CF','JOINT_AB','JOINT_CF')
RAW654=[];ABL654=[];CAUSAL654=[];T654=time.perf_counter()
def extract_region654(response,regions):
 hits=[name for name in regions if re.search(r'(?<![A-Za-z])'+re.escape(name)+r'(?![A-Za-z])',response,re.IGNORECASE)]
 return hits[0] if len(hits)==1 else None
def score654(ans,gold,regions,error=None):
 ext=None if error else extract_region654(ans,regions)
 return dict(strict=not error and exact654(ans,gold),region=not error and ext is not None and ext.casefold()==gold.casefold(),extracted=ext,ambiguous=not error and ext is None)
@torch.no_grad()
def run654(keys,q):return answer(validate654(native_memory654(keys),keys),q)
for z in PANEL654:
 i=z['case'];a=z['a'];acf=z['acf'];bs=z['bs'];q=z['q'];regions=z['regions']
 # All cases retain the original order; rotated order moves B0/B1 away from fixed slots.
 shift=(i%3)+1
 orders=(('ORIGINAL',(0,1,2,3)),('ROTATED',tuple((j+shift)%4 for j in range(4))))
 for order_name,order in orders:
  ordered_bs=[bs[j] for j in order];print('CASE',i,'ORDER',order_name,'B_ORIGINAL_INDICES',list(order),flush=True)
  try:
   cases={'B_ONLY':(ordered_bs,'native'),'NATIVE_AB':([a]+ordered_bs,'native'),'NATIVE_CF':([acf]+ordered_bs,'native'),
    'BASE_DC_AB':([a]+ordered_bs,'base'),'BASE_DC_CF':([acf]+ordered_bs,'base'),
    'JOINT_AB':([a]+ordered_bs,'joint'),'JOINT_CF':([acf]+ordered_bs,'joint')}
   for arm in ARMS654:
    keys,engine=cases[arm];gold=regions[1] if arm.endswith('_CF') else regions[0]
    try:
     mem=baseline_memory654(keys) if engine=='base' else joint654(keys) if engine=='joint' else native_memory654(keys)
     ans=answer(validate654(mem,keys),q);err=None
    except Exception as e:ans='';err=f'{type(e).__name__}: {e}'
    rec=dict(case=i,order=order_name,arm=arm,gold=gold,answer=ans,error=err,**score654(ans,gold,regions,err))
    RAW654.append(rec)
    print(' ',arm,'STRICT',int(rec['strict']),'REGION',int(rec['region']),repr(ans),err or '',flush=True)
   for tag,source,gold,critical in (('A',a,regions[0],0),('CF',acf,regions[1],1)):
    for j in range(4):
     keys=[source]+[bs[k] for k in order if k!=j]
     try:ans=run654(keys,q);err=None
     except Exception as e:ans='';err=f'{type(e).__name__}: {e}'
     metrics=score654(ans,gold,regions,err)
     rec=dict(case=i,order=order_name,source=tag,removed_original=j,removed_position=order.index(j),critical_original=critical,
      gold=gold,answer=ans,error=err,**metrics)
     ABL654.append(rec)
     print(' ',tag,'DROP_ORIGINAL_B',j,'AT_SLOT',order.index(j),'STRICT_RETAIN',int(rec['strict']),'REGION_RETAIN',int(rec['region']),repr(ans),err or '',flush=True)
   rr={r['arm']:r for r in RAW654 if r['case']==i and r['order']==order_name}
   aa={(r['source'],r['removed_original']):r for r in ABL654 if r['case']==i and r['order']==order_name}
   valid=len(aa)==8 and all(r['error'] is None for r in aa.values()) and len(rr)==7 and all(r['error'] is None for r in rr.values())
   parity=valid and all(rr[n]['answer']==rr[b]['answer'] for n,b in (('NATIVE_AB','BASE_DC_AB'),('NATIVE_CF','BASE_DC_CF')))
   for metric in ('strict','region'):
    full=valid and all(rr[k][metric] for k in ('NATIVE_AB','NATIVE_CF','BASE_DC_AB','BASE_DC_CF','JOINT_AB','JOINT_CF')) and not rr['B_ONLY'][metric]
    a_critical=valid and not aa[('A',0)][metric]
    cf_critical=valid and not aa[('CF',1)][metric]
    a_others=valid and all(aa[('A',j)][metric] for j in (1,2,3))
    cf_others=valid and all(aa[('CF',j)][metric] for j in (0,2,3))
    passed=bool(full and parity and a_critical and cf_critical and a_others and cf_others)
    CAUSAL654.append(dict(case=i,order=order_name,metric=metric,valid=valid,full=full,parity=parity,
     a_critical=a_critical,cf_critical=cf_critical,a_noncritical=a_others,cf_noncritical=cf_others,passed=passed))
    print(' CAUSAL',metric,'PASS' if passed else 'NOT_DEMONSTRATED','FULL',full,'PARITY',parity,'A_CRITICAL',a_critical,'CF_CRITICAL',cf_critical,'OTHERS',a_others,cf_others,flush=True)
  except Exception:print('CASE_FATAL',i,order_name,traceback.format_exc(),flush=True)
  gc.collect();torch.cuda.empty_cache()
SCORES654={arm:{m:sum(bool(r[m]) for r in RAW654 if r['arm']==arm) for m in ('strict','region')}|{'total':N654*2,'errors':sum(bool(r['error']) for r in RAW654 if r['arm']==arm)} for arm in ARMS654}
CAUSAL_SUMMARY654={m:{o:sum(r['passed'] for r in CAUSAL654 if r['metric']==m and r['order']==o) for o in ('ORIGINAL','ROTATED')} for m in ('strict','region')}
POSITION_INVARIANT654={m:sum(all(any(r['passed'] for r in CAUSAL654 if r['case']==i and r['order']==o and r['metric']==m) for o in ('ORIGINAL','ROTATED')) for i in range(N654)) for m in ('strict','region')}
report=dict(test=TEST,base_blob=BLOB654,panel_sha256=panel_sha,cut=CUT654,model='Qwen/Qwen2.5-7B-Instruct',
 scoring='strict unchanged; region requires exactly one standalone region from all four candidate regions',
 native_scope='CUDA BF16 KV append; frozen PyTorch H3 continuation and model inference',
 scores=SCORES654,causal_summary=CAUSAL_SUMMARY654,position_invariant=POSITION_INVARIANT654,
 raw=RAW654,ablation=ABL654,causal=CAUSAL654,
 model_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,
 seconds=round(time.perf_counter()-T654,2))
path=ROOT654/'TEST654_results.json';path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('='*116+'\nTEST654_FINAL\n'+json.dumps(SCORES654,indent=2))
print('CAUSAL_BY_ORDER',json.dumps(CAUSAL_SUMMARY654))
print('POSITION_INVARIANT',json.dumps(POSITION_INVARIANT654),'/',N654)
print('WEIGHTS_FROZEN',report['model_frozen'],'SENTINEL',report['sentinel_unchanged'])
print('RESULT',path,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
expected=N654*2
ok=report['model_frozen'] and report['sentinel_unchanged'] and len(RAW654)==expected*len(ARMS654) and len(ABL654)==expected*8 and len(CAUSAL654)==expected*2 and not any(r['error'] for r in RAW654+ABL654)
print('STATUS','EXPERIMENT_COMPLETE' if ok else 'ERRORS_PRESENT')
print('LIMITATION: Position permutations and candidate ablations test causal necessity, not autonomous retrieval or full C++ inference.')
