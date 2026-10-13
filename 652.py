# TEST652 — AKBASCORE MAM · NATIVE CUDA KV FABRIC · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='652';SEED652=6502026;N652=8;CUT652=3;MAX_NEW652=16
ROOT652=Path('/content/AKBASCORE_TEST652');ROOT652.mkdir(parents=True,exist_ok=True)
URL652='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB652='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND652='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob652(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate652(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST652 — NATIVE CUDA DYNAMIC KV FABRIC / SOURCE-LOCKED QWEN\n'+'='*116)
gate652('CUDA_AVAILABLE',torch.cuda.is_available())
if importlib.util.find_spec('ninja') is None:subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
nvcc652=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin'/'nvcc') if CUDA_HOME else None)
gate652('NINJA',importlib.util.find_spec('ninja') is not None)
gate652('NVCC',bool(nvcc652 and Path(nvcc652).is_file()))
gate652('CXX',bool(shutil.which('c++') or shutil.which('g++')))
os.environ.setdefault('MAX_JOBS','2')
# Compile native extension BEFORE downloading model weights.
CPP652=r'''
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
CUDA652=r'''
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
native652=load_inline(name='akbascore_kvfabric652_v1',cpp_sources=[CPP652],cuda_sources=[CUDA652],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
@torch.no_grad()
def splice652(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native652.append_kv(out,a,b);return out
for t0,t1 in ((29,7),(43,31),(1,1)):
 x=torch.randn((1,4,t0,128),device='cuda',dtype=torch.bfloat16);y=torch.randn((1,4,t1,128),device='cuda',dtype=torch.bfloat16)
 gate652('NATIVE_BITWISE_'+str(t0)+'_'+str(t1),torch.equal(splice652(x,y),torch.cat((x,y),dim=2)))
# Load exactly the same verified baseline as TEST650.
p=Path('/content/566.py')
if p.is_file() and blob652(p.read_bytes())==BLOB652:raw=p.read_bytes();origin='LOCAL_VERIFIED'
else:
 req=urllib.request.Request(URL652,headers={'User-Agent':'AKBASCORE-TEST652'})
 with urllib.request.urlopen(req,timeout=120) as f:raw=f.read()
 origin='GITHUB_VERIFIED'
gate652('TEST566_BLOB',blob652(raw)==BLOB652)
src=raw.decode('utf-8');gate652('TEST566_BOUNDARY',src.count(BOUND652)==1)
exec(compile(src.split(BOUND652,1)[0],'566.py','exec'),globals())
gate652('FROZEN',all(not p.requires_grad for p in model.parameters()))
gate652('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate652('PANEL_LOCK',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45')
gate652('SENTINEL',sentinel()==S0)
print('BASE',origin,'BLOB',blob652(raw),'GPU',torch.cuda.get_device_name(0),flush=True)
# Same panel generator and random seed as TEST650: identical A/A' and four competing B facts.
rng=random.Random(SEED652);alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';seen=set()
def word652():
 while True:
  s=''.join(rng.choice(alphabet) for _ in range(9)).capitalize()
  if s not in seen:seen.add(s);return s
PANEL652=[];CAR652=[]
def add652(s):
 ids=tok(source(s),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('PREFIX_MISMATCH')
 i=len(CAR652);CAR652.append(dict(fact=s,body=ids[P:]));return i
for i in range(N652):
 obj=word652();owners=[word652() for _ in range(4)];regions=[word652() for _ in range(4)]
 a=add652(f'The owner of artifact {obj} is {owners[0]}.')
 acf=add652(f'The owner of artifact {obj} is {owners[1]}.')
 bs=[add652(f'{owners[j]} lives in region {regions[j]}.') for j in range(4)]
 PANEL652.append(dict(case=i,object=obj,owners=owners,regions=regions,a=a,acf=acf,bs=bs,q=f'In which region does the owner of artifact {obj} live?',qa=f'Who owns artifact {obj}?'))
CAR=CAR652;MAX_NEW=MAX_NEW652
panel_sha=hashlib.sha256(json.dumps(PANEL652,sort_keys=True).encode()).hexdigest()
gate652('TEST650_PANEL_LOCK',panel_sha=='db8b211abed216793bf2e4010eb006c862a74202d7512ae53a8a463fe929d466')
@torch.no_grad()
def native_append652(memory,ci):
 # Preserve the locked TEST566 upper-layer computation and H3 injection exactly.
 # Only lower-layer K/V concatenation is replaced with native CUDA, after RoPE position alignment.
 cp=checkpoint(ci,CUT652);oldn=memory[0][0].shape[-2];q=cp['body']
 pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp['h'][:,P:,:]
 if body.shape!=(1,q,H):raise RuntimeError('H3_BODY_SHAPE')
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError('H3_INJECTION_SHAPE')
  return replace(out,body)
 hook=model.model.layers[CUT652].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  for li in range(CUT652+1):
   ok,ov=memory[li];k,v=cp['kv'][li]
   k=rephase(k[:,:,P:,:],cp['pos'][P:],pos);v=v[:,:,P:,:]
   grown[li]=(splice652(ok,k),splice652(ov,v))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('PREFIX_MUTATED')
  if len(result)!=NL or any(k.shape!=(1,KVH,oldn+q,HD) or v.shape!=(1,KVH,oldn+q,HD) for k,v in result):raise RuntimeError('CACHE_SHAPE')
  return result
 finally:hook.remove()
@torch.no_grad()
def native_memory652(keys):
 mem=init(keys[0],CUT652)
 for ci in keys[1:]:mem=native_append652(mem,ci)
 return mem
@torch.no_grad()
def baseline_memory652(keys):
 mem=init(keys[0],CUT652)
 for ci in keys[1:]:mem=append(mem,ci,CUT652)
 return mem
@torch.no_grad()
def independent652(keys):
 mem=None
 for ci in keys:
  cp=checkpoint(ci,CUT652);kv=cp['kv']
  if mem is None:mem=kv;continue
  oldn=mem[0][0].shape[-2];q=len(CAR[ci]['body'])
  oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
  mem=tuple((splice652(ok,rephase(k[:,:,P:,:],oldpos,newpos)),splice652(ov,v[:,:,P:,:])) for (ok,ov),(k,v) in zip(mem,kv))
 return mem
@torch.no_grad()
def joint652(keys):
 ids=list(PIDS)
 for ci in keys:ids.extend(CAR[ci]['body'])
 n=len(ids);pos=torch.arange(n,device=DEVICE)
 out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 return clone(out.past_key_values)
def validate652(mem,keys):
 n=P+sum(len(CAR[i]['body']) for i in keys)
 if len(mem)!=NL or any(k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD) or not torch.isfinite(k).all() or not torch.isfinite(v).all() for k,v in mem):raise RuntimeError('INVALID_CACHE')
 return mem
def exact652(a,b):return re.sub(r'[^a-z]','',a.casefold())==re.sub(r'[^a-z]','',b.casefold())
# Latent readout: frozen H3 query-state from A, frozen H3 body-token states from each B.
# Native CUDA ranks four B cartridges by mean/max cosine. No decoded owner, gold label or source text is used in routing.
@torch.no_grad()
def probe652(mem,question):
 state={}
 def capture(module,args,out):
  h=out[0] if isinstance(out,(tuple,list)) else out
  state['h']=h.detach().clone()
 hook=model.model.layers[CUT652].register_forward_hook(capture)
 try:
  n=mem[0][0].shape[-2];ids=tok(suffix(question),add_special_tokens=False).input_ids
  pos=torch.arange(n,n+len(ids),device=DEVICE)
  model(input_ids=torch.tensor([ids],device=DEVICE),past_key_values=cache_build(mem),attention_mask=torch.ones((1,n+len(ids)),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 finally:hook.remove()
 if 'h' not in state:raise RuntimeError('PROBE_MISSING')
 return state['h'][0].float().contiguous()
@torch.no_grad()
def body652(ci):
 cp=checkpoint(ci,CUT652);h=cp['h'][0,P:,:].float().contiguous()
 if h.numel()==0 or not torch.isfinite(h).all():raise RuntimeError('INVALID_BODY')
 return h
@torch.no_grad()
def cosine652(q,c):
 q=q.float().contiguous();c=c.float().contiguous();out=torch.empty((c.shape[0],),device=DEVICE,dtype=torch.float32)
 native652.cosine_scores(out,q,c)
 return out
@torch.no_grad()
def select652(qstates,bodies,mode):
 # Four candidate cartridges, each scored from frozen model tensors only.
 # MEAN and MAX are prespecified competing readouts; no ground-truth-driven selection.
 scores=[]
 for b in bodies:
  v=cosine652(qstates[-1],b)
  scores.append(v.mean() if mode=='MEAN' else v.max())
 score=torch.stack(scores);return int(score.argmax().item()),[round(float(x),7) for x in score]
# Native numerical parity for readout, independently of model outcome.
qtest=torch.randn((H,),device=DEVICE);ctest=torch.randn((7,H),device=DEVICE)
ref=torch.nn.functional.cosine_similarity(ctest,qtest.unsqueeze(0),dim=1)
gate652('NATIVE_COSINE_PARITY',bool(torch.allclose(cosine652(qtest,ctest),ref,atol=1e-5,rtol=1e-5)))
ARMS652=('B_ONLY','DC_ALL','JOINT_ALL','ROUTE_MEAN','ROUTE_MAX','ROUTE_MEAN_CF','ROUTE_MAX_CF')
RAW652=[];ROUTES652=[];T652=time.perf_counter()
for z in PANEL652:
 print('CASE',z['case'],'LATENT ROUTING',flush=True)
 try:
  a,acf,bs=z['a'],z['acf'],z['bs'];mb=validate652(native_memory652(bs),bs)
  ba=validate652(native_memory652([a]),[a]);bcf=validate652(native_memory652([acf]),[acf])
  qa=probe652(ba,z['qa']);qcf=probe652(bcf,z['qa']);bodies=[body652(i) for i in bs]
  selections={}
  for tag,query in (('A',qa),('CF',qcf)):
   for mode in ('MEAN','MAX'):
    idx,score=select652(query,bodies,mode);selections[(tag,mode)]=idx
    ROUTES652.append(dict(case=z['case'],source=tag,mode=mode,selected=idx,expected=0 if tag=='A' else 1,correct=idx==(0 if tag=='A' else 1),scores=score))
    print(' ROUTE',tag,mode,'INDEX',idx,'SCORES',score,flush=True)
  arms={
   'B_ONLY':lambda:answer(mb,z['q']),
   'DC_ALL':lambda:answer(validate652(native_memory652([a]+bs),[a]+bs),z['q']),
   'JOINT_ALL':lambda:answer(validate652(joint652([a]+bs),[a]+bs),z['q']),
   'ROUTE_MEAN':lambda:answer(validate652(native_memory652([a,bs[selections[('A','MEAN')]]]),[a,bs[selections[('A','MEAN')]]]),z['q']),
   'ROUTE_MAX':lambda:answer(validate652(native_memory652([a,bs[selections[('A','MAX')]]]),[a,bs[selections[('A','MAX')]]]),z['q']),
   'ROUTE_MEAN_CF':lambda:answer(validate652(native_memory652([acf,bs[selections[('CF','MEAN')]]]),[acf,bs[selections[('CF','MEAN')]]]),z['q']),
   'ROUTE_MAX_CF':lambda:answer(validate652(native_memory652([acf,bs[selections[('CF','MAX')]]]),[acf,bs[selections[('CF','MAX')]]]),z['q'])}
  for arm in ARMS652:
   gold=z['regions'][1] if arm.endswith('_CF') else z['regions'][0]
   try:ans=arms[arm]();err=None
   except Exception as e:ans='';err=f'{type(e).__name__}: {e}'
   rec=dict(case=z['case'],arm=arm,gold=gold,answer=ans,exact=err is None and exact652(ans,gold),error=err)
   RAW652.append(rec);print(' ',arm,'PASS' if rec['exact'] else 'FAIL',repr(ans),err or '',flush=True)
 except Exception as e:
  print('CASE_FATAL',z['case'],traceback.format_exc(),flush=True)
  for arm in ARMS652:
   if not any(r['case']==z['case'] and r['arm']==arm for r in RAW652):RAW652.append(dict(case=z['case'],arm=arm,gold='',answer='',exact=False,error=f'CASE_FATAL: {e}'))
 gc.collect();torch.cuda.empty_cache()
SCORES652={arm:dict(correct=sum(r['exact'] for r in RAW652 if r['arm']==arm),total=N652,errors=sum(bool(r['error']) for r in RAW652 if r['arm']==arm)) for arm in ARMS652}
ROUTE_SCORES652={f'{tag}_{mode}':sum(r['correct'] for r in ROUTES652 if r['source']==tag and r['mode']==mode) for tag in ('A','CF') for mode in ('MEAN','MAX')}
CAUSAL652=[]
for z in PANEL652:
 r={v['arm']:v for v in RAW652 if v['case']==z['case']}
 s={(v['source'],v['mode']):v for v in ROUTES652 if v['case']==z['case']}
 for mode in ('MEAN','MAX'):
  valid=all(k in s for k in (('A',mode),('CF',mode)))
  flip=valid and s[('A',mode)]['selected']==0 and s[('CF',mode)]['selected']==1
  good=r['ROUTE_'+mode]['exact'] and r['ROUTE_'+mode+'_CF']['exact']
  causal=bool(flip and good and r['DC_ALL']['exact'] and r['JOINT_ALL']['exact'] and not r['B_ONLY']['exact'])
  CAUSAL652.append(dict(case=z['case'],mode=mode,route_flip=bool(flip),answers_correct=bool(good),causal_pass=causal))
  print('CAUSAL',z['case'],mode,'PASS' if causal else 'NOT_DEMONSTRATED',flush=True)
report=dict(test=TEST,base_blob=BLOB652,panel_sha256=panel_sha,cut=CUT652,model='Qwen/Qwen2.5-7B-Instruct',native_scope='CUDA KV append and cosine ranking; Qwen H3 capture, RoPE and forward via locked PyTorch',scores=SCORES652,route_scores=ROUTE_SCORES652,routes=ROUTES652,causal=CAUSAL652,raw=RAW652,model_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,seconds=round(time.perf_counter()-T652,2))
path=ROOT652/'TEST652_results.json';path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('='*116+'\nTEST652_FINAL\n'+json.dumps(SCORES652,indent=2))
print('LATENT_ROUTING',ROUTE_SCORES652)
print('CAUSAL_LATENT_ROUTING',sum(x['causal_pass'] for x in CAUSAL652),'/',len(CAUSAL652))
print('WEIGHTS_FROZEN',report['model_frozen'],'SENTINEL',report['sentinel_unchanged'])
print('RESULT',path,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
print('STATUS','EXPERIMENT_COMPLETE' if report['sentinel_unchanged'] and not any(r['error'] for r in RAW652) and len(ROUTES652)==4*N652 else 'ERRORS_PRESENT')
print('LIMITATION: Frozen H3 cosine is a candidate readout, not a proven semantic router. A remains in final DC3 memory; routing independence is not established.')
