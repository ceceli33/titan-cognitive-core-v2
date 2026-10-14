# TEST658 — AKBASCORE MAM · NUMERIC ASSOCIATIVE ADDRESSING · LOCKED TEST566
# Single Colab cell; frozen Qwen, CUT3, original TEST566 checkpoint/init/append/answer untouched.
# Native CUDA copies aligned K/V bodies into persistent cache; upper layers use verified DC3 continuation.
# Demonstrates KV connectivity/parity, NOT autonomous memory search or full C++ inference.
import os,sys,re,json,time,random,hashlib,urllib.request,traceback,gc,shutil,subprocess,importlib.util
from pathlib import Path
import torch
TEST='658';SEED658=6502026;N658=8;CUT658=3;MAX_NEW658=16
ROOT658=Path('/content/AKBASCORE_TEST658');ROOT658.mkdir(parents=True,exist_ok=True)
URL658='https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/566.py'
BLOB658='b117c4b37dc50e6e52cc0b1635b0f6eb04dfbb71'
BOUND658='# TEST566 ADDITIONS — original TEST564 checkpoint/init/append/answer remain unchanged.'
def blob658(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def gate658(n,v):print(('PASS' if v else 'FAIL'),n,flush=True);assert v,n
print('='*116+'\nTEST658 — NATIVE CUDA DYNAMIC KV FABRIC / SOURCE-LOCKED QWEN\n'+'='*116)
gate658('CUDA_AVAILABLE',torch.cuda.is_available())
if importlib.util.find_spec('ninja') is None:subprocess.check_call([sys.executable,'-m','pip','install','-q','ninja'])
from torch.utils.cpp_extension import load_inline,CUDA_HOME
nvcc658=shutil.which('nvcc') or (str(Path(CUDA_HOME)/'bin'/'nvcc') if CUDA_HOME else None)
gate658('NINJA',importlib.util.find_spec('ninja') is not None)
gate658('NVCC',bool(nvcc658 and Path(nvcc658).is_file()))
gate658('CXX',bool(shutil.which('c++') or shutil.which('g++')))
os.environ.setdefault('MAX_JOBS','2')
# Compile native extension BEFORE downloading model weights.
CPP658=r'''
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
CUDA658=r'''
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
native658=load_inline(name='akbascore_kvfabric658_v1',cpp_sources=[CPP658],cuda_sources=[CUDA658],functions=None,extra_cflags=['-O2'],extra_cuda_cflags=['-O2'],with_cuda=True,verbose=True)
@torch.no_grad()
def splice658(a,b):
 a=a.contiguous();b=b.contiguous();out=torch.empty((a.shape[0],a.shape[1],a.shape[2]+b.shape[2],a.shape[3]),device=a.device,dtype=a.dtype)
 native658.append_kv(out,a,b);return out
for t0,t1 in ((29,7),(43,31),(1,1)):
 x=torch.randn((1,4,t0,128),device='cuda',dtype=torch.bfloat16);y=torch.randn((1,4,t1,128),device='cuda',dtype=torch.bfloat16)
 gate658('NATIVE_BITWISE_'+str(t0)+'_'+str(t1),torch.equal(splice658(x,y),torch.cat((x,y),dim=2)))
# Load exactly the same verified baseline as TEST650.
p=Path('/content/566.py')
if p.is_file() and blob658(p.read_bytes())==BLOB658:raw=p.read_bytes();origin='LOCAL_VERIFIED'
else:
 req=urllib.request.Request(URL658,headers={'User-Agent':'AKBASCORE-TEST658'})
 with urllib.request.urlopen(req,timeout=120) as f:raw=f.read()
 origin='GITHUB_VERIFIED'
gate658('TEST566_BLOB',blob658(raw)==BLOB658)
src=raw.decode('utf-8');gate658('TEST566_BOUNDARY',src.count(BOUND658)==1)
exec(compile(src.split(BOUND658,1)[0],'566.py','exec'),globals())
gate658('FROZEN',all(not p.requires_grad for p in model.parameters()))
gate658('ARCH',(NL,H,QH,KVH,HD)==(28,3584,28,4,128))
gate658('PANEL_LOCK',PANEL_SHA=='69facd5448bcfdb75db17a30b633ec5212f97b8caf64106224c1c893423e6b45')
gate658('SENTINEL',sentinel()==S0)
print('BASE',origin,'BLOB',blob658(raw),'GPU',torch.cuda.get_device_name(0),flush=True)
# Same panel generator and random seed as TEST650: identical A/A' and four competing B facts.
rng=random.Random(SEED658);alphabet='ABCDEFGHJKLMNPQRSTUVWXYZ';seen=set()
def word658():
 while True:
  s=''.join(rng.choice(alphabet) for _ in range(9)).capitalize()
  if s not in seen:seen.add(s);return s
PANEL658=[];CAR658=[]
def add658(s):
 ids=tok(source(s),add_special_tokens=False).input_ids
 if ids[:P]!=PIDS:raise RuntimeError('PREFIX_MISMATCH')
 i=len(CAR658);CAR658.append(dict(fact=s,body=ids[P:]));return i
for i in range(N658):
 obj=word658();owners=[word658() for _ in range(4)];regions=[word658() for _ in range(4)]
 a=add658(f'The owner of artifact {obj} is {owners[0]}.')
 acf=add658(f'The owner of artifact {obj} is {owners[1]}.')
 bs=[add658(f'{owners[j]} lives in region {regions[j]}.') for j in range(4)]
 PANEL658.append(dict(case=i,object=obj,owners=owners,regions=regions,a=a,acf=acf,bs=bs,q=f'In which region does the owner of artifact {obj} live?',qa=f'Who owns artifact {obj}?'))
CAR=CAR658;MAX_NEW=MAX_NEW658
panel_sha=hashlib.sha256(json.dumps(PANEL658,sort_keys=True).encode()).hexdigest()
gate658('TEST650_PANEL_LOCK',panel_sha=='db8b211abed216793bf2e4010eb006c862a74202d7512ae53a8a463fe929d466')
@torch.no_grad()
def native_append658(memory,ci):
 # Preserve the locked TEST566 upper-layer computation and H3 injection exactly.
 # Only lower-layer K/V concatenation is replaced with native CUDA, after RoPE position alignment.
 cp=checkpoint(ci,CUT658);oldn=memory[0][0].shape[-2];q=cp['body']
 pos=torch.arange(oldn,oldn+q,device=DEVICE);body=cp['h'][:,P:,:]
 if body.shape!=(1,q,H):raise RuntimeError('H3_BODY_SHAPE')
 def inject(module,args,out):
  current=out[0] if isinstance(out,(tuple,list)) else out
  if current.shape!=body.shape:raise RuntimeError('H3_INJECTION_SHAPE')
  return replace(out,body)
 hook=model.model.layers[CUT658].register_forward_hook(inject)
 try:
  out=model(inputs_embeds=torch.zeros((1,q,H),device=DEVICE,dtype=DTYPE),past_key_values=cache_build(memory),attention_mask=torch.ones((1,oldn+q),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
  grown=list(clone(out.past_key_values))
  for li in range(CUT658+1):
   ok,ov=memory[li];k,v=cp['kv'][li]
   k=rephase(k[:,:,P:,:],cp['pos'][P:],pos);v=v[:,:,P:,:]
   grown[li]=(splice658(ok,k),splice658(ov,v))
  result=tuple(grown)
  for (ok,ov),(nk,nv) in zip(memory,result):
   if not torch.equal(ok,nk[:,:,:oldn,:]) or not torch.equal(ov,nv[:,:,:oldn,:]):raise RuntimeError('PREFIX_MUTATED')
  if len(result)!=NL or any(k.shape!=(1,KVH,oldn+q,HD) or v.shape!=(1,KVH,oldn+q,HD) for k,v in result):raise RuntimeError('CACHE_SHAPE')
  return result
 finally:hook.remove()
@torch.no_grad()
def native_memory658(keys):
 mem=init(keys[0],CUT658)
 for ci in keys[1:]:mem=native_append658(mem,ci)
 return mem
@torch.no_grad()
def baseline_memory658(keys):
 mem=init(keys[0],CUT658)
 for ci in keys[1:]:mem=append(mem,ci,CUT658)
 return mem
@torch.no_grad()
def independent658(keys):
 mem=None
 for ci in keys:
  cp=checkpoint(ci,CUT658);kv=cp['kv']
  if mem is None:mem=kv;continue
  oldn=mem[0][0].shape[-2];q=len(CAR[ci]['body'])
  oldpos=torch.arange(P,P+q,device=DEVICE);newpos=torch.arange(oldn,oldn+q,device=DEVICE)
  mem=tuple((splice658(ok,rephase(k[:,:,P:,:],oldpos,newpos)),splice658(ov,v[:,:,P:,:])) for (ok,ov),(k,v) in zip(mem,kv))
 return mem
@torch.no_grad()
def joint658(keys):
 ids=list(PIDS)
 for ci in keys:ids.extend(CAR[ci]['body'])
 n=len(ids);pos=torch.arange(n,device=DEVICE)
 out=model(input_ids=torch.tensor([ids],device=DEVICE),attention_mask=torch.ones((1,n),device=DEVICE,dtype=torch.long),position_ids=pos.unsqueeze(0),cache_position=pos,use_cache=True,return_dict=True)
 return clone(out.past_key_values)
def validate658(mem,keys):
 n=P+sum(len(CAR[i]['body']) for i in keys)
 if len(mem)!=NL or any(k.shape!=(1,KVH,n,HD) or v.shape!=(1,KVH,n,HD) or not torch.isfinite(k).all() or not torch.isfinite(v).all() for k,v in mem):raise RuntimeError('INVALID_CACHE')
 return mem
def exact658(a,b):return re.sub(r'[^a-z]','',a.casefold())==re.sub(r'[^a-z]','',b.casefold())

# TEST658 — CONTENT-CAUSAL ROUTING: physical slot reversal vs newly forged semantic facts.
# All cartridges remain numeric KV/H3; no persistent text retrieval or weight changes.
import itertools
T658=time.perf_counter();RESULTS658=[];ROUTES658=[];ERRORS658=[]
@torch.no_grad()
def h3sig658(ci):
 cp=checkpoint(ci,CUT658);h=cp['h'][:,P:,:].float()
 if h.ndim!=3 or h.shape[-1]!=H or not h.shape[1]:raise RuntimeError('H3_SHAPE')
 return h.mean(1).squeeze(0).contiguous()
@torch.no_grad()
def scores658(query,candidates):
 out=torch.empty((len(candidates),),device=DEVICE,dtype=torch.float32)
 native658.cosine_scores(out,query.float().contiguous(),torch.stack([v.float() for v in candidates]).contiguous())
 return out.cpu().tolist()
def order658(sc):return sorted(range(len(sc)),key=lambda j:(-sc[j],j))
def grade658(ans,gold,regions):
 hits=[v for v in regions if re.search(r'(?<![A-Za-z])'+re.escape(v)+r'(?![A-Za-z])',ans,re.I)]
 return dict(strict=exact658(ans,gold),region=(len(hits)==1 and hits[0].casefold()==gold.casefold()))
QIDS658={z['case']:add658(z['q']) for z in PANEL658}
# Factorial controls: BASE and SEMANTIC_SWAP (owners 0/1 now live in the opposite regions),
# each under BASE_ORDER and SLOT_SWAP01. This distinguishes fact identity from slot identity.
for z in PANEL658:
 i=z['case'];regions=z['regions'];owners=z['owners'];qci=QIDS658[i]
 try:
  qh=h3sig658(qci)
  b_sem=[add658(f'{owners[0]} lives in region {regions[1]}.'),add658(f'{owners[1]} lives in region {regions[0]}.')]+list(z['bs'][2:])
  for semantic,bs0,region_by_owner in (('BASE',list(z['bs']),[0,1,2,3]),('SEMANTIC_SWAP',b_sem,[1,0,2,3])):
   for order_tag,perm in (('BASE_ORDER',(0,1,2,3)),('SLOT_SWAP01',(1,0,2,3))):
    bs=[bs0[j] for j in perm];sig=[h3sig658(b) for b in bs]
    for source_label,a_ci,owner_index in (('A',z['a'],0),('CF',z['acf'],1)):
     gold_slot=perm.index(owner_index);gold_region=regions[region_by_owner[owner_index]]
     sc=scores658(h3sig658(a_ci)+qh,sig);rank=order658(sc);pred=rank[0]
     keys_map={'ORACLE_ONE':[a_ci,bs[gold_slot]],'ROUTED_ONE':[a_ci,bs[pred]],'A_ONLY':[a_ci]}
     decoded={}
     for arm,keys in keys_map.items():
      key=tuple(keys)
      if key not in decoded:
       try:decoded[key]=(answer(validate658(native_memory658(keys),keys),z['q']),None)
       except Exception as e:decoded[key]=('',f'{type(e).__name__}: {e}')
      ans,err=decoded[key]
      row=dict(case=i,semantic=semantic,order=order_tag,source=source_label,arm=arm,answer=ans,error=err,gold_region=gold_region,**grade658(ans,gold_region,regions))
      RESULTS658.append(row)
      if err:ERRORS658.append(row)
     rivals=[j for j in range(4) if j!=gold_slot]
     route=dict(case=i,semantic=semantic,order=order_tag,source=source_label,gold_slot=gold_slot,selected_slot=pred,
      gold_fact_identity=owner_index,selected_fact_identity=perm[pred],correct=pred==gold_slot,
      gold_region=gold_region,scores=sc,rank=rank,gold_margin=sc[gold_slot]-max(sc[j] for j in rivals),
      top_margin=sc[rank[0]]-sc[rank[1]])
     ROUTES658.append(route)
     print('CASE',i,semantic,order_tag,source_label,'GOLD_SLOT',gold_slot,'SELECTED',pred,'GOLD_REGION',gold_region,
      'SCORES',[round(x,5) for x in sc],'GOLD_MARGIN',round(route['gold_margin'],6),flush=True)
     for row in RESULTS658[-3:]:print(' ',row['arm'],'STRICT',int(row['strict']),'REGION',int(row['region']),repr(row['answer']),row['error'] or '',flush=True)
 except Exception as e:
  ERRORS658.append(dict(case=i,error=f'{type(e).__name__}: {e}'))
  print('CASE_ERROR',i,traceback.format_exc(),flush=True)
 gc.collect();torch.cuda.empty_cache()
ARMS658=('ORACLE_ONE','ROUTED_ONE','A_ONLY')
SUMMARY658={}
for semantic in ('BASE','SEMANTIC_SWAP'):
 for order_tag in ('BASE_ORDER','SLOT_SWAP01'):
  key=semantic+'/'+order_tag
  SUMMARY658[key]={arm:{metric:sum(bool(r[metric]) for r in RESULTS658 if r['semantic']==semantic and r['order']==order_tag and r['arm']==arm) for metric in ('strict','region')}|{'n':sum(r['semantic']==semantic and r['order']==order_tag and r['arm']==arm for r in RESULTS658)} for arm in ARMS658}
  SUMMARY658[key]['routing_correct']=sum(r['correct'] for r in ROUTES658 if r['semantic']==semantic and r['order']==order_tag)
  SUMMARY658[key]['routing_n']=sum(r['semantic']==semantic and r['order']==order_tag for r in ROUTES658)
# Pairwise check for whether slot permutation changes the selected FACT identity.
PAIR658=[]
for semantic in ('BASE','SEMANTIC_SWAP'):
 for i in range(N658):
  for source_label in ('A','CF'):
   a=next((r for r in ROUTES658 if r['case']==i and r['semantic']==semantic and r['source']==source_label and r['order']=='BASE_ORDER'),None)
   b=next((r for r in ROUTES658 if r['case']==i and r['semantic']==semantic and r['source']==source_label and r['order']=='SLOT_SWAP01'),None)
   if a and b:PAIR658.append(dict(semantic=semantic,case=i,source=source_label,identity_invariant=a['selected_fact_identity']==b['selected_fact_identity']))
report658=dict(test='658',base_blob=BLOB658,panel_sha256=panel_sha,model='Qwen/Qwen2.5-7B-Instruct',cut=CUT658,
 hypothesis='Disentangle B0/B1 physical slot identity from semantic owner-to-region facts by newly forging swapped-content cartridges and counterbalancing candidate slots.',
 routes=ROUTES658,results=RESULTS658,scores=SUMMARY658,slot_identity_audit=PAIR658,errors=ERRORS658,
 weights_frozen=all(not p.requires_grad for p in model.parameters()),sentinel_unchanged=sentinel()==S0,
 seconds=round(time.perf_counter()-T658,2),
 limitations='Semantic swap reverses owner-to-region assignments, not owner identity. Slot swap reorders candidate list; ORACLE_ONE independently verifies new factual mappings. Same eight source cases, so this is a controlled within-panel audit, not out-of-sample generalization. Routing is external numeric readout, not model-internal autonomous retrieval.')
out658=ROOT658/'TEST658_results.json';out658.write_text(json.dumps(report658,indent=2,ensure_ascii=False),encoding='utf-8')
print('='*116+'\nTEST658_FINAL\n'+json.dumps(SUMMARY658,indent=2))
print('SLOT_IDENTITY_INVARIANCE',sum(x['identity_invariant'] for x in PAIR658),'/',len(PAIR658))
print('WEIGHTS_FROZEN',report658['weights_frozen'],'SENTINEL',report658['sentinel_unchanged'])
print('RESULT',out658,'SHA256',hashlib.sha256(out658.read_bytes()).hexdigest())
complete=len(RESULTS658)==N658*2*2*2*len(ARMS658) and len(ROUTES658)==N658*2*2*2 and not ERRORS658 and report658['weights_frozen'] and report658['sentinel_unchanged']
print('STATUS','EXPERIMENT_COMPLETE' if complete else 'ERRORS_PRESENT')
