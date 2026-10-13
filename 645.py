
# ======================================================================================================================
# TEST645-R3 — AKBASCORE NATIVE MIGRATION / C++ CUDA
# LOCKED TEST627 · 1453 CUT3 · PCA128 · FAISS IVF128 · TOP2 · PEAK50
# C++ NATIVE FILE READER · CUDA BF16 AUDIT · GEOMETRY PARITY · MULTI-CARTRIDGE WORKSPACE
# SINGLE COLAB CELL · NO GRADIO · NO TRAINING · ORIGINAL ENGINE UNCHANGED
# ======================================================================================================================
import os,sys,json,time,hashlib,urllib.request,subprocess,shutil,random,gc,traceback,importlib.util,math,struct
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
TEST="645-R3";SEED=645;TOTAL=1453;NQ=16;NC=16;WORKSPACE_N=4;TOL=0.002
ROOT=Path("/content/AKBASCORE_TEST645");ROOT.mkdir(parents=True,exist_ok=True)
NATIVE_POOL=ROOT/"native_pool";NATIVE_POOL.mkdir(exist_ok=True)
URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/HIPPOCAMPUS_full.py"
SOURCE_BLOB="c2c77b170365a08499bbc55d523bb04d46a12a6c"
ENGINE_EXPECTED="ab53e2d58fb3dc5f11409a5112e92feea8762544e4b30b8a77a39b60332de4d1"
MAGIC=b"AKBCUT3\x00";VERSION=1
def log(*a):print(*a,flush=True)
def utc():return datetime.now(timezone.utc).isoformat()
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def save(name,obj):
 p=ROOT/name
 p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
 return p
def fetch(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"AKBASCORE-TEST645-R3"}),timeout=180) as r:return r.read()
log("="*120);log("TEST645-R3 — AKBASCORE NATIVE MIGRATION");log("START",utc());log("="*120)
log("[1/11] GPU / CUDA ENVIRONMENT DIAGNOSTICS")
if importlib.util.find_spec("torch") is None:
 raise RuntimeError("PYTORCH NOT INSTALLED — USE A CUDA-ENABLED COLAB RUNTIME")
import torch
log("TORCH",torch.__version__,"TORCH_CUDA",torch.version.cuda)
log("CUDA_AVAILABLE",torch.cuda.is_available())
if shutil.which("nvidia-smi"):
 p=subprocess.run(["nvidia-smi"],capture_output=True,text=True)
 log("NVIDIA_SMI_RETURN",p.returncode)
 log((p.stdout or p.stderr)[-3000:])
else:log("NVIDIA_SMI_NOT_FOUND")
if not torch.cuda.is_available():
 reason="CUDA GPU IS NOT AVAILABLE. In Colab select Runtime > Change runtime type > GPU, reconnect, then rerun this cell. No CPU fallback is allowed because CUDA parity is required."
 save("TEST645_ENVIRONMENT_FAILURE.json",dict(test=TEST,status="BLOCKED_NO_CUDA",utc=utc(),torch=str(torch.__version__),torch_cuda=str(torch.version.cuda),reason=reason))
 raise RuntimeError(reason)
if torch.version.cuda is None:raise RuntimeError("CPU-ONLY PYTORCH BUILD — CUDA-ENABLED PYTORCH REQUIRED")
for pkg,mod in [("transformers","transformers"),("accelerate","accelerate"),("faiss-cpu","faiss"),("ninja","ninja")]:
 if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import transformers,faiss,ninja
from torch.utils.cpp_extension import load_inline,CUDA_HOME
if not shutil.which("ninja"):
 ninja_dir=Path(ninja.__file__).resolve().parent/"data"/"bin"
 if (ninja_dir/"ninja").is_file():os.environ["PATH"]=str(ninja_dir)+os.pathsep+os.environ.get("PATH","")
assert shutil.which("ninja"),"NINJA COMPILER MISSING"
assert shutil.which("c++"),"C++ COMPILER MISSING"
assert CUDA_HOME is not None,"CUDA TOOLKIT MISSING"
assert (Path(CUDA_HOME)/"bin"/"nvcc").is_file(),"NVCC MISSING"
assert torch.cuda.get_device_properties(0).total_memory>=30*1024**3,"GPU MEMORY BELOW 30 GiB"
log("GPU",torch.cuda.get_device_name(0),"CUDA",torch.version.cuda,"NINJA",shutil.which("ninja"))
log("[2/11] VERIFIED HIPPOCAMPUS SOURCE")
raw=fetch(URL)
assert blob(raw)==SOURCE_BLOB,"HIPPOCAMPUS SOURCE HASH MISMATCH"
source=raw.decode("utf-8")
assert source.count("#<<ENGINE_END>>")==1,"ENGINE BOUNDARY MISMATCH"
part1=source.split("#<<ENGINE_END>>",1)[0]
dep='("gradio","gradio"),'
imp="import numpy as np,torch,transformers,gradio as gr"
assert part1.count(dep)==1 and part1.count(imp)==1,"BOOTSTRAP IMPORT STRUCTURE CHANGED"
part1=part1.replace(dep,"")
part1=part1.replace(imp,'import numpy as np,torch,transformers\nimport types as _types\ngr=_types.SimpleNamespace(__version__="DISABLED_TEST645")')
log("SOURCE_VERIFIED",SOURCE_BLOB)
log("[3/11] LOCKED ENGINE")
ready=bool(globals().get("ENGINE_READY",False) and globals().get("MAM_SH_PART1_OK",False) and globals().get("ENGINE_SHA256")==ENGINE_EXPECTED and "ENG" in globals() and "INSTR" in globals())
if ready:log("REUSING EXISTING ENGINE")
else:
 log("COLD START — VERIFIED ORIGINAL ENGINE")
 t=time.perf_counter()
 exec(compile(part1,"HIPPOCAMPUS_TEST645_BOOTSTRAP.py","exec"),globals())
 log("ENGINE_BOOT_SECONDS",round(time.perf_counter()-t,3))
assert globals().get("ENGINE_READY",False) and globals().get("MAM_SH_PART1_OK",False),"ENGINE NOT READY"
assert ENGINE_SHA256==ENGINE_EXPECTED,"ENGINE SHA MISMATCH"
assert all(ok for _,ok in ENGINE_LOCK),"ENGINE LOCK FAILURE"
assert len(ENG.MANIFEST)==TOTAL and len(ENG.K)==TOTAL,"CARTRIDGE BANK INCOMPLETE"
assert len(ENG.QUEST)==48 and len(ENG.PANEL624)==48,"QUERY PANEL INCOMPLETE"
assert tuple(ENG.LOCKED627)==("PEAK50",0.5,0.0),"PEAK50 LOCK MISMATCH"
assert not ENG.model.training and not any(p.requires_grad for p in ENG.model.parameters()),"MODEL NOT FROZEN"
assert ENG.RESULT627["status"]=="COMPLETE","TEST627 BASELINE INCOMPLETE"
assert all(ENG.RESULT627["integrity"].values()),"TEST627 BASELINE INTEGRITY FAILURE"
weight_before=INSTR.info["weight_sha0"]
sentinel_before=INSTR.info["sentinel0"]
assert INSTR.sentinel()==sentinel_before,"INITIAL SENTINEL MISMATCH"
beta=float(ENG.BETA)
assert math.isfinite(beta) and beta>0,"INVALID BETA"
log("ENGINE_READY",ENGINE_SHA256,"CARTRIDGES",TOTAL,"BETA",beta)
log("[4/11] NATIVE C++ / CUDA SOURCE")
CPP=r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
torch::Tensor compare_bf16_cuda(torch::Tensor a,torch::Tensor b);
torch::Tensor native_top2(torch::Tensor q,torch::Tensor k,double beta){
 TORCH_CHECK(q.is_cuda() && k.is_cuda(),"CUDA required");
 TORCH_CHECK(q.device()==k.device(),"Device mismatch");
 TORCH_CHECK(q.scalar_type()==torch::kFloat32 && k.scalar_type()==torch::kFloat32,"FP32 required");
 TORCH_CHECK(q.dim()==2 && k.dim()==2 && q.size(1)==k.size(1),"Geometry mismatch");
 TORCH_CHECK(q.size(0)>0 && k.size(0)>0,"Empty geometry");
 TORCH_CHECK(q.is_contiguous() && k.is_contiguous(),"Contiguous required");
 TORCH_CHECK(std::isfinite(beta) && beta>0,"Invalid beta");
 auto sim=at::matmul(q,k.transpose(0,1));
 auto v=std::get<0>(at::topk(sim,std::min<int64_t>(2,k.size(0)),1,true,true));
 return ((at::logsumexp(v.to(at::kDouble)*beta,1)-std::log((double)k.size(0)))/beta).mean();
}
torch::Tensor score_candidates(torch::Tensor q,std::vector<torch::Tensor> memories,double beta){
 TORCH_CHECK(!memories.empty(),"Empty candidates");
 std::vector<torch::Tensor> out;out.reserve(memories.size());
 for(const auto& k:memories)out.push_back(native_top2(q,k,beta));
 return at::stack(out);
}
torch::Tensor project(torch::Tensor q,torch::Tensor center,torch::Tensor basis){
 TORCH_CHECK(q.is_cuda() && center.is_cuda() && basis.is_cuda(),"CUDA required");
 TORCH_CHECK(q.device()==center.device() && q.device()==basis.device(),"Device mismatch");
 TORCH_CHECK(q.scalar_type()==torch::kFloat32 && center.scalar_type()==torch::kFloat32 && basis.scalar_type()==torch::kFloat32,"FP32 required");
 TORCH_CHECK(q.dim()==2 && center.dim()==1 && basis.dim()==2,"Projection dimension mismatch");
 TORCH_CHECK(q.size(1)==center.size(0) && center.size(0)==basis.size(0),"Projection shape mismatch");
 return at::matmul(q-center,basis);
}
uint32_t read32(std::ifstream& f){
 uint8_t b[4];f.read(reinterpret_cast<char*>(b),4);
 if(!f)throw std::runtime_error("Truncated uint32");
 return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);
}
uint64_t read64(std::ifstream& f){
 uint8_t b[8];f.read(reinterpret_cast<char*>(b),8);
 if(!f)throw std::runtime_error("Truncated uint64");
 uint64_t v=0;
 for(int i=0;i<8;i++)v|=uint64_t(b[i])<<(8*i);
 return v;
}
std::vector<torch::Tensor> read_cut3(const std::string& filename,int64_t expected_id){
 std::ifstream f(filename,std::ios::binary);
 if(!f)throw std::runtime_error("Cannot open native cartridge");
 char magic[8];f.read(magic,8);
 if(!f || std::memcmp(magic,"AKBCUT3\0",8)!=0)throw std::runtime_error("Native magic mismatch");
 auto version=read32(f),id=read32(f),count=read32(f);
 if(version!=1 || int64_t(id)!=expected_id || count!=56)throw std::runtime_error("Native header mismatch");
 std::vector<torch::Tensor> out;out.reserve(count);
 for(uint32_t j=0;j<count;j++){
  uint32_t ndim=read32(f);
  if(ndim==0 || ndim>8)throw std::runtime_error("Invalid tensor rank");
  std::vector<int64_t> dims;dims.reserve(ndim);
  uint64_t numel=1;
  for(uint32_t i=0;i<ndim;i++){
   uint64_t d=read64(f);
   if(d==0 || d>1000000 || numel>uint64_t(100000000)/d)throw std::runtime_error("Invalid tensor shape");
   numel*=d;dims.push_back(int64_t(d));
  }
  auto t=torch::empty(dims,torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
  f.read(reinterpret_cast<char*>(t.data_ptr()),static_cast<std::streamsize>(numel*2));
  if(!f)throw std::runtime_error("Truncated tensor payload");
  out.push_back(t);
 }
 if(f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("Trailing bytes");
 return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("top2",&native_top2);
 m.def("score_candidates",&score_candidates);
 m.def("project",&project);
 m.def("read_cut3",&read_cut3);
 m.def("compare",&compare_bf16_cuda);
}
"""
CUDA=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
__global__ void compare_bits_kernel(const uint16_t* a,const uint16_t* b,int64_t n,int* mismatch){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<n && a[i]!=b[i])atomicAdd(mismatch,1);
}
torch::Tensor compare_bf16_cuda(torch::Tensor a,torch::Tensor b){
 TORCH_CHECK(a.is_cuda() && b.is_cuda(),"CUDA required");
 TORCH_CHECK(a.device()==b.device(),"Device mismatch");
 TORCH_CHECK(a.scalar_type()==torch::kBFloat16 && b.scalar_type()==torch::kBFloat16,"BF16 required");
 TORCH_CHECK(a.sizes()==b.sizes(),"Shape mismatch");
 TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),"Contiguous required");
 auto out=torch::zeros({1},torch::TensorOptions().dtype(torch::kInt32).device(a.device()));
 int64_t n=a.numel();
 if(n){
  auto stream=at::cuda::getCurrentCUDAStream(a.get_device());
  compare_bits_kernel<<<static_cast<unsigned int>((n+255)/256),256,0,stream.stream()>>>(
   reinterpret_cast<const uint16_t*>(a.data_ptr()),
   reinterpret_cast<const uint16_t*>(b.data_ptr()),n,out.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
 return out;
}
"""
(ROOT/"native645.cpp").write_text(CPP)
(ROOT/"native645.cu").write_text(CUDA)
native_sha=sha((CPP+"\n"+CUDA).encode())
log("NATIVE_SOURCE_SHA256",native_sha)
log("[5/11] NATIVE COMPILATION")
os.environ["MAX_JOBS"]="2"
os.environ["TORCH_CUDA_ARCH_LIST"]="8.0" if torch.cuda.get_device_capability(0)==(8,0) else f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
build=ROOT/"build_r3";build.mkdir(exist_ok=True)
t=time.perf_counter()
native=load_inline(name="akbascore_native645_r3",cpp_sources=[CPP],cuda_sources=[CUDA],
 functions=None,extra_cflags=["-O1"],extra_cuda_cflags=["-O1"],
 with_cuda=True,verbose=True,build_directory=str(build))
compile_seconds=round(time.perf_counter()-t,3)
log("NATIVE_COMPILE_PASS",compile_seconds)
log("[6/11] NATIVE SELFTEST")
controls={}
a=torch.tensor([1.,2.,3.],dtype=torch.bfloat16,device="cuda")
b=a.clone()
controls["identical"]=int(native.compare(a,b).item())==0
b[1]=8.
controls["mutation"]=int(native.compare(a,b).item())==1
q=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float32,device="cuda")
k=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float32,device="cuda")
with torch.inference_mode():
 ref=float(ENG.top2(q.cpu(),k.cpu()))
 got=float(native.top2(q,k,beta).item())
controls["top2"]=math.isfinite(got) and abs(ref-got)<=TOL
try:native.top2(q,k[:,:1].contiguous(),beta);controls["shape_guard"]=False
except Exception:controls["shape_guard"]=True
try:native.compare(a.float(),b.float());controls["dtype_guard"]=False
except Exception:controls["dtype_guard"]=True
try:native.top2(q,k,0.0);controls["beta_guard"]=False
except Exception:controls["beta_guard"]=True
log("CONTROLS",controls,"REFERENCE",ref,"NATIVE",got)
assert all(controls.values()),"NATIVE_SELFTEST_FAILED"
del a,b,q,k
log("[7/11] LOSSLESS CUT3 EXPORT")
def export_cut3(ci):
 meta=ENG.MANIFEST[ci];source=Path(ENG.POOL)/meta["file"]
 assert source.is_file() and source.stat().st_size==int(meta["bytes"]),"SOURCE FILE SIZE MISMATCH"
 assert file_sha(source)==meta["sha256"],"SOURCE FILE SHA MISMATCH"
 stored=torch.load(source,map_location="cpu",weights_only=True)
 assert len(stored)==28,"INVALID CUT3 LAYER COUNT"
 destination=NATIVE_POOL/f"cut3_{ci:05d}.akb";temporary=destination.with_suffix(".tmp")
 try:
  with open(temporary,"wb") as f:
   f.write(struct.pack("<8sIII",MAGIC,VERSION,ci,56))
   for pair in stored:
    assert isinstance(pair,(list,tuple)) and len(pair)==2,"INVALID KV PAIR"
    for tensor in pair:
     assert torch.is_tensor(tensor) and tensor.dtype==torch.bfloat16,"INVALID BF16 TENSOR"
     tensor=tensor.detach().cpu().contiguous()
     shape=tuple(int(s) for s in tensor.shape)
     assert 1<=len(shape)<=8 and all(s>0 for s in shape),"INVALID TENSOR SHAPE"
     f.write(struct.pack("<I",len(shape)))
     f.write(struct.pack("<"+"Q"*len(shape),*shape))
     f.write(tensor.view(torch.int16).numpy().tobytes())
  os.replace(temporary,destination)
 finally:
  if temporary.exists():temporary.unlink()
 return dict(id=ci,file=destination.name,bytes=destination.stat().st_size,sha256=file_sha(destination))
manifest=[]
for ci in range(TOTAL):
 manifest.append(export_cut3(ci))
 if (ci+1)%100==0 or ci+1==TOTAL:
  log("NATIVE_EXPORT",ci+1,"/",TOTAL)
  save("NATIVE_MANIFEST_PROGRESS.json",manifest)
save("NATIVE_MANIFEST.json",manifest)
log("NATIVE_EXPORT_PASS",len(manifest))
log("[8/11] C++ READ + CUDA BF16 BITWISE AUDIT")
sample_ids=sorted(set([0,1,95,96,1452]+random.Random(SEED).sample(range(TOTAL),11)))
audit=[]
for ci in sample_ids:
 row=dict(id=ci)
 try:
  original=torch.load(Path(ENG.POOL)/ENG.MANIFEST[ci]["file"],map_location="cpu",weights_only=True)
  restored=native.read_cut3(str(NATIVE_POOL/manifest[ci]["file"]),ci)
  assert len(restored)==56 and len(original)==28,"TENSOR COUNT MISMATCH"
  mismatches=0
  for layer in range(28):
   for arm in range(2):
    old=original[layer][arm].contiguous();new=restored[2*layer+arm]
    assert old.dtype==new.dtype and old.shape==new.shape,"TENSOR STRUCTURE MISMATCH"
    mismatches+=int(native.compare(old.to("cuda"),new.to("cuda")).item())
  row.update(status="PASS" if mismatches==0 else "FAIL",tensors=56,mismatches=mismatches)
  del original,restored
 except Exception as e:row.update(status="ERROR",error=repr(e))
 audit.append(row)
 log("NATIVE_AUDIT",ci,row["status"],row.get("mismatches"))
 save("NATIVE_AUDIT_PROGRESS.json",audit)
log("[9/11] PCA128 / TOP2 / LOCKED PEAK50")
rng=random.Random(SEED);questions=sorted(rng.sample(range(48),NQ))
geometry_rows=[];address_rows=[]
center=ENG.CENTER.detach().float().cpu().contiguous()
basis=ENG.BASIS.detach().float().cpu().contiguous()
center_gpu=center.to("cuda");basis_gpu=basis.to("cuda")
for qi in questions:
 row=dict(question=qi)
 try:
  target=int(ENG.PANEL624[qi]["target"])
  ids=sorted([target]+rng.sample([j for j in range(TOTAL) if j!=target],NC-1))
  q=ENG.QUEST[qi].detach().float().cpu().contiguous()
  ks=[ENG.K[j].detach().float().cpu().contiguous() for j in ids]
  assert q.ndim==2 and q.shape[1]==3584,"QUERY GEOMETRY MISMATCH"
  assert all(k.ndim==2 and k.shape[1]==3584 for k in ks),"CARTRIDGE GEOMETRY MISMATCH"
  qgpu=q.to("cuda");kgpu=[k.to("cuda") for k in ks]
  with torch.inference_mode():
   py=[float(ENG.top2(q,k)) for k in ks]
   nv=native.score_candidates(qgpu,kgpu,beta).detach().cpu().tolist()
   cpu_proj=(q-center)@basis
   gpu_proj=native.project(qgpu,center_gpu,basis_gpu).detach().cpu()
  error=max(abs(x-y) for x,y in zip(py,nv))
  proj_error=float((cpu_proj-gpu_proj).abs().max())
  py_rank=sorted(range(NC),key=lambda j:(-py[j],ids[j]))
  nv_rank=sorted(range(NC),key=lambda j:(-nv[j],ids[j]))
  finite=all(math.isfinite(x) for x in py+nv)
  row.update(target=target,candidates=ids,max_error=error,projection_error=proj_error,
   ranking_equal=py_rank==nv_rank,status="PASS" if finite and error<=TOL and proj_error<=TOL and py_rank==nv_rank else "FAIL")
  del qgpu,kgpu
 except Exception as e:row.update(status="ERROR",error=repr(e))
 geometry_rows.append(row)
 log("GEOMETRY",qi,row["status"],row.get("max_error"),row.get("projection_error"))
 save("GEOMETRY_PROGRESS.json",geometry_rows)
 try:
  q=ENG.QUEST[qi];target=int(ENG.PANEL624[qi]["target"])
  candidates,route=ENG.candidate626(q)
  features=ENG.features626(q,candidates)
  ranking,_=ENG.rerank626(candidates,features,ENG.LOCKED627)
  ar=dict(question=qi,target=target,selected=ranking[0] if ranking else None,
   correct=bool(ranking and ranking[0]==target),top5=ranking[:5],
   candidates=len(candidates),touched=route.get("touched"))
 except Exception as e:ar=dict(question=qi,correct=False,error=repr(e))
 address_rows.append(ar)
 log("LOCKED_ADDRESS",qi,"PASS" if ar["correct"] else "FAIL",ar.get("selected"))
 save("ADDRESS_PROGRESS.json",address_rows)
log("[10/11] MULTI-CARTRIDGE WORKSPACE")
workspace_ids=[int(ENG.PANEL624[qi]["target"]) for qi in questions[:WORKSPACE_N]]
workspace=[]
for ci in workspace_ids:
 native_file=NATIVE_POOL/manifest[ci]["file"]
 workspace.append(dict(id=ci,source_sha256=ENG.MANIFEST[ci]["sha256"],
  native_sha256=manifest[ci]["sha256"],native_file=str(native_file),
  fact=ENG.CAR[ci]["fact"],gold=ENG.CAR[ci]["gold"]))
workspace_report=dict(status="READY",cartridges=workspace,
 capability="MULTI_CARTRIDGE_REFERENCES_PREPARED",
 native_binary_reader_available=True,relational_inference_tested=False,
 latent_composition_tested=False,provenance_available=True)
save("RELATIONAL_WORKSPACE.json",workspace_report)
log("WORKSPACE_READY",workspace_ids)
log("[11/11] IMMUTABILITY AND FINAL REPORT")
frozen=INSTR.frozen()
weight_after=INSTR.full_weight_sha()
integrity=dict(
 source_verified=blob(raw)==SOURCE_BLOB,
 engine_verified=ENGINE_SHA256==ENGINE_EXPECTED,
 engine_lock=all(ok for _,ok in ENGINE_LOCK),
 weight_unchanged=weight_after==weight_before,
 sentinel_unchanged=frozen["sentinel"]==sentinel_before,
 zero_trainable=frozen["trainable_tensors"]==0,
 no_optimizer=not frozen["optimizer"],
 no_foreign_hooks=frozen["foreign_hooks"]==0,
 no_residual_hooks=not frozen["residual_hooks"])
summary=dict(
 test=TEST,utc=utc(),gpu=torch.cuda.get_device_name(0),
 native_files=len(manifest),native_audited=len(audit),
 native_audit_pass=sum(r["status"]=="PASS" for r in audit),
 geometry_cases=len(geometry_rows),
 geometry_pass=sum(r["status"]=="PASS" for r in geometry_rows),
 locked_address_pass=sum(bool(r["correct"]) for r in address_rows),
 workspace_cartridges=len(workspace),
 compile_seconds=compile_seconds,native_source_sha256=native_sha,
 integrity=integrity,
 limitations=[
  "1453 CUT3 files are exported into native binary storage.",
  "C++ reads native CUT3; CUDA checks sampled BF16 tensors bitwise.",
  "Full export SHA manifests do not establish full-population tensor parity.",
  "C++ ATen computes TOP2 and PCA projection on CUDA.",
  "Original FAISS IVF128 and PEAK50 remain the reference, not a native replacement.",
  "The frozen model still runs through PyTorch and Transformers.",
  "Workspace contains provenance references, not demonstrated relational reasoning.",
  "The panel is not a blind generalization experiment."
 ])
summary["status"]="PASS" if (
 len(manifest)==TOTAL and
 len(audit)==len(sample_ids) and all(r["status"]=="PASS" for r in audit) and
 len(geometry_rows)==NQ and all(r["status"]=="PASS" for r in geometry_rows) and
 len(address_rows)==NQ and all(r["correct"] for r in address_rows) and
 all(controls.values()) and all(integrity.values())
) else "FAIL"
result_path=save("TEST645_RESULT.json",dict(summary=summary,controls=controls,
 native_audit=audit,geometry=geometry_rows,addressing=address_rows,workspace=workspace_report))
log("="*120)
log("TEST645_STATUS",summary["status"])
log("NATIVE_FILES",summary["native_files"])
log("CUDA_BITWISE_AUDIT",summary["native_audit_pass"],"/",summary["native_audited"])
log("GEOMETRY_PARITY",summary["geometry_pass"],"/",summary["geometry_cases"])
log("LOCKED_ADDRESS",summary["locked_address_pass"],"/",len(address_rows))
log("WORKSPACE",summary["workspace_cartridges"])
log("INTEGRITY",json.dumps(integrity))
log("RESULT",result_path)
log("="*120)
if summary["status"]!="PASS":raise RuntimeError("TEST645_FAILED — RESULTS SAVED")
