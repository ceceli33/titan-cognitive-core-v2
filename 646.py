
# ======================================================================================================================
# TEST646 — AKBASCORE / BLIND MULTI-CARTRIDGE DISCOVERY
# TEST645 WORKING ENGINE BACKBONE · LOCKED TEST627 · 1453 CUT3 · PCA128 · FAISS IVF128 · PEAK50
# NATIVE C++/CUDA SCORING · 48 REFERENCE PARITY · 16 BLIND TWO-SOURCE TASKS
# NO GOLD IN ROUTING · NO TRAINING · NO GRADIO · NO ENGINE MODIFICATION · SINGLE COLAB CELL
# ======================================================================================================================
import os,sys,time,json,hashlib,urllib.request,subprocess,shutil,importlib.util,random,math,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
TEST="646";SEED=646;TOTAL=1453;NBLIND=16;NREF=48;TOL=0.002
ROOT=Path("/content/AKBASCORE_TEST646");ROOT.mkdir(parents=True,exist_ok=True)
URL="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/HIPPOCAMPUS_full.py"
SOURCE_BLOB="c2c77b170365a08499bbc55d523bb04d46a12a6c"
ENGINE_EXPECTED="ab53e2d58fb3dc5f11409a5112e92feea8762544e4b30b8a77a39b60332de4d1"
def log(*a):print(*a,flush=True)
def utc():return datetime.now(timezone.utc).isoformat()
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def save(name,obj):
 p=ROOT/name;p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding="utf-8");return p
def fetch(url):
 with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"AKBASCORE-TEST646"}),timeout=180) as r:return r.read()
log("="*120);log("TEST646 — BLIND MULTI-CARTRIDGE DISCOVERY");log("START",utc());log("="*120)
log("[1/10] ENVIRONMENT")
import torch
assert torch.cuda.is_available(),"CUDA GPU REQUIRED — ENABLE GPU IN COLAB"
assert torch.version.cuda is not None,"CUDA-ENABLED PYTORCH REQUIRED"
for pkg,mod in [("transformers","transformers"),("accelerate","accelerate"),("faiss-cpu","faiss"),("ninja","ninja")]:
 if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import transformers,faiss,ninja
from torch.utils.cpp_extension import load_inline,CUDA_HOME
if not shutil.which("ninja"):
 p=Path(ninja.__file__).resolve().parent/"data"/"bin"
 if (p/"ninja").exists():os.environ["PATH"]=str(p)+os.pathsep+os.environ.get("PATH","")
assert CUDA_HOME is not None and (Path(CUDA_HOME)/"bin"/"nvcc").exists(),"CUDA TOOLKIT/NVCC REQUIRED"
assert shutil.which("ninja") and shutil.which("c++"),"NATIVE BUILD TOOLS MISSING"
log("GPU",torch.cuda.get_device_name(0),"TORCH",torch.__version__,"CUDA",torch.version.cuda)
log("[2/10] VERIFIED ORIGINAL ENGINE SOURCE")
raw=fetch(URL)
assert blob(raw)==SOURCE_BLOB,"SOURCE HASH MISMATCH"
src=raw.decode("utf-8")
assert src.count("#<<ENGINE_END>>")==1,"ENGINE BOUNDARY CHANGED"
part1=src.split("#<<ENGINE_END>>",1)[0]
dependency='("gradio","gradio"),'
gradio_import="import numpy as np,torch,transformers,gradio as gr"
assert part1.count(dependency)==1 and part1.count(gradio_import)==1,"BOOTSTRAP STRUCTURE CHANGED"
part1=part1.replace(dependency,"")
part1=part1.replace(gradio_import,'import numpy as np,torch,transformers\nimport types as _types\ngr=_types.SimpleNamespace(__version__="DISABLED_TEST646")')
log("SOURCE_VERIFIED",SOURCE_BLOB)
log("[3/10] REUSE OR BOOT LOCKED TEST627 ENGINE")
ready=bool(globals().get("ENGINE_READY",False) and globals().get("MAM_SH_PART1_OK",False) and globals().get("ENGINE_SHA256")==ENGINE_EXPECTED and "ENG" in globals() and "INSTR" in globals())
if ready:log("REUSING LOCKED ENGINE — NO REBUILD")
else:
 log("COLD BOOT — ORIGINAL LOCKED ENGINE")
 t=time.perf_counter()
 exec(compile(part1,"HIPPOCAMPUS_TEST646_BOOTSTRAP.py","exec"),globals())
 log("ENGINE_BOOT_SECONDS",round(time.perf_counter()-t,3))
assert ENGINE_READY and MAM_SH_PART1_OK and ENGINE_SHA256==ENGINE_EXPECTED,"ENGINE LOCK FAILURE"
assert all(ok for _,ok in ENGINE_LOCK),"ENGINE INTEGRITY FAILURE"
assert len(ENG.MANIFEST)==TOTAL and len(ENG.K)==TOTAL,"CARTRIDGE COUNT FAILURE"
assert len(ENG.QUEST)==48 and len(ENG.PANEL624)==48,"PANEL FAILURE"
assert tuple(ENG.LOCKED627)==("PEAK50",0.5,0.0),"RERANK CONFIGURATION CHANGED"
assert not ENG.model.training and not any(p.requires_grad for p in ENG.model.parameters()),"MODEL NOT FROZEN"
weight_before=INSTR.info["weight_sha0"];sentinel_before=INSTR.sentinel()
beta=float(ENG.BETA)
log("ENGINE_READY",ENGINE_SHA256,"CARTRIDGES",TOTAL,"BETA",beta)
log("[4/10] BUILD NATIVE CUDA ADDRESS SCORER")
CPP=r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <algorithm>
#include <cmath>
#include <vector>
torch::Tensor native_features(torch::Tensor q,torch::Tensor k,double beta){
 TORCH_CHECK(q.is_cuda() && k.is_cuda(),"CUDA required");
 TORCH_CHECK(q.device()==k.device(),"Device mismatch");
 TORCH_CHECK(q.scalar_type()==torch::kFloat32 && k.scalar_type()==torch::kFloat32,"FP32 required");
 TORCH_CHECK(q.dim()==2 && k.dim()==2 && q.size(1)==k.size(1),"Shape mismatch");
 TORCH_CHECK(q.size(0)>0 && k.size(0)>0,"Empty input");
 TORCH_CHECK(q.is_contiguous() && k.is_contiguous(),"Contiguous required");
 TORCH_CHECK(std::isfinite(beta) && beta>0,"Invalid beta");
 auto sim=at::matmul(q,k.transpose(0,1)).to(at::kDouble);
 auto n=(at::logsumexp(sim*beta,1)-std::log((double)k.size(0))).mean()/beta;
 auto p=std::get<0>(sim.max(1)).mean();
 return at::stack({n,p});
}
torch::Tensor native_batch(torch::Tensor q,std::vector<torch::Tensor> memories,double beta){
 TORCH_CHECK(!memories.empty(),"No candidates");
 std::vector<torch::Tensor> scores;scores.reserve(memories.size());
 for(const auto& k:memories)scores.push_back(native_features(q,k,beta));
 return at::stack(scores);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
 m.def("features",&native_features);
 m.def("batch",&native_batch);
}
"""
native_sha=sha(CPP.encode())
build=ROOT/"build_native";build.mkdir(exist_ok=True)
(ROOT/"native646.cpp").write_text(CPP)
os.environ["MAX_JOBS"]="2"
t=time.perf_counter()
native646=load_inline(name="akbascore_native646",cpp_sources=[CPP],functions=None,
 extra_cflags=["-O1"],with_cuda=False,verbose=True,build_directory=str(build))
compile_seconds=round(time.perf_counter()-t,3)
log("NATIVE_COMPILE_PASS",compile_seconds,"SOURCE_SHA256",native_sha)
log("[5/10] NATIVE SELFTEST")
q=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float32,device="cuda")
k=torch.tensor([[1.,0.],[0.,1.],[1.,1.]],dtype=torch.float32,device="cuda")
with torch.inference_mode():
 nv=native646.features(q,k,beta).cpu().tolist()
 sim=(q.cpu()@k.cpu().T).double()
 py_native=float(((torch.logsumexp(sim*beta,dim=1)-math.log(k.shape[0]))/beta).mean())
 py_peak=float(sim.max(dim=1).values.mean())
selftest=dict(native=abs(nv[0]-py_native)<=TOL,peak=abs(nv[1]-py_peak)<=TOL)
try:native646.features(q,k[:,:1].contiguous(),beta);selftest["shape_guard"]=False
except Exception:selftest["shape_guard"]=True
try:native646.features(q,k,0.0);selftest["beta_guard"]=False
except Exception:selftest["beta_guard"]=True
assert all(selftest.values()),f"NATIVE SELFTEST FAILED: {selftest}"
log("SELFTEST_PASS",selftest)
del q,k
log("[6/10] NATIVE PEAK50 PARITY — 48 REFERENCE QUESTIONS")
reference=[]
for qi in range(NREF):
 row=dict(question=qi)
 try:
  q=ENG.QUEST[qi].detach().float().cpu().contiguous()
  ids,route=ENG.candidate626(q)
  assert ids,"EMPTY REFERENCE CANDIDATES"
  fs=ENG.features626(q,ids)
  original_ranking,_=ENG.rerank626(ids,fs,ENG.LOCKED627)
  qgpu=q.to("cuda")
  with torch.inference_mode():
   scores=native646.batch(qgpu,[ENG.K[ci].detach().float().contiguous().to("cuda") for ci in ids],beta).cpu().tolist()
  native_scores={int(ci):0.5*float(pair[0])+0.5*float(pair[1]) for ci,pair in zip(ids,scores)}
  native_ranking=sorted(ids,key=lambda ci:(-native_scores[ci],ci))
  native_errors=[abs(float(fs[0][ci])-float(scores[j][0])) for j,ci in enumerate(ids)]
  peak_errors=[abs(float(fs[1][ci])-float(scores[j][1])) for j,ci in enumerate(ids)]
  max_error=max(native_errors+peak_errors)
  row.update(status="PASS" if max_error<=TOL and native_ranking==original_ranking else "FAIL",
   reference_top5=original_ranking[:5],native_top5=native_ranking[:5],
   selected_equal=bool(native_ranking[0]==original_ranking[0]),
   ranking_equal=bool(native_ranking==original_ranking),max_error=max_error,candidates=len(ids))
 except Exception as e:row.update(status="ERROR",error=repr(e))
 reference.append(row)
 if (qi+1)%8==0:
  log("REFERENCE",qi+1,"/",NREF,"PASS",sum(x["status"]=="PASS" for x in reference))
  save("REFERENCE_PROGRESS.json",reference)
save("REFERENCE_PARITY.json",reference)
ref_pass=sum(x["status"]=="PASS" for x in reference)
log("REFERENCE_PARITY",ref_pass,"/",NREF)
log("[7/10] FREEZE BLIND TWO-SOURCE TASKS")
# Gold IDs exist only in sealed evaluation records. They are never passed into routing.
# Source facts are used to construct new questions, not supplied to the model at retrieval.
rng=random.Random(SEED)
all_countries=[]
for ci,item in enumerate(ENG.CAR):
 try:
  fact=str(item["fact"])
  country=fact.split("The current capital of ",1)[1].split(" is ",1)[0].strip()
  if country:all_countries.append((ci,country))
 except Exception:continue
assert len(all_countries)>=2*NBLIND,"NOT ENOUGH STRUCTURED FACTS"
rng.shuffle(all_countries)
pairs=[];used=set()
for ci,country in all_countries:
 if ci in used:continue
 for cj,other in all_countries:
  if cj==ci or cj in used or other==country:continue
  used.update((ci,cj))
  pairs.append((ci,country,cj,other))
  break
 if len(pairs)==NBLIND:break
assert len(pairs)==NBLIND,"UNABLE TO CONSTRUCT BLIND PAIRS"
sealed=[];public=[]
for i,(a,ca,b,cb) in enumerate(pairs):
 question=f"Identify the present capital of {ca} and the present capital of {cb}."
 sealed.append(dict(case=i,question=question,gold_ids=sorted([a,b]),countries=[ca,cb],
  gold_answers=[str(ENG.CAR[a]["gold"]),str(ENG.CAR[b]["gold"])]))
 public.append(dict(case=i,question=question))
seal_sha=sha(json.dumps(sealed,sort_keys=True,ensure_ascii=False).encode())
save("BLIND_TASKS_PUBLIC.json",public)
save("BLIND_GOLD_SEALED.json",dict(sha256=seal_sha,cases=sealed))
log("BLIND_TASKS_FROZEN",NBLIND,"SEAL",seal_sha)
log("[8/10] AUTONOMOUS TWO-SOURCE DISCOVERY — NO GOLD IN ROUTER")
# The frozen router receives one joint question. It is not told that the correct IDs
# are in a particular subset and it receives no answer key or corrective feedback.
blind_predictions=[]
for case in public:
 i=case["case"];question=case["question"]
 row=dict(case=i,question=question)
 try:
  t=time.perf_counter()
  with torch.inference_mode():
   rep,_=ENG.question_rep(question)
   q=ENG.apply(rep[3],ENG.FIT).detach().float().cpu().contiguous()
  ids,route=ENG.candidate626(q)
  fs=ENG.features626(q,ids)
  ranking,_=ENG.rerank626(ids,fs,ENG.LOCKED627)
  row.update(status="OK",selected=ranking[:2],top5=ranking[:5],
   candidates=len(ids),touched=route["touched"],elapsed_ms=round((time.perf_counter()-t)*1000,3))
 except Exception as e:row.update(status="ERROR",error=repr(e),selected=[],top5=[])
 blind_predictions.append(row)
 log("BLIND",i+1,"/",NBLIND,row["status"],"SELECTED",row["selected"])
 save("BLIND_PREDICTIONS_PROGRESS.json",blind_predictions)
save("BLIND_PREDICTIONS_LOCKED.json",blind_predictions)
prediction_sha=sha(json.dumps(blind_predictions,sort_keys=True,ensure_ascii=False).encode())
log("PREDICTIONS_FROZEN",prediction_sha)
log("[9/10] POST-HOC BLIND EVALUATION")
# Gold is opened only after every prediction has been written.
evaluation=[]
for pred,gold in zip(blind_predictions,sealed):
 assert pred["case"]==gold["case"],"CASE ALIGNMENT FAILURE"
 expected=set(gold["gold_ids"])
 selected=set(pred.get("selected",[]))
 top5=set(pred.get("top5",[]))
 row=dict(case=pred["case"],expected_ids=sorted(expected),
  selected_ids=sorted(selected),both_top2=bool(expected<=selected),
  both_top5=bool(expected<=top5),
  at_least_one_top2=bool(expected&selected),
  status=pred["status"])
 evaluation.append(row)
 log("EVALUATE",row["case"],"BOTH_TOP2",row["both_top2"],"BOTH_TOP5",row["both_top5"])
save("BLIND_EVALUATION.json",evaluation)
log("[10/10] FINAL IMMUTABILITY / REPORT")
frozen=INSTR.frozen()
weight_after=INSTR.full_weight_sha()
integrity=dict(source_verified=blob(raw)==SOURCE_BLOB,
 engine_verified=ENGINE_SHA256==ENGINE_EXPECTED,
 engine_lock=all(ok for _,ok in ENGINE_LOCK),
 weight_unchanged=weight_after==weight_before,
 sentinel_unchanged=INSTR.sentinel()==sentinel_before,
 zero_trainable=frozen["trainable_tensors"]==0,
 no_optimizer=not frozen["optimizer"],
 no_foreign_hooks=frozen["foreign_hooks"]==0,
 no_residual_hooks=not frozen["residual_hooks"])
summary=dict(test=TEST,utc=utc(),gpu=torch.cuda.get_device_name(0),
 native_compile_seconds=compile_seconds,native_source_sha256=native_sha,
 reference_parity=dict(pass_count=ref_pass,total=NREF),
 blind_discovery=dict(cases=NBLIND,
  both_top2=sum(x["both_top2"] for x in evaluation),
  both_top5=sum(x["both_top5"] for x in evaluation),
  at_least_one_top2=sum(x["at_least_one_top2"] for x in evaluation),
  errors=sum(x["status"]!="OK" for x in evaluation)),
 gold_seal_sha256=seal_sha,predictions_sha256=prediction_sha,
 integrity=integrity,
 experimental_scope=[
  "Native C++/ATen CUDA scoring is compared against the original PEAK50 features.",
  "The original FAISS IVF128 candidate generator is retained unchanged.",
  "Blind tasks use new two-country question combinations over previously stored synthetic facts.",
  "No gold IDs or expected answers are passed to candidate selection or reranking.",
  "This evaluates independent two-source discovery, not yet multi-cartridge answer synthesis.",
  "The task generator inspects source facts; this is an internal sealed split, not external third-party blinding.",
  "The model remains frozen and the original engine is not modified."
 ])
summary["infrastructure_status"]="PASS" if ref_pass==NREF and all(integrity.values()) and all(selftest.values()) else "FAIL"
summary["blind_status"]="MEASURED" if len(evaluation)==NBLIND else "INCOMPLETE"
summary["status"]="PASS" if summary["infrastructure_status"]=="PASS" and summary["blind_status"]=="MEASURED" else "FAIL"
out=save("TEST646_RESULT.json",dict(summary=summary,selftest=selftest,
 reference=reference,blind_predictions=blind_predictions,blind_evaluation=evaluation))
log("="*120)
log("TEST646_STATUS",summary["status"])
log("REFERENCE_PARITY",ref_pass,"/",NREF)
log("BLIND_BOTH_TOP2",summary["blind_discovery"]["both_top2"],"/",NBLIND)
log("BLIND_BOTH_TOP5",summary["blind_discovery"]["both_top5"],"/",NBLIND)
log("BLIND_AT_LEAST_ONE",summary["blind_discovery"]["at_least_one_top2"],"/",NBLIND)
log("INTEGRITY",json.dumps(integrity))
log("RESULT",out)
log("="*120)
if summary["status"]!="PASS":raise RuntimeError("TEST646_INFRASTRUCTURE_FAILURE — RESULTS SAVED")
