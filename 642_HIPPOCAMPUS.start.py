
# ==========================================================================================================
# TEST642 — AKBASCORE · PHASE 1 · RELATIONAL MEMORY / FIRST COMPOSITION PROBE
# SOURCE: HIPPOCAMPUS_full.py · TEST627 LOCKED ENGINE · NO GRADIO · NO FIGURES
# NO TASK-SPECIFIC PATCHES · NO TRAINING · NO ANSWER INJECTION · FROZEN QWEN
# ==========================================================================================================
import os,sys,re,json,time,hashlib,urllib.request,types,traceback,random,gc
from pathlib import Path
from datetime import datetime,timezone
TEST="642";SEED=642;N_PAIRS=8
REPO="https://raw.githubusercontent.com/ceceli33/titan-cognitive-core-v2/main/"
SOURCE="HIPPOCAMPUS_full.py"
SOURCE_BLOB="c2c77b170365a08499bbc55d523bb04d46a12a6c"
OUT=Path("/content/AKBASCORE_TEST642");OUT.mkdir(parents=True,exist_ok=True)
random.seed(SEED)
def log(*a):print(*a,flush=True)
def blob_sha(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"AKBASCORE-TEST642"}),timeout=180) as f:return f.read()
def norm(s):return " ".join(re.sub(r"[^\w\s-]"," ",str(s).casefold()).split())
def exact(s,g):return norm(s)==norm(g)
def contains(s,g):return re.search(r"(?<!\w)"+re.escape(norm(g))+r"(?!\w)",norm(s)) is not None
def utc():return datetime.now(timezone.utc).isoformat()
log("="*108)
log("TEST642 — PHASE 1 / RELATIONAL MEMORY")
log("START",utc())
log("="*108)
import importlib.util,subprocess
for mod,pkg in [("torch","torch"),("transformers","transformers"),("accelerate","accelerate"),("faiss","faiss-cpu"),("numpy","numpy")]:
    if importlib.util.find_spec(mod) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",pkg])
import torch,numpy as np,transformers
assert torch.cuda.is_available(),"CUDA GPU REQUIRED"
log("GPU",torch.cuda.get_device_name(0))
log("[1/7] FETCHING VERIFIED HIPPOCAMPUS SOURCE")
raw=fetch(REPO+SOURCE)
actual_blob=blob_sha(raw)
assert actual_blob==SOURCE_BLOB,f"SOURCE CHANGED: {actual_blob} != {SOURCE_BLOB}"
(OUT/"HIPPOCAMPUS_SOURCE_SHA256.txt").write_text(hashlib.sha256(raw).hexdigest())
src=raw.decode("utf-8")
assert src.count("#<<ENGINE_END>>")==1
part1=src.split("#<<ENGINE_END>>",1)[0]
old='("gradio","gradio"),'
assert part1.count(old)==1
part1=part1.replace(old,"")
old_import="import numpy as np,torch,transformers,gradio as gr"
assert part1.count(old_import)==1
part1=part1.replace(old_import,'import numpy as np,torch,transformers\nimport types as _types\n_gr_version="NOT_INSTALLED_TEST642"\ngr=_types.SimpleNamespace(__version__=_gr_version)')
assert "gr.Blocks(" not in part1
log("SOURCE BLOB VERIFIED",actual_blob)
log("[2/7] STARTING ORIGINAL LOCKED ENGINE WITHOUT GRADIO")
t0=time.perf_counter()
exec(compile(part1,"HIPPOCAMPUS_TEST642_ENGINE.py","exec"),globals())
assert MAM_SH_PART1_OK and ENGINE_READY
assert all(v for _,v in ENGINE_LOCK)
assert not any(p.requires_grad for p in ENG.model.parameters())
assert not ENG.model.training
log("ENGINE READY",round(time.perf_counter()-t0,2),"seconds")
log("CARTRIDGES",ENG.TOTAL,"ENGINE_SHA256",ENGINE_SHA256)
log("[3/7] BUILDING PREDECLARED RELATIONAL PANEL")
# This panel uses two distinct existing fact cartridges per case.
# Expected answers are retained only in the scorer, never passed to the synthesis prompt.
cases=INSTR.cases()
assert len(cases)==48
rng=random.Random(SEED)
indices=list(range(len(cases)));rng.shuffle(indices)
pairs=[]
for i in range(0,len(indices)-1,2):
    a,b=cases[indices[i]],cases[indices[i+1]]
    if a["target"]==b["target"]:continue
    ga=str(ENG.PANEL624[a["case"]]["gold"]).strip()
    gb=str(ENG.PANEL624[b["case"]]["gold"]).strip()
    if not ga or not gb or norm(ga)==norm(gb):continue
    fa=str(ENG.CAR[a["target"]]["fact"])
    fb=str(ENG.CAR[b["target"]]["fact"])
    pat=r"The current capital of (.*?) is (.*?)\."
    ma=re.search(pat,fa);mb=re.search(pat,fb)
    if not ma or not mb:continue
    country_a=ma.group(1);country_b=mb.group(1)
    expected=min((ga,gb),key=lambda x:x.casefold())
    pairs.append(dict(id=len(pairs),a=a,b=b,country_a=country_a,country_b=country_b,
                      gold_a=ga,gold_b=gb,expected=expected))
    if len(pairs)>=N_PAIRS:break
assert len(pairs)==N_PAIRS
panel=[dict(id=p["id"],country_a=p["country_a"],country_b=p["country_b"],
            expected=p["expected"],target_a=p["a"]["target"],target_b=p["b"]["target"]) for p in pairs]
(OUT/"TEST642_PRIVATE_SCORING_PANEL.json").write_text(json.dumps(panel,indent=2))
log("PANEL READY",len(pairs),"pairs")
log("[4/7] DEFINING FROZEN MODEL INFERENCE")
device=ENG.DEVICE
tokenizer=ENG.tok
model=ENG.model
def generate(prompt,max_new=64):
    messages=[{"role":"user","content":prompt}]
    ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt").to(device)
    with torch.inference_mode():
        y=model.generate(input_ids=ids,max_new_tokens=max_new,do_sample=False,
                         pad_token_id=tokenizer.eos_token_id,use_cache=True)
    return tokenizer.decode(y[0,ids.shape[-1]:],skip_special_tokens=True).strip()
def retrieve(question,expected_target=None,override=None):
    with torch.inference_mode():
        rep,_=ENG.question_rep(question)
        q=ENG.apply(rep[3],ENG.FIT).detach().float().cpu().contiguous()
        candidates,route=ENG.candidate626(q)
        features=ENG.features626(q,candidates)
        ranking,_=ENG.rerank626(candidates,features,ENG.LOCKED627)
    if not ranking:raise RuntimeError("NO_ROUTING_CANDIDATE")
    selected=int(ranking[0]) if override is None else int(override)
    path=ENG.POOL/f"cut3_{selected:05d}.pt"
    if ENG.sha627(path)!=ENG.MANIFEST[selected]["sha256"]:raise RuntimeError("CARTRIDGE_SHA_MISMATCH")
    stored=torch.load(path,map_location="cpu",weights_only=True)
    mem=tuple((k.to(device),v.to(device)) for k,v in stored)
    try:
        with torch.inference_mode():answer=str(ENG.answer(mem,question)).strip()
    finally:
        del mem,stored
    return dict(answer=answer,selected=selected,top5=[int(x) for x in ranking[:5]],
                expected_target=expected_target,address_correct=(selected==expected_target if expected_target is not None else None),
                touched=int(route["touched"]))
def synthesis_prompt(country_a,country_b,capital_a,capital_b):
    return (
        "You are comparing two independent memory observations.\n"
        "Observation A: The capital of "+country_a+" is "+capital_a+".\n"
        "Observation B: The capital of "+country_b+" is "+capital_b+".\n"
        "Question: Which of these two capital names comes first in alphabetical order?\n"
        "Return only the capital name. If the observations are insufficient, return UNKNOWN."
    )
def blind_prompt(country_a,country_b):
    return (
        "Which capital name comes first alphabetically: the current capital of "+
        country_a+" or the current capital of "+country_b+
        "? Return only the capital name. If you cannot determine it, return UNKNOWN."
    )
log("[5/7] RUNNING MEMORY / COMPOSITION / CONTROLS")
records=[]
for p in pairs:
    i=p["id"];a=p["a"];b=p["b"]
    row=dict(id=i,country_a=p["country_a"],country_b=p["country_b"],
             target_a=a["target"],target_b=b["target"])
    try:
        start=time.perf_counter()
        ra=retrieve(a["question"],a["target"])
        rb=retrieve(b["question"],b["target"])
        row["read_a"]=ra;row["read_b"]=rb
        row["fact_a_correct"]=contains(ra["answer"],p["gold_a"])
        row["fact_b_correct"]=contains(rb["answer"],p["gold_b"])
        row["both_fact_correct"]=row["fact_a_correct"] and row["fact_b_correct"]
        row["both_address_correct"]=ra["address_correct"] and rb["address_correct"]
        # Memory-derived composition. No original source fact is given to this stage.
        composed=generate(synthesis_prompt(p["country_a"],p["country_b"],ra["answer"],rb["answer"]))
        row["composition_answer"]=composed
        row["composition_correct"]=exact(composed,p["expected"])
        # Control 1: the frozen model receives only the original relational question.
        blind=generate(blind_prompt(p["country_a"],p["country_b"]))
        row["no_memory_answer"]=blind
        row["no_memory_correct"]=exact(blind,p["expected"])
        # Control 2: replace the second observation with a wrong cartridge's decoded answer.
        wrong_id=int(a["target"])
        wrong=retrieve(b["question"],b["target"],override=wrong_id)
        row["wrong_cartridge_answer"]=wrong["answer"]
        corrupted=generate(synthesis_prompt(p["country_a"],p["country_b"],ra["answer"],wrong["answer"]))
        row["wrong_cartridge_composition"]=corrupted
        row["wrong_control_matches_gold"]=exact(corrupted,p["expected"])
        row["wrong_control_changed"]=norm(corrupted)!=norm(composed)
        row["seconds"]=round(time.perf_counter()-start,3)
        row["status"]="COMPLETE"
    except Exception as e:
        row["status"]="ERROR";row["error"]=repr(e)
        log("CASE ERROR",i,traceback.format_exc()[-1800:])
    records.append(row)
    log("CASE",i,"STATUS",row["status"],
        "FACTS",row.get("both_fact_correct"),
        "COMPOSITION",row.get("composition_correct"),
        "NO_MEMORY",row.get("no_memory_correct"),
        "WRONG_CONTROL_CHANGED",row.get("wrong_control_changed"))
    (OUT/"TEST642_PROGRESS.json").write_text(json.dumps(records,indent=2,ensure_ascii=False))
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
log("[6/7] FROZEN-WEIGHT AND ENGINE INTEGRITY AUDIT")
after=INSTR.frozen()
weight_after=INSTR.full_weight_sha()
integrity=dict(
    engine_sha=ENGINE_SHA256==ENGINE_SHA256_EXPECTED,
    source_blob=actual_blob==SOURCE_BLOB,
    engine_lock=all(ok for _,ok in ENGINE_LOCK),
    weight_unchanged=weight_after==INSTR.info["weight_sha0"],
    sentinel_unchanged=after["sentinel"]==INSTR.info["sentinel0"],
    zero_trainable=after["trainable_tensors"]==0,
    no_optimizer=not after["optimizer"],
    no_foreign_hooks=after["foreign_hooks"]==0,
    no_residual_hooks=not after["residual_hooks"])
def count(key):return sum(int(r.get(key) is True) for r in records)
valid=[r for r in records if r["status"]=="COMPLETE"]
summary=dict(
    test="TEST642",phase="RELATIONAL_MEMORY_FIRST_COMPOSITION",
    date_utc=utc(),model=ENG.MODEL_ID,
    total=len(records),completed=len(valid),errors=len(records)-len(valid),
    address_both_correct=count("both_address_correct"),
    facts_both_correct=count("both_fact_correct"),
    composition_correct=count("composition_correct"),
    no_memory_correct=count("no_memory_correct"),
    wrong_control_changed=count("wrong_control_changed"),
    wrong_control_matches_gold=count("wrong_control_matches_gold"),
    integrity=integrity,
    interpretation="Sequential source-free cartridge retrieval followed by text-mediated composition; not latent multi-cartridge fusion or persistent synthesis.",
    limitations=[
        "Existing synthetic capital facts are reused; this is not a blind new-domain holdout.",
        "The second stage receives decoded intermediate answers as text.",
        "The model does not autonomously choose the intermediate questions.",
        "Wrong-cartridge intervention is diagnostic, not proof of causal independence.",
        "Exact-match scoring may reject semantically correct verbose answers.",
        "No synthesis cartridge is written or reloaded.",
        "No task-specific correction or automatic retry is performed."
    ])
status="INTEGRITY_PASS" if all(integrity.values()) else "INTEGRITY_FAIL"
summary["status"]=status
result=dict(summary=summary,records=records,source_sha256=hashlib.sha256(raw).hexdigest(),
            source_blob=actual_blob,engine_sha256=ENGINE_SHA256)
(OUT/"TEST642_RESULT.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
log("[7/7] FINAL RESULT")
log("="*108)
for k,v in summary.items():
    if k not in ("limitations","integrity"):log(k,":",v)
log("INTEGRITY",json.dumps(integrity))
log("OUTPUT",OUT/"TEST642_RESULT.json")
log("="*108)
if status!="INTEGRITY_PASS":raise RuntimeError("TEST642_INTEGRITY_FAILURE — RESULTS PRESERVED")
