# ================================================================================================
# AKBASCORE - TEST 291
# CAPABILITY GATED CROSS LINGUAL LATENT TRANSFER
# EN / TR / ES / FR / DE - VANILLA GATE - CROSS LANGUAGE GEOMETRY - CAUSAL LOGIT READOUT
# SEASC L0-L25 - L26-L27 OFF - FIXED RSS
# NO TRAINING - NO CLASSIFIER - NO SELECTION
# ================================================================================================
import os,sys,json,time,random,hashlib,subprocess,importlib.util,re
from datetime import datetime,timezone
from pathlib import Path
for m,p in [("torch","torch"),("transformers","transformers"),("numpy","numpy")]:
    if importlib.util.find_spec(m) is None:subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
import numpy as np,torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA required")
SEED=291
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
DEVICE=torch.device("cuda");MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
TOTAL=28;H_EXPECT=3584;EPS=1e-8;END=25;TARGET_RSS=.250235055
IVME=.10;SONUM=.30;ZIRVE=.70;TABAN=.20;REPORT=[19,23,25,26,27]
ROOT=Path("/content/AKBASCORE_TEST291") if os.path.isdir("/content") else Path("/tmp/AKBASCORE_TEST291")
ROOT.mkdir(parents=True,exist_ok=True);START=datetime.now(timezone.utc).isoformat(timespec="milliseconds")
LANG=["EN","TR","ES","FR","DE"]
ITEMS=[
{"id":"RED","EN":"red","TR":"kırmızı","ES":"rojo","FR":"rouge","DE":"rot"},
{"id":"BLUE","EN":"blue","TR":"mavi","ES":"azul","FR":"bleu","DE":"blau"},
{"id":"GREEN","EN":"green","TR":"yeşil","ES":"verde","FR":"vert","DE":"grün"},
{"id":"BLACK","EN":"black","TR":"siyah","ES":"negro","FR":"noir","DE":"schwarz"},
{"id":"WHITE","EN":"white","TR":"beyaz","ES":"blanco","FR":"blanc","DE":"weiß"},
{"id":"IRON","EN":"iron","TR":"demir","ES":"hierro","FR":"fer","DE":"Eisen"},
{"id":"GLASS","EN":"glass","TR":"cam","ES":"vidrio","FR":"verre","DE":"Glas"},
{"id":"HAMMER","EN":"hammer","TR":"çekiç","ES":"martillo","FR":"marteau","DE":"Hammer"},
{"id":"ANCHOR","EN":"anchor","TR":"çapa","ES":"ancla","FR":"ancre","DE":"Anker"},
{"id":"JUSTICE","EN":"justice","TR":"adalet","ES":"justicia","FR":"justice","DE":"Gerechtigkeit"},
{"id":"FREEDOM","EN":"freedom","TR":"özgürlük","ES":"libertad","FR":"liberté","DE":"Freiheit"},
{"id":"BRIGHT","EN":"bright","TR":"parlak","ES":"brillante","FR":"brillant","DE":"hell"}]
CTX={
"EN":["The recorded value is {x}.","Question: What value was recorded?\nAnswer: {x}.","Operator: Which value should I use?\nAssistant: Use {x}.","SYSTEM_RECORD\nfield=value\nvalue={x}\nEND_RECORD","After checking the sealed note, the researcher found a single entry. It was {x}."],
"TR":["Kaydedilen değer {x}.","Soru: Hangi değer kaydedildi?\nCevap: {x}.","Operatör: Hangi değeri kullanmalıyım?\nAsistan: {x} kullan.","SISTEM_KAYDI\nalan=deger\ndeger={x}\nKAYIT_SONU","Mühürlü notu kontrol eden araştırmacı tek bir kayıt buldu. Bu {x} idi."],
"ES":["El valor registrado es {x}.","Pregunta: ¿Qué valor se registró?\nRespuesta: {x}.","Operador: ¿Qué valor debo usar?\nAsistente: Usa {x}.","REGISTRO_SISTEMA\ncampo=valor\nvalor={x}\nFIN_REGISTRO","Tras revisar la nota sellada, el investigador encontró una sola entrada. Era {x}."],
"FR":["La valeur enregistrée est {x}.","Question : Quelle valeur a été enregistrée ?\nRéponse : {x}.","Opérateur : Quelle valeur dois-je utiliser ?\nAssistant : Utilisez {x}.","ENREGISTREMENT_SYSTEME\nchamp=valeur\nvaleur={x}\nFIN","Après avoir vérifié la note scellée, le chercheur a trouvé une seule entrée. C'était {x}."],
"DE":["Der aufgezeichnete Wert ist {x}.","Frage: Welcher Wert wurde aufgezeichnet?\nAntwort: {x}.","Operator: Welchen Wert soll ich verwenden?\nAssistent: Verwende {x}.","SYSTEMDATENSATZ\nfeld=wert\nwert={x}\nENDE","Nach Prüfung der versiegelten Notiz fand der Forscher einen einzigen Eintrag. Es war {x}."]}
GATE={
"EN":"Repeat exactly the value after VALUE. VALUE: {x}",
"TR":"VALUE kelimesinden sonraki değeri aynen tekrar et. VALUE: {x}",
"ES":"Repite exactamente el valor después de VALUE. VALUE: {x}",
"FR":"Répétez exactement la valeur après VALUE. VALUE: {x}",
"DE":"Wiederhole exakt den Wert nach VALUE. VALUE: {x}"}
BLIND={
"EN":"Return the single value represented by the internal state. Answer with only the value.",
"TR":"İç durumda temsil edilen tek değeri döndür. Yalnızca değeri yaz.",
"ES":"Devuelve el único valor representado por el estado interno. Responde solo con el valor.",
"FR":"Retournez la valeur unique représentée par l'état interne. Répondez uniquement avec la valeur.",
"DE":"Gib den einzelnen Wert zurück, der durch den internen Zustand dargestellt wird. Antworte nur mit dem Wert."}
def utc():return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
def unit(x):return x/x.norm().clamp_min(EPS)
def cos(a,b):return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(EPS))
def nt(x):return re.sub(r"[^\w]+","",x.lower(),flags=re.UNICODE)
def canon(o):return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
print("="*108);print("TEST 291 - CAPABILITY GATED CROSS LINGUAL LATENT TRANSFER");print("="*108);print("START:",START)
print("[1/14] MODEL LOAD")
tv=tuple(int("".join(c for c in x if c.isdigit()) or 0) for x in transformers.__version__.split(".")[:2]);DT="dtype" if tv>=(4,56) else "torch_dtype"
torch.cuda.synchronize();t=time.perf_counter();tok=AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:tok.pad_token=tok.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,device_map={"":0},attn_implementation="sdpa",**{DT:torch.bfloat16});model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;H=model.config.hidden_size;PDT=next(model.parameters()).dtype;torch.cuda.synchronize()
if len(layers)!=TOTAL or H!=H_EXPECT or PDT!=torch.bfloat16:raise RuntimeError("Architecture/dtype mismatch")
print(f"OK | {MODEL_ID} | 28L | H={H} | {PDT} | {time.perf_counter()-t:.2f}s")
print("[2/14] WEIGHT SENTINEL AND FIXED RSS")
FP_T=[layers[0].self_attn.q_proj.weight,layers[8].self_attn.o_proj.weight,layers[19].mlp.down_proj.weight,layers[27].mlp.down_proj.weight,model.model.norm.weight,model.lm_head.weight]
@torch.inference_mode()
def fp():return tuple(float(x.sum(dtype=torch.float32)) for x in FP_T)
FP0=fp();base=[IVME*(ZIRVE*np.exp(-SONUM*L)*(1+SONUM*L)+TABAN)/(ZIRVE+TABAN) for L in range(END+1)]
scale=TARGET_RSS/np.sqrt(sum(x*x for x in base));RHO=[x*scale for x in base];RSS=np.sqrt(sum(x*x for x in RHO))
print("FP:",[f"{x:.4f}" for x in FP0]);print(f"RSS={RSS:.9f} | RHO0={RHO[0]*100:.3f}% | RHO25={RHO[25]*100:.3f}%")
def enc(x):
    s=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
    return tok(s,return_tensors="pt",add_special_tokens=False).to(DEVICE)
@torch.inference_mode()
def cap(x):
    o=model(**enc(x),use_cache=False,output_hidden_states=True,return_dict=True)
    return [o.hidden_states[L+1][0,-1].float().detach().clone() for L in range(TOTAL)]
@torch.inference_mode()
def gen(x,n=12):
    e=enc(x);m=e["input_ids"].shape[1];y=model.generate(**e,max_new_tokens=n,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    return tok.decode(y[0,m:],skip_special_tokens=True).strip()
print("[3/14] VANILLA LANGUAGE CAPABILITY GATE")
GATEPASS={};GATEROW=[]
for la in LANG:
    ok=0
    for it in ITEMS:
        x=it[la];o=gen(GATE[la].format(x=x));p=nt(o)==nt(x);ok+=p;GATEROW.append({"lang":la,"id":it["id"],"target":x,"output":o,"pass":p})
    GATEPASS[la]=ok/len(ITEMS);print(f"{la} | {ok}/{len(ITEMS)} = {GATEPASS[la]:.3f}")
print("[4/14] CACHE MULTILINGUAL ABSOLUTE STATES")
STATE={}
for la in LANG:
    STATE[la]={}
    for it in ITEMS:STATE[la][it["id"]]=[[*cap(f.format(x=it[la]))] for f in CTX[la]]
    print(la,"READY")
print("[5/14] WITHIN LANGUAGE TARGET CONTRAST FORGE")
RAW={}
for la in LANG:
    RAW[la]={}
    for it in ITEMS:
        iid=it["id"];RAW[la][iid]=[]
        refs=[r for r in ITEMS if r["id"]!=iid]
        for ci in range(5):
            RAW[la][iid].append([])
            for L in range(TOTAL):
                h=STATE[la][iid][ci][L]
                RAW[la][iid][ci].append(unit(torch.stack([unit(h-STATE[la][r["id"]][ci][L]) for r in refs]).mean(0)))
print("RAW READY")
print("[6/14] ITEM CENTERING AND FIVE CONTEXT CONSENSUS")
PACK={}
for la in LANG:
    PACK[la]={}
    for ci in range(5):
        for L in range(TOTAL):
            mu=torch.stack([RAW[la][it["id"]][ci][L] for it in ITEMS]).mean(0)
            for it in ITEMS:
                iid=it["id"];PACK[la].setdefault(iid,[[] for _ in range(TOTAL)])
                PACK[la][iid][L].append(unit(RAW[la][iid][ci][L]-mu))
    for it in ITEMS:
        iid=it["id"];PACK[la][iid]=[unit(torch.stack(PACK[la][iid][L]).mean(0)) for L in range(TOTAL)]
print("PACK READY")
print("[7/14] CROSS LANGUAGE GEOMETRY")
GEO={}
for L in REPORT:
    same=[];wrong=[]
    for it in ITEMS:
        iid=it["id"]
        for a in range(len(LANG)):
            for b in range(a+1,len(LANG)):
                same.append(cos(PACK[LANG[a]][iid][L],PACK[LANG[b]][iid][L]))
                wid=ITEMS[(ITEMS.index(it)+5)%len(ITEMS)]["id"];wrong.append(cos(PACK[LANG[a]][iid][L],PACK[LANG[b]][wid][L]))
    GEO[str(L)]={"same":float(np.mean(same)),"wrong":float(np.mean(wrong)),"margin":float(np.mean(same)-np.mean(wrong))}
    x=GEO[str(L)];print(f"L{L:02d} | SAME={x['same']:+.4f} | WRONG={x['wrong']:+.4f} | MARGIN={x['margin']:+.4f}")
print("[8/14] LEAVE ONE LANGUAGE OUT CONSENSUS")
LOO={}
for L in REPORT:
    vals=[]
    for hold in LANG:
        train=[x for x in LANG if x!=hold]
        for it in ITEMS:
            iid=it["id"];v=unit(torch.stack([PACK[x][iid][L] for x in train]).mean(0));vals.append(cos(v,PACK[hold][iid][L]))
    LOO[str(L)]={"mean":float(np.mean(vals)),"median":float(np.median(vals)),"positive":float(np.mean(np.array(vals)>0))}
    x=LOO[str(L)];print(f"L{L:02d} | LOO={x['mean']:+.4f} | MED={x['median']:+.4f} | POS={x['positive']:.3f}")
print("[9/14] CAUSAL ENGINE")
AUD={"calls":0,"max_dev":0.0};ACTIVE=set()
def install(v):
    hs=[]
    for L in range(END+1):
        rho=RHO[L]
        def hook(mod,args,out,L=L,rho=rho):
            y=out[0] if isinstance(out,tuple) else out;z=y[:,-1,:].float();d=v[L].to(z.device)*z.norm(dim=-1,keepdim=True)*rho
            rel=float((d.norm(dim=-1)/(z.norm(dim=-1)+EPS)).max());AUD["calls"]+=1;AUD["max_dev"]=max(AUD["max_dev"],abs(rel-rho))
            yy=y.clone();yy[:,-1,:]=(z+d).to(y.dtype)
            return (yy,)+out[1:] if isinstance(out,tuple) else yy
        h=layers[L].register_forward_hook(hook);hs.append(h);ACTIVE.add(id(h))
    return hs
@torch.inference_mode()
def logits(prompt,v=None):
    hs=install(v) if v is not None else []
    try:return model(**enc(prompt),use_cache=False,return_dict=True).logits[0,-1].float().detach().clone()
    finally:
        for h in hs:h.remove();ACTIVE.discard(id(h))
print("[10/14] CROSS LANGUAGE CAUSAL LOGIT READOUT")
ROWS=[]
for hold in LANG:
    train=[x for x in LANG if x!=hold]
    for it in ITEMS:
        iid=it["id"];target=it[hold];ids=tok.encode(target,add_special_tokens=False)
        if not ids:continue
        tid=ids[0];v=[unit(torch.stack([PACK[x][iid][L] for x in train]).mean(0)) for L in range(TOTAL)]
        n=logits(BLIND[hold]);s=logits(BLIND[hold],v);d=float(s[tid]-n[tid]);r0=int((n>n[tid]).sum())+1;r1=int((s>s[tid]).sum())+1
        capable=next(r["pass"] for r in GATEROW if r["lang"]==hold and r["id"]==iid)
        ROWS.append({"lang":hold,"id":iid,"target":target,"capable":capable,"token_id":tid,"delta":d,"rank_null":r0,"rank_write":r1,"improved":r1<r0})
    rr=[r for r in ROWS if r["lang"]==hold and r["capable"]]
    print(f"{hold} | ELIGIBLE={len(rr)} | DLOGIT={np.mean([r['delta'] for r in rr]) if rr else float('nan'):+.3f} | POS={np.mean([r['delta']>0 for r in rr]) if rr else float('nan'):.3f} | IMP={np.mean([r['improved'] for r in rr]) if rr else float('nan'):.3f}")
print("[11/14] CAPABILITY GATED GLOBAL")
EL=[r for r in ROWS if r["capable"]]
GLOBAL={"eligible":len(EL),"total":len(ROWS),"mean_delta":float(np.mean([r["delta"] for r in EL])) if EL else None,
"positive_delta":float(np.mean([r["delta"]>0 for r in EL])) if EL else None,"rank_improved":float(np.mean([r["improved"] for r in EL])) if EL else None,
"median_rank_null":float(np.median([r["rank_null"] for r in EL])) if EL else None,"median_rank_write":float(np.median([r["rank_write"] for r in EL])) if EL else None}
print(GLOBAL)
print("[12/14] LANGUAGE ATLAS")
ATLAS={}
for la in LANG:
    rr=[r for r in ROWS if r["lang"]==la and r["capable"]]
    ATLAS[la]={"gate":GATEPASS[la],"n":len(rr),"delta":float(np.mean([r["delta"] for r in rr])) if rr else None,"positive":float(np.mean([r["delta"]>0 for r in rr])) if rr else None,"improved":float(np.mean([r["improved"] for r in rr])) if rr else None}
    print(la,ATLAS[la])
print("[13/14] TOKENIZATION AUDIT")
for la in LANG:
    ns=[len(tok.encode(it[la],add_special_tokens=False)) for it in ITEMS]
    print(f"{la} | MEAN_TOK={np.mean(ns):.2f} | ONE_TOKEN={np.mean(np.array(ns)==1):.3f} | MAX={max(ns)}")
print("[14/14] INTEGRITY AND SEAL")
if fp()!=FP0 or model.training or any(p.requires_grad for p in model.parameters()):raise RuntimeError("Integrity failure")
if ACTIVE:raise RuntimeError("AkbasCore hook leak")
if abs(RSS-TARGET_RSS)>1e-9 or AUD["max_dev"]>1e-6:raise RuntimeError("Dose audit failure")
R={"schema":"akbascore.test291.v1","test":"TEST 291","start":START,"end":utc(),"model":MODEL_ID,
"question":"Does target-specific latent geometry transfer across languages, and does a latent compiled from four languages improve the held-out language target-token readout when vanilla capability is verified?",
"languages":LANG,"items":ITEMS,"capability_gate":GATEPASS,"gate_rows":GATEROW,"geometry":GEO,"leave_one_language_out":LOO,
"causal_rows":ROWS,"global":GLOBAL,"language_atlas":ATLAS,
"seasc":{"on_layers":[0,25],"off_layers":[26,27],"rss_definition":"sqrt(sum(rho_L^2))","rss":RSS,"rho0":RHO[0],"rho25":RHO[25]},
"integrity":{"training":False,"classifier":False,"selection":False,"weight_update":False,"injection_calls":AUD["calls"],"max_dose_deviation":AUD["max_dev"],"akbascore_hooks_remaining":len(ACTIVE),"fp_start":FP0,"fp_end":fp(),"result":"PASS"}}
sha=hashlib.sha256(canon(R)).hexdigest();run=f"T291-{datetime.now().strftime('%Y%m%d-%H%M%S')}";jp=ROOT/f"{run}.json";tp=ROOT/f"{run}.txt"
jp.write_bytes(json.dumps(R,ensure_ascii=True,sort_keys=True,indent=2).encode())
out=["="*108,"TEST 291 - CAPABILITY GATED CROSS LINGUAL LATENT TRANSFER","="*108]
for la in LANG:out.append(f"{la} GATE={GATEPASS[la]:.6f} N={ATLAS[la]['n']} DLOGIT={ATLAS[la]['delta']} POS={ATLAS[la]['positive']} IMP={ATLAS[la]['improved']}")
for L in REPORT:out.append(f"L{L:02d} SAME={GEO[str(L)]['same']:+.6f} WRONG={GEO[str(L)]['wrong']:+.6f} LOO={LOO[str(L)]['mean']:+.6f}")
out+=["",f"ELIGIBLE={GLOBAL['eligible']}/{GLOBAL['total']}",f"MEAN_DLOGIT={GLOBAL['mean_delta']}",f"POSITIVE={GLOBAL['positive_delta']}",f"RANK_IMPROVED={GLOBAL['rank_improved']}",f"RSS={RSS:.9f}",f"CALLS={AUD['calls']}",f"MAX_DOSE_DEV={AUD['max_dev']:.3e}","NO TRAINING - NO CLASSIFIER - NO SELECTION - WEIGHT INTEGRITY PASS",f"JSON: {jp}",f"TXT: {tp}",f"SHA: {sha}"]
tp.write_text("\n".join(out),encoding="utf-8")
print("="*108);print("TEST 291 COMPLETE")
print(f"ELIGIBLE={GLOBAL['eligible']}/{GLOBAL['total']} | DLOGIT={GLOBAL['mean_delta']} | POS={GLOBAL['positive_delta']} | IMP={GLOBAL['rank_improved']}")
print(f"RSS={RSS:.9f} | CALLS={AUD['calls']} | MAX_DOSE_DEV={AUD['max_dev']:.3e}")
print("NO TRAINING | NO CLASSIFIER | NO SELECTION | WEIGHT INTEGRITY PASS | AKBASCORE HOOKS=0")
print("JSON:",jp);print("TXT :",tp);print("SHA :",sha);print("="*108)
