# TEST460 (compact, development). Hypothesis: concatenated independently-forged modules are near-identical "clones" in key space, so the frozen
# reader selects records by surface frame instead of container identity (TEST452: JOINT F1 always answers a place U=0/16, F2 never L=0/16;
# CONTEXT discriminates in both). Solution under test: ISOLATED BATCHED READ = module-wise HOP executed as ONE batched forward per decode step
# (each batch row = one module cache, left-padded, module-local RoPE positions, masked padding) + deterministic aggregation.
# Arms: CONTEXT_HOP (visible text ref), JOINT_HOP (concat baseline), COFORGE_HOP (all records forged together; diagnostic), SEQ_MW (TEST452 path),
# BMW_K0 (batched, case modules), BMW_K6 (batched + 6 distractor modules of other entities). Engine (forge/codebook/K120-V128-OWN/install) = TEST452.
# Changed vs TEST452: new seed/entities, distractor modules, COFORGE arm, batched isolated reader, verdict logic. Frozen weights; no source at readout
# (COFORGE/CONTEXT are references only).
import os, re, json, time, random, hashlib
from collections import Counter

SEED = 460
N_ENT = 8
MAX_NEW = 32
FMT = "QUESTION:\n{q}\n\nANSWER:"
SEP = "\n\n"
STYLE = " Give only the answer on the first line; do not explain."
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
CORPUS = ["Jonas Weber carried the wooden crate across the quiet market square.", "Priya Nair wrote a long letter to her cousin in the morning.",
          "The small boat drifted slowly toward the rocky shore.", "Omar Haddad fixed the broken clock in the village school.",
          "A tired teacher closed the green door of the library.", "Sofia Rossi baked fresh bread for the harvest festival.",
          "The children watched the kites rising above the hill.", "Liam O'Connor sold his old bicycle to a neighbor.",
          "Heavy rain flooded the narrow street near the market.", "Nadia Petrova translated the ancient manuscript into French.",
          "The farmer counted the sheep before sunset.", "Hiro Sato opened a tiny bakery beside the river.",
          "An old dog slept under the kitchen table all afternoon.", "Carlos Mendes washed the windows of his grandmother's house.",
          "The museum guard locked the heavy gate at midnight.", "Fatima Zahra grew tomatoes in the backyard garden.",
          "The pilot announced a short delay because of fog.", "Anna Kowalski found a lost wallet on the bus.",
          "Snow covered the mountain village during the night.", "Ravi Kumar taught his brother how to play chess.",
          "The chef sharpened every knife before dinner service.", "Lucas Martin cleaned the roof of the barn after the storm.",
          "A young violinist practiced scales in the empty hall.", "Mei Lin delivered the package to the wrong address.",
          "Marek Novak placed the glass bottle beneath the wooden bench.", "Sara Ibrahim carried a red notebook into the quiet classroom.",
          "Noah Schmidt repaired the small radio beside the kitchen window.", "Yuki Mori left a paper envelope near the station entrance.",
          "Amira Hassan moved the ceramic bowl onto the upper shelf.", "Peter Novak opened the metal box behind the old theater.",
          "Lucia Costa placed the yellow scarf inside the travel bag.", "Daniel Kim carried a black umbrella through the central courtyard."]
FAM = {"F1": ("The {O} sits in container {T}.", "Container {T} is located at the {P}."),
       "F2": ("Container {T} holds the {O}.", "The {P} houses container {T}.")}
HOP1 = "Which container holds the {obj}? Give only the container ID on the first line; do not explain."
HOP2 = "Name the place (not a container) where container {cid} is located. If the records do not connect container {cid} to any place, answer UNKNOWN." + STYLE
MW1 = "Which container holds the {obj}? If this record does not say which container holds the {obj}, answer NONE." + STYLE
MW2 = "Where is container {cid} located according to this record? Give the place name. If this record does not say where container {cid} is, answer NONE." + STYLE
LINKED = ("AB", "BA", "AXB", "BXA")
UNLINKED = ("AX", "XA", "AXW", "WXA")
P_ADJ = ["cinder", "dove", "ermine", "flint", "heather", "iris", "lemon", "nutmeg", "onyx", "rosewood"]
P_NOUN = ["abbey", "bakery", "citadel", "dairy", "forum", "grotto", "hostel", "kennel", "rampart", "wharf"]
O_ADJ = ["dusty", "enameled", "frayed", "gilt", "mottled", "varnished"]
O_NOUN = ["anvil", "basin", "chime", "dagger", "easel", "flask"]
ID_RE = re.compile(r"\b([A-Z]{2}-\d{3})\b", re.I)
AMBIG = {"not", "no", "never", "nor", "or", "either", "maybe", "perhaps", "possibly", "probably", "might", "unclear", "but", "unsure"}

def norm(x):
    return re.sub(r"[^\w]+", " ", str(x).casefold()).strip()

def first(text):
    line = next((s.strip() for s in str(text).splitlines() if s.strip()), "")
    line = re.sub(r"^\W*(final answer|answer)\s*[:\-]\s*", "", line, flags=re.I).strip(" *`\"'")
    return re.split(r"(?<=[.!?])\s+", line)[0] if line else ""

def classify(text):
    fl = first(text); n = norm(fl); w = n.split(); ids = sorted({m.upper() for m in ID_RE.findall(fl)})
    if not w: return "empty", ids, fl
    if set(w) <= {"none", "unknown"}: return "none", ids, fl
    if (set(w) & AMBIG) or len(ids) > 1 or (set(w) & {"none", "unknown"}): return "ambiguous", ids, fl
    return "answer", ids, fl

def place_key(a):
    return re.sub(r"^(?:the|in the|at the|in|at)\s+", "", norm(a)).strip()

def stage1_parse(text):
    cls, ids, fl = classify(text)
    if cls == "answer": return ("id", ids[0]) if len(ids) == 1 else ("unparsed", None)
    return cls, None

def decide(parsed):
    if any(c in ("ambiguous", "unparsed", "empty") for c, _ in parsed): return None, "unresolved_stage1"
    ids = sorted({i for c, i in parsed if c == "id"})
    if not ids: return None, "no_id"
    return (None, "multi_id") if len(ids) > 1 else (ids[0], "single_id")

def aggregate(texts):
    cl = [classify(t) for t in texts]
    if any(c in ("ambiguous", "empty") for c, _, _ in cl): return "UNKNOWN", "unresolved_stage2"
    ans = [fl for c, _, fl in cl if c == "answer"]; keys = sorted({place_key(a) for a in ans})
    if not keys: return "UNKNOWN", "no_place"
    return ("UNKNOWN", "multi_place") if len(keys) > 1 else (ans[0], "single_place")

def judge(text, c, target):
    cls, ids, fl = classify(text); k = place_key(fl)
    if target == "UNKNOWN":
        if cls == "none": return "correct"
        return "false_P" if k == place_key(c["P"]) else "false_Q" if k == place_key(c["Q"]) else "other_wrong"
    if cls == "answer" and k == place_key(target): return "correct"
    if cls == "none": return "false_unknown"
    return "decoy_Q" if k == place_key(c["Q"]) else "other_wrong"

def make_entities():
    rng = random.Random(SEED); places = [f"{a} {n}" for a in P_ADJ for n in P_NOUN]; objs = [f"{a} {n}" for a in O_ADJ for n in O_NOUN]
    rng.shuffle(places); rng.shuffle(objs); ids = set(); out = []
    def nid():
        while True:
            x = "".join(rng.choice("BCDFGHJKLMNPRSTVWXZ") for _ in range(2)) + "-" + str(rng.randint(100, 999))
            if x not in ids: ids.add(x); return x
    for i in range(N_ENT):
        P = places.pop(0); Q = places.pop(next(j for j, q in enumerate(places) if not set(q.split()) & set(P.split())))
        out.append(dict(e=i, obj=objs[i], P=P, Q=Q, T=nid(), D=nid(), D2=nid(), fam="F1" if i % 2 == 0 else "F2"))
    return out

def make_cases(ents):
    cases = []
    for e in ents:
        A, L = FAM[e["fam"]]
        mods = dict(A=A.format(O=e["obj"], T=e["T"]), B=L.format(T=e["T"], P=e["P"]), X=L.format(T=e["D"], P=e["Q"]), W=L.format(T=e["D2"], P=e["P"]))
        for br in LINKED + UNLINKED:
            cases.append(dict(e, id=f"t460-{e['e']}:{br}", br=br, linked=br in LINKED, target=e["P"] if br in LINKED else "UNKNOWN",
                              order=[f"{e['e']}:{k}" for k in br], role={f"{e['e']}:{k}": k for k in br},
                              mods={f"{e['e']}:{k}": mods[k] for k in br}, source="\n".join(mods[k] for k in br)))
    return cases

def probe_expected(c, cid):
    if cid == c["T"]: return c["target"]
    if cid == c["D"]: return c["Q"] if "X" in c["br"] else "UNKNOWN"
    return "UNKNOWN"

# ---- protocols (reader(question, memory) -> generated text) ----
def hop(reader, c, mem):
    t1 = reader(HOP1.format(obj=c["obj"]), mem); m = ID_RE.search(first(t1)); cid = m.group(1).upper() if m else None
    t2 = reader(HOP2.format(cid=cid), mem) if cid else "UNKNOWN"
    out = judge(t2, c, c["target"])
    return dict(stage1=t1, id=cid, stage2=t2, outcome=out, strict=out == "correct" and cid == c["T"])

def mw_stage2(reader, c, mems, cid):
    texts = [reader(MW2.format(cid=cid), mems[m]) for m in c["order"]]; final, flag = aggregate(texts)
    return texts, final, flag

def mwhop(reader, c, mems):
    s1 = [reader(MW1.format(obj=c["obj"]), mems[m]) for m in c["order"]]; cid, flag = decide([stage1_parse(t) for t in s1])
    if cid is None: return dict(stage1=s1, id=None, flag=flag, stage2=None, final="UNKNOWN", outcome=judge("UNKNOWN", c, c["target"]), strict=False)
    s2, final, pflag = mw_stage2(reader, c, mems, cid); out = judge(final, c, c["target"]); f = flag + "/" + pflag
    want = "single_id/single_place" if c["linked"] else "single_id/no_place"
    return dict(stage1=s1, id=cid, flag=f, stage2=s2, final=final, outcome=out, strict=out == "correct" and cid == c["T"] and f == want)

def mwprobe(reader, c, mems, cid):
    s2, final, flag = mw_stage2(reader, c, mems, cid); exp = probe_expected(c, cid); out = judge(final, c, exp)
    want = "no_place" if exp == "UNKNOWN" else "single_place"
    return dict(id=cid, expected=exp, stage2=s2, final=final, flag=flag, strict=out == "correct" and flag == want)

def distractors(c, cases):  # 6 modules (A,B,X) of the next two entities: different objects, IDs and places
    out = []
    for k in (1, 2):
        e = (c["e"] + k) % N_ENT; d = next(x for x in cases if x["e"] == e and x["br"] == "AXB")
        out += [m for m in d["order"]]
    return out

def mw_set(reader, c, mems, order, batch=None):  # module-wise HOP over an arbitrary module list; batch(q, [mem]) -> [text]
    rd = (lambda q, ms: batch(q, ms)) if batch else (lambda q, ms: [reader(q, m) for m in ms])
    s1 = rd(MW1.format(obj=c["obj"]), [mems[m] for m in order]); cid, flag = decide([stage1_parse(t) for t in s1])
    if cid is None: return dict(stage1=s1, id=None, flag=flag, stage2=None, final="UNKNOWN", outcome=judge("UNKNOWN", c, c["target"]), strict=False)
    s2 = rd(MW2.format(cid=cid), [mems[m] for m in order]); final, pflag = aggregate(s2); out = judge(final, c, c["target"]); f = flag + "/" + pflag
    want = "single_id/single_place" if c["linked"] else "single_id/no_place"
    return dict(stage1=s1, id=cid, flag=f, stage2=s2, final=final, outcome=out, strict=out == "correct" and cid == c["T"] and f == want)

def run_case(reader, batch, c, M, dmods):
    r = dict(id=c["id"], e=c["e"], br=c["br"], fam=c["fam"], linked=c["linked"], target=c["target"])
    r["CONTEXT_HOP"] = hop(reader, c, ("ctx", c["source"])); r["JOINT_HOP"] = hop(reader, c, M["joint"]); r["COFORGE_HOP"] = hop(reader, c, M["coforge"])
    r["SEQ_MW"] = mw_set(reader, c, M["mod"], c["order"])
    for arm, order in (("BMW_K0", c["order"]), ("BMW_K6", sorted(c["order"] + dmods, key=lambda m: hashlib.md5((c["id"] + m).encode()).hexdigest()))):
        try: r[arm] = mw_set(reader, c, M["mod"], order, batch)
        except Exception as ex: r[arm] = dict(error=f"{type(ex).__name__}: {ex}", outcome="batch_error", strict=False, flag="error")
    a = [x == y for x, y in zip(r["SEQ_MW"]["stage1"], r["BMW_K0"].get("stage1") or [])]
    r["batch_agree_stage1"] = (sum(a), len(c["order"]))
    return r

def wilson_lo(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d

ARMS = ("CONTEXT_HOP", "JOINT_HOP", "COFORGE_HOP", "SEQ_MW", "BMW_K0", "BMW_K6")
def lu(rows, arm, key):
    lk = [r for r in rows if r["linked"]]; ul = [r for r in rows if not r["linked"]]
    L = sum(bool(r[arm][key] if key == "strict" else r[arm]["outcome"] == "correct") for r in lk); U = sum(bool(r[arm][key] if key == "strict" else r[arm]["outcome"] == "correct") for r in ul)
    return L, len(lk), U, len(ul)

def summarize(rows, errors, n_cases):
    S = {}; fams = ("F1", "F2")
    for arm in ARMS:
        for fam in (None,) + fams:
            rr = [r for r in rows if fam is None or r["fam"] == fam]; L, nl, U, nu = lu(rr, arm, "strict")
            S[f"{arm}{'' if fam is None else '/' + fam}"] = dict(L=f"{L}/{nl}", U=f"{U}/{nu}", J=round(L / max(nl, 1) + U / max(nu, 1) - 1, 3), L_lo=round(wilson_lo(L, nl), 3), U_lo=round(wilson_lo(U, nu), 3))
    S["errors_by_arm"] = {a: dict(Counter(r[a]["outcome"] for r in rows if r[a]["outcome"] != "correct")) for a in ARMS}
    ag = [r["batch_agree_stage1"] for r in rows]; S["batch_stage1_agreement"] = f"{sum(a for a, _ in ag)}/{sum(n for _, n in ag)}"
    S["batch_errors"] = sum(r["BMW_K0"]["outcome"] == "batch_error" or r["BMW_K6"]["outcome"] == "batch_error" for r in rows)
    S["BMW_K6_flags"] = dict(Counter(r["BMW_K6"]["flag"] for r in rows)); S["entities_all8_BMW_K0"] = sum(all(r["BMW_K0"]["strict"] for r in rows if r["e"] == e) for e in {r["e"] for r in rows})
    J = lambda k: S[k]["J"]; frac = lambda k, x: (lambda a, b: a / max(b, 1))(*map(int, S[k][x].split("/")))
    mech = "INDEPENDENT_ENCODING_INDIVIDUATION_FAILURE" if (J("COFORGE_HOP") - J("JOINT_HOP") >= 0.40 and J("COFORGE_HOP/F2") >= 0.40 and J("JOINT_HOP/F2") <= 0.20) else \
           ("NOT_INDEPENDENT_ENCODING" if J("COFORGE_HOP") - J("JOINT_HOP") <= 0.10 else "MECHANISM_INCONCLUSIVE")
    agree = sum(a for a, _ in ag) / max(sum(n for _, n in ag), 1)
    k0 = all(frac(f"BMW_K0{f}", x) >= t for f, t in (("", 0.80), ("/F1", 0.70), ("/F2", 0.70)) for x in ("L", "U"))
    k6 = frac("BMW_K6", "L") >= 0.75 and frac("BMW_K6", "U") >= 0.80
    if errors or len(rows) != n_cases: arch = "TECHNICAL_INVALID"
    elif S["batch_errors"] or agree < 0.95: arch = "BATCH_PATH_MISMATCH"
    elif k0 and k6: arch = "ISOLATED_BATCHED_READ_READY"
    elif k0: arch = "ISOLATION_WORKS_SCALING_FAILS"
    else: arch = "ISOLATED_READ_FAILS"
    S["mechanism_verdict"] = mech; S["architecture_verdict"] = arch
    return S

def selftest():
    for t, want in [("NONE", "none"), ("Unknown.", "none"), ("", "empty"), ("NONE, but KR-417", "ambiguous"), ("KR-417 or MT-238", "ambiguous"),
                    ("Not KR-417", "ambiguous"), ("The russet quay", "answer"), ("russet quay. The ewer sits there.", "answer")]:
        assert classify(t)[0] == want, (t, classify(t))
    assert decide([("none", None), ("id", "KR-417")]) == ("KR-417", "single_id") and decide([("id", "KR-417"), ("unparsed", None)])[1] == "unresolved_stage1"
    assert aggregate(["NONE", "The russet quay."]) == ("The russet quay.", "single_place") and aggregate(["russet quay", "ochre loft"])[1] == "multi_place"
    cases = make_cases(make_entities())
    assert len(cases) == 8 * N_ENT and sum(c["linked"] for c in cases) == 4 * N_ENT
    for c in cases:
        d = distractors(c, cases); assert len(d) == 6 and not set(d) & set(c["order"])
        assert all(c["T"] not in x for x in d) and all(r not in p for r in c["mods"].values() for p in (HOP1.format(obj=c["obj"]), MW1.format(obj=c["obj"]), MW2.format(cid=c["T"]), HOP2.format(cid=c["T"])))
    return cases

CASES = selftest(); MODS = {m: t for c in CASES for m, t in c["mods"].items()}
print(f"[TEST460] CPU self-test PASS: {len(CASES)} cases ({N_ENT} entities x 8 branches)")

# ------------------------------------------------ GPU part ------------------------------------------------
def run_gpu():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    assert torch.cuda.is_available(), "A100 GPU runtime required"
    torch.manual_seed(SEED); DEV = torch.device("cuda"); torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    try: model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map={"": 0}, attn_implementation="sdpa", dtype=torch.bfloat16).eval()
    except TypeError: model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map={"": 0}, attn_implementation="sdpa", torch_dtype=torch.bfloat16).eval()
    for p in model.parameters(): p.requires_grad_(False)
    cfg = model.config; layers = model.model.layers; NL = len(layers); NKV = cfg.num_key_value_heads
    HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads; KVD = NKV * HD
    PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ge = model.generation_config.eos_token_id; EOS = sorted({tok.eos_token_id, *(ge if isinstance(ge, (list, tuple)) else [ge])} - {None})
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    def sentinel():
        h = hashlib.sha256()
        for t in (layers[0].self_attn.q_proj.weight, layers[NL // 2].mlp.down_proj.weight, model.lm_head.weight):
            h.update(t.detach().reshape(-1)[:4096].float().cpu().numpy().tobytes())
        return h.hexdigest()
    SENT0 = sentinel()

    @torch.inference_mode()
    def kv_from_ids(ids):
        o = model(input_ids=torch.tensor([ids], device=DEV), output_hidden_states=True, use_cache=False, return_dict=True); K = []; V = []
        for L in range(NL):
            z = layers[L].input_layernorm(o.hidden_states[L][0]); a = layers[L].self_attn
            K.append(a.k_proj(z).contiguous()); V.append(a.v_proj(z).contiguous())
        return K, V

    def forge(s): return kv_from_ids([PAD] + enc(s + SEP))

    @torch.inference_mode()
    def install(K, V):
        T = K[0].shape[0]; cos, sin = model.model.rotary_emb(K[0][None], torch.arange(T, device=DEV)[None]); KK = []; VV = []
        for L in range(NL):
            k = K[L].reshape(1, T, NKV, HD).transpose(1, 2); v = V[L].reshape(1, T, NKV, HD).transpose(1, 2)
            KK.append(apply_rotary_pos_emb(k, k, cos, sin, unsqueeze_dim=1)[1].contiguous()); VV.append(v.contiguous())
        return tuple(KK), tuple(VV), T

    def mkcache(kv):
        try: c = DynamicCache(config=cfg)
        except TypeError: c = DynamicCache()
        for L in range(NL): c.update(kv[0][L].clone(), kv[1][L].clone(), L)
        assert c.get_seq_length() == kv[2]
        return c

    print("[TEST460] building codebook from fixed neutral corpus ...")
    with torch.inference_mode():
        CO = [forge(s) for s in CORPUS]; CB = []
        for L in range(NL):
            e = {}
            for j, n in enumerate(("K", "V")):
                R = torch.cat([c[j][L][1:] for c in CO]).float().view(-1, NKV, HD); MU = []; B = []
                for h in range(NKV):
                    X = R[:, h]; mu = X.mean(0); _, _, Vh = torch.linalg.svd(X - mu, full_matrices=False); m = min(128, Vh.shape[0]); bb = Vh[:m].T.contiguous()
                    if m < 128: bb = torch.nn.functional.pad(bb, (0, 128 - m))
                    MU.append(mu); B.append(bb)
                e[n] = (torch.stack(MU), torch.stack(B))
            CB.append(e)
        del CO

    @torch.inference_mode()
    def packet(source):  # K: 120 PCA coefficients, V: 128, source-own slot0 kept exactly; returns decoded pre-RoPE K,V
        K, V = forge(source); out = {}
        for n, X, d in (("K", K, 120), ("V", V, 128)):
            rows = []
            for L in range(NL):
                mu, B = CB[L][n]; coeff = torch.einsum("thi,hid->thd", X[L][1:].float().view(-1, NKV, HD) - mu, B[:, :, :d]).to(torch.bfloat16)
                content = (mu + torch.einsum("thd,hid->thi", coeff.float(), B[:, :, :d])).reshape(-1, KVD).to(torch.bfloat16)
                rows.append(torch.cat([X[L][:1], content]))
            out[n] = rows
        return out["K"], out["V"]

    STATS = Counter()
    @torch.inference_mode()
    def reader(q, mem):
        qids = enc(FMT.format(q=q))
        if isinstance(mem, tuple) and mem and mem[0] == "ctx": pre = [PAD] + enc(mem[1] + SEP); cache = None
        else: pre = [PAD] * mem[2]; cache = mkcache(mem)
        ids = torch.tensor([pre + qids], device=DEV)
        y = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache, max_new_tokens=MAX_NEW, do_sample=False,
                           repetition_penalty=1.0, use_cache=True, pad_token_id=PAD, eos_token_id=EOS)
        new = y[0, ids.shape[1]:].tolist(); STATS["generations"] += 1
        if cache is not None and cache.get_seq_length() != len(pre) + len(qids) + len(new) - 1: raise RuntimeError("cache length mismatch")
        return tok.decode(new, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def batch(q, kvs):  # ISOLATED BATCHED READ: one forward per decode step for all modules; each row sees only its own module
        qids = enc(FMT.format(q=q)); B = len(kvs); Tm = max(kv[2] for kv in kvs); nq = len(qids)
        try: cache = DynamicCache(config=cfg)
        except TypeError: cache = DynamicCache()
        for L in range(NL):
            Ks, Vs = [], []
            for kv in kvs:
                k, v, p = kv[0][L], kv[1][L], Tm - kv[2]
                if p: z = k.new_zeros(1, NKV, p, HD); k = torch.cat([z, k], 2); v = torch.cat([z.clone(), v], 2)
                Ks.append(k); Vs.append(v)
            cache.update(torch.cat(Ks).clone(), torch.cat(Vs).clone(), L)
        mask = torch.zeros(B, Tm + nq, dtype=torch.long, device=DEV)
        for b, kv in enumerate(kvs): mask[b, Tm - kv[2]:] = 1
        pos = torch.tensor([kv[2] for kv in kvs], device=DEV)[:, None] + torch.arange(nq, device=DEV)[None]
        ids = torch.tensor([qids] * B, device=DEV); out = [[] for _ in range(B)]; done = [False] * B
        for _ in range(MAX_NEW):
            o = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True, return_dict=True)
            nxt = o.logits[:, -1].float().argmax(-1).tolist()
            for b, t in enumerate(nxt):
                if not done[b]:
                    if t in EOS: done[b] = True
                    else: out[b].append(t)
            if all(done): break
            ids = torch.tensor([[t] for t in nxt], device=DEV); pos = pos[:, -1:] + 1; mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=DEV)], 1)
        STATS["batched_calls"] += 1
        return [tok.decode(x, skip_special_tokens=True).strip() for x in out]

    def finite(kv): return all(bool(torch.isfinite(t).all()) for t in list(kv[0]) + list(kv[1]))
    def fp(kv): return float(sum(t.float().abs().sum() for t in kv[0][:2]))

    rows = []; errors = []; BANK = {}; t0 = time.time()
    def mod_kv(m):
        if m not in BANK: BANK[m] = packet(MODS[m])
        return BANK[m]
    for i, c in enumerate(CASES, 1):
        try:
            dmods = distractors(c, CASES); allm = c["order"] + dmods
            joint = install([torch.cat([mod_kv(m)[0][L] for m in c["order"]]) for L in range(NL)], [torch.cat([mod_kv(m)[1][L] for m in c["order"]]) for L in range(NL)])
            coforge = install(*packet(c["source"]))  # DIAGNOSTIC ONLY: all records forged together (breaks modularity)
            mod = {m: install(*mod_kv(m)) for m in allm}
            allkv = [joint, coforge] + list(mod.values())
            if not all(finite(kv) for kv in allkv): raise RuntimeError("non-finite cache")
            before = [fp(kv) for kv in allkv]
            row = run_case(reader, batch, c, dict(joint=joint, coforge=coforge, mod=mod), dmods)
            if [fp(kv) for kv in allkv] != before: raise RuntimeError("installed cache changed during reading")
            rows.append(row)
            print(f"[{i:02d}/{len(CASES)}] {c['id']:<14} {c['fam']} tgt={c['target'][:16]:<16} " + " ".join(f"{a}={int(bool(row[a]['strict']))}" for a in ARMS) + f" agree={row['batch_agree_stage1']}", flush=True)
        except Exception as ex:
            errors.append(dict(case=c["id"], error=f"{type(ex).__name__}: {ex}")); print("ERROR", errors[-1], flush=True)
    frozen = sentinel() == SENT0 and not model.training and all(not p.requires_grad for p in model.parameters())
    if not frozen: errors.append(dict(case=None, error="weight sentinel / frozen check failed"))
    S = summarize(rows, errors, len(CASES)); S["generations"] = STATS["generations"]; S["minutes"] = round((time.time() - t0) / 60, 1); S["frozen_weights_ok"] = frozen
    path = "/content/test460_results.json" if os.path.isdir("/content") else "test460_results.json"
    json.dump(dict(summary=S, rows=rows, errors=errors), open(path, "w"), indent=1, ensure_ascii=False)
    print("=" * 90)
    for k, v in S.items(): print(f"{k}: {v}")
    print("=" * 90)
    print(f"TEST460 VERDICT: mechanism={S['mechanism_verdict']} | architecture={S['architecture_verdict']}  (results: {path})")
    print("Scope: development panel (new entities, F1/F2 only). COFORGE is a diagnostic that breaks modularity. A READY verdict licenses the frozen F3/F4 held-out test, not a generalization claim.")

if os.environ.get("T460_CPU_ONLY") != "1":
    run_gpu()
