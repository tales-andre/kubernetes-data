# ─────────────────────────────────────────────────────────────
#  K8s Cluster Auditor – versão otimizada
# ─────────────────────────────────────────────────────────────
import streamlit as st
import subprocess, os, re, yaml, json, io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =============== CONSTANTES ===============
OUTPUT_DIR        = "eks_data"
ALL_RESOURCES     = os.path.join(OUTPUT_DIR, "all_resources.yaml")
LOGS_DIR          = os.path.join(OUTPUT_DIR, "pod_logs")
ERROR_LOGS_FILE   = os.path.join(OUTPUT_DIR, "error_logs.csv")
CLUSTER_INFO_DUMP = os.path.join(OUTPUT_DIR, "cluster_info_dump.yaml")

RX_IPS   = [re.compile(r"10\.35\.\d+\.\d+"), re.compile(r"10-35[\d\.]*")]
REGION   = "sa-east-1"
RX_ERR   = re.compile(r"(error|exception|failed|fail|timeout)", re.I)
FMT_ERRS = [", ,", ",,", ", , ", ",,,", " , ,", ",, ,"]

KUBE_CTX: str | None = None

# =============== LOADER YAML seguro ===============
class SafeLoader(yaml.SafeLoader): ...
def _ignore(loader, tag, node):
    if isinstance(node, yaml.ScalarNode):   return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):  return loader.construct_mapping(node)
    return ""
SafeLoader.add_multi_constructor("", _ignore)

# ------------- util -------------
def kube_cmd(*args: str) -> list[str]:
    cmd = ["kubectl"]
    if KUBE_CTX:
        cmd += ["--context", KUBE_CTX]
    return cmd + list(args)

def run_kubectl(*args: str, **kw):
    return subprocess.check_output(kube_cmd(*args), **kw)

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

# =============== CACHE PESADO ===============
@st.cache_data(show_spinner=False, ttl="1h")
def cached_read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@st.cache_data(show_spinner=False, ttl="1h")
def cached_load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(yaml.load_all(f, Loader=SafeLoader))

# =============== COLETAS ====================
def collect_resources():
    ensure_dirs()
    with st.spinner("Dumpando recursos…"):
        with open(ALL_RESOURCES, "w", encoding="utf-8") as f:
            f.write("# Dump\n")
        res_ns = run_kubectl("api-resources", "--verbs=list", "--namespaced=true",
                             "-o", "name").decode().split()
        res_cl = run_kubectl("api-resources", "--verbs=list", "--namespaced=false",
                             "-o", "name").decode().split()
        for kind in sorted(set(res_ns + res_cl)):
            try:
                dump = run_kubectl("get", kind, "--all-namespaces", "-o", "yaml")
                with open(ALL_RESOURCES, "ab") as out:
                    out.write(b"\n# " + kind.encode() + b"\n")
                    out.write(dump)
            except subprocess.CalledProcessError:
                pass
    st.success("Recursos coletados.")

def collect_logs():
    ensure_dirs()
    nss = run_kubectl("get", "namespaces", "-o",
                      "jsonpath={.items[*].metadata.name}").decode().split()
    nss = [n for n in nss if n.lower() != "velero"]

    for ns in nss:
        pods = run_kubectl("get", "pods", "-n", ns,
                           "-o", "jsonpath={.items[*].metadata.name}").decode().split()
        for pod in pods:
            base = os.path.join(LOGS_DIR, f"{ns}__{pod}")
            try:
                open(base + ".log", "wb").write(
                    run_kubectl("logs", pod, "-n", ns, stderr=subprocess.STDOUT))
            except: pass
            # containers adicionais
            try:
                conts = run_kubectl("get", "pod", pod, "-n", ns,
                                    "-o", "jsonpath={.spec.containers[*].name}").decode().split()
                if len(conts) > 1:
                    for c in conts:
                        try:
                            open(base + f"__{c}.log", "wb").write(
                                run_kubectl("logs", pod, "-n", ns, "-c", c,
                                            stderr=subprocess.STDOUT))
                        except: pass
            except: pass
    st.success("Logs coletados.")

# =============== PARSE LOGS (paralelo) ===============
def _parse_file(path: str):
    out = []
    fname = os.path.basename(path)
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.strip(): continue
            lower = ln.lower()
            out.append({
                "file": fname,
                "line": ln.rstrip("\n"),
                "is_error": bool(RX_ERR.search(lower)),
                "timestamp": None,   # preenchido depois
            })
    return out

@st.cache_data(show_spinner=False, ttl="30m", max_entries=5)
def load_logs_df():
    paths = [os.path.join(LOGS_DIR, p) for p in os.listdir(LOGS_DIR)
             if p.endswith(".log")]
    rows = []
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as exe:
        fut = {exe.submit(_parse_file, p): p for p in paths}
        for f in as_completed(fut):
            rows.extend(f.result())
    df = pd.DataFrame(rows)
    # timestamp (expressivo – só calcula 1x)
    rx_ts = re.compile(r"(\\d{4}-\\d{2}-\\d{2}[T\\s]\\d{2}:\\d{2}:\\d{2})")
    def to_ts(s):
        m = rx_ts.search(s); 
        return datetime.fromisoformat(m.group(1).replace(" ", "T")) if m else None
    df["timestamp"] = df["line"].apply(to_ts)
    return df

# =============== ANÁLISE RECURSOS =========
def analyze_resources_fast():
    if not os.path.exists(ALL_RESOURCES):
        st.warning("Faça a coleta primeiro.")
        return
    raw = cached_read_file(ALL_RESOURCES)
    # IPs proibidos
    found = {m.group(0) for rx in RX_IPS for m in rx.finditer(raw)}
    st.write("IPs proibidos encontrados:" if found else "Sem IPs 10.35*/10-35*", found)
    # região
    if REGION in raw:
        st.error(f"Encontrado '{REGION}'")
    # erros de formatação
    fmt_bad = [l for l in raw.splitlines() if any(f in l for f in FMT_ERRS)]
    if fmt_bad:
        st.warning(f"{len(fmt_bad)} possíveis linhas mal formatadas.")
    # anomalias Ingress/Service
    docs = [d for d in cached_load_yaml(ALL_RESOURCES)
            if isinstance(d, dict) and d.get("metadata", {}).get("namespace") != "velero"]
    ingress = [d for d in docs if d.get("kind") == "Ingress"]
    svcs    = [d for d in docs if d.get("kind") == "Service"]
    ing_anom = detect_ingress_anomalies(ingress)
    svc_anom = detect_service_anomalies(svcs)
    st.write("Ingress anômalos:", len(ing_anom))
    st.write("Services anômalos:", len(svc_anom))

# =============== DASHBOARD LOGS ===========
def analyze_logs_fast():
    df = load_logs_df()
    if df.empty:
        st.info("Nenhum log carregado.")
        return
    st.write(f"Total linhas log: {len(df)}")
    st.write("Erros:", int(df["is_error"].sum()))
    st.bar_chart(df["is_error"].value_counts().rename(
        {True: "Erro", False: "Normal"}))
    # clusterização rápida
    errs = df[df["is_error"]]
    if not errs.empty and st.checkbox("Clusterizar erros (TF-IDF + KMeans)"):
        vec = TfidfVectorizer(max_features=500, stop_words="english")
        X = vec.fit_transform(errs["line"])
        k = 4 if len(errs) < 200 else 6
        errs["cluster"] = KMeans(n_clusters=k, n_init="auto", random_state=1).fit_predict(X)
        p2d = PCA(n_components=2, random_state=1).fit_transform(X.toarray())
        errs["x"], errs["y"] = p2d[:,0], p2d[:,1]
        st.plotly_chart(px.scatter(errs, x="x", y="y", color="cluster",
                                   hover_data=["line"]), use_container_width=True)
    # mostrar últimos erros
    st.subheader("Últimos erros")
    st.write(errs.tail(30)[["file", "line"]])

# ================= STREAMLIT UI ===========
def main():
    global KUBE_CTX
    st.set_page_config("K8s Auditor Fast", layout="wide")
    st.title("🚀 K8s Cluster Auditor – versão rápida")

    ctxs = subprocess.check_output(
        ["kubectl", "config", "get-contexts", "-o", "name"]).decode().split()
    if not ctxs:
        st.stop()
    KUBE_CTX = st.sidebar.selectbox("Contexto", ctxs, key="ctx")
    st.sidebar.markdown("---")
    action = st.sidebar.radio(
        "Ação", (
            "Coletar recursos",
            "Coletar logs",
            "Analizar recursos",
            "Analizar logs",
        )
    )
    if st.sidebar.button("Executar"):
        if action == "Coletar recursos":
            collect_resources()
        elif action == "Coletar logs":
            collect_logs()
        elif action == "Analizar recursos":
            analyze_resources_fast()
        elif action == "Analizar logs":
            analyze_logs_fast()

if __name__ == "__main__":
    ensure_dirs()
    main()
