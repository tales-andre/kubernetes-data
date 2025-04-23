# ─────────────────────────────────────────────────────────────
#  K8s Cluster Auditor & Log Analyzer  –  app.py (Streamlit)
# ─────────────────────────────────────────────────────────────

import streamlit as st
import subprocess
import os
import re
import yaml
import pandas as pd
import numpy as np  # (pode ser útil em extensões futuras)
import json
from datetime import datetime
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ========================================
# ============ CONFIGURAÇÕES =============
# ========================================

OUTPUT_DIR = "eks_data"
ALL_RESOURCES_FILE = os.path.join(OUTPUT_DIR, "all_resources.yaml")
CLUSTER_INFO_DUMP = os.path.join(OUTPUT_DIR, "cluster_info_dump.yaml")
LOGS_DIR = os.path.join(OUTPUT_DIR, "pod_logs")
ERROR_LOGS_FILE = os.path.join(OUTPUT_DIR, "error_logs.csv")

# IPs/região proibidos
PROHIBITED_IP_PATTERNS = [r"10\\.35\\.\\d+\\.\\d+", r"10-35"]
PROHIBITED_REGION = "sa-east-1"

FORMAT_ERRORS = [", ,", ",,", ", , ", ",,,", " , ,", ",, ,"]

ERROR_KEYWORDS = ["ERROR", "Error", "Exception", "Failed", "Fail"]
KUBE_CONTEXT: str | None = None

# ========================================
# =========== YAML LOADER SEGURO =========
# ========================================

class SafeIgnoringLoader(yaml.SafeLoader):
    """Ignora qualquer tag YAML desconhecida."""

def _ignore_unknown(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return ""

SafeIgnoringLoader.add_multi_constructor("", _ignore_unknown)

def load_yaml_all(stream):
    """Carrega todos os docs YAML ignorando tags estranhas."""
    return list(yaml.load_all(stream, Loader=SafeIgnoringLoader))

# ========================================
# ======= kubectl helper / contexto ======
# ========================================

def kube_cmd(*args: str) -> list[str]:
    cmd = ["kubectl"]
    if KUBE_CONTEXT:
        cmd.extend(["--context", KUBE_CONTEXT])
    cmd.extend(args)
    return cmd

def run_kubectl(*args: str, **kwargs):
    return subprocess.check_output(kube_cmd(*args), **kwargs)

# ========================================
# ===== Funções utilitárias e coleta =====
# ========================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

def get_available_contexts() -> list[str]:
    try:
        ctx = subprocess.check_output(
            ["kubectl", "config", "get-contexts", "-o", "name"]
        )
        return ctx.decode().split()
    except Exception as e:
        st.error(f"Erro ao listar contexts: {e}")
        return []

# ---------- coleta de recursos ----------
def collect_all_resources():
    ensure_output_dir()
    with st.spinner("Coletando recursos do cluster…"):
        try:
            with open(ALL_RESOURCES_FILE, "w", encoding="utf-8") as f:
                f.write("# Dump de TODOS os recursos\n")

            res_ns = run_kubectl(
                "api-resources", "--verbs=list", "--namespaced=true",
                "-o", "name"
            ).decode().split()
            res_cl = run_kubectl(
                "api-resources", "--verbs=list", "--namespaced=false",
                "-o", "name"
            ).decode().split()

            for kind in sorted(set(res_ns + res_cl)):
                with open(ALL_RESOURCES_FILE, "a", encoding="utf-8") as f:
                    f.write(f"\n# Recurso: {kind}\n")
                try:
                    dump = run_kubectl("get", kind, "--all-namespaces", "-o", "yaml")
                    with open(ALL_RESOURCES_FILE, "ab") as bf:
                        bf.write(dump)
                except subprocess.CalledProcessError:
                    pass
                except Exception as e:
                    st.warning(f"Falha ao coletar {kind}: {e}")

            st.success(f"Coleta concluída: {ALL_RESOURCES_FILE}")
        except Exception as e:
            st.error(f"Erro na coleta: {e}")

# ---------- cluster-info dump ----------
def cluster_info_dump():
    ensure_output_dir()
    with st.spinner("Executando cluster-info dump…"):
        try:
            out = run_kubectl("cluster-info", "dump", "--all-namespaces")
            with open(CLUSTER_INFO_DUMP, "wb") as f:
                f.write(out)
            st.success(f"Dump salvo em {CLUSTER_INFO_DUMP}")
        except Exception as e:
            st.error(f"Falha no dump: {e}")

# ---------- coleta de logs ----------
def collect_pod_logs():
    ensure_output_dir()
    with st.spinner("Coletando logs dos pods…"):
        try:
            namespaces = run_kubectl(
                "get", "namespaces", "-o", "jsonpath={.items[*].metadata.name}"
            ).decode().split()
        except Exception as e:
            st.error(f"Erro ao listar namespaces: {e}")
            return

        namespaces = [ns for ns in namespaces if ns.lower() != "velero"]

        for ns in namespaces:
            try:
                pods = run_kubectl(
                    "get", "pods", "-n", ns,
                    "-o", "jsonpath={.items[*].metadata.name}"
                ).decode().split()
            except Exception as e:
                st.warning(f"Erro em {ns}: {e}")
                continue

            for pod in pods:
                # container principal
                try:
                    p_path = os.path.join(LOGS_DIR, f"{ns}__{pod}.log")
                    pdata = run_kubectl("logs", pod, "-n", ns, stderr=subprocess.STDOUT)
                    with open(p_path, "wb") as f:
                        f.write(pdata)
                except Exception:
                    pass

                # containers adicionais
                try:
                    conts = run_kubectl(
                        "get", "pod", pod, "-n", ns,
                        "-o", "jsonpath={.spec.containers[*].name}"
                    ).decode().split()
                    if len(conts) > 1:
                        for c in conts:
                            try:
                                c_path = os.path.join(LOGS_DIR, f"{ns}__{pod}__{c}.log")
                                cdata = run_kubectl(
                                    "logs", pod, "-n", ns, "-c", c,
                                    stderr=subprocess.STDOUT
                                )
                                with open(c_path, "wb") as f:
                                    f.write(cdata)
                            except Exception:
                                pass
                except Exception:
                    pass
        st.success(f"Logs salvos em {LOGS_DIR}")

# ========================================
# ======= Anomalias de recursos ==========
# ========================================
def detect_ingress_anomalies(ingress_list):
    if not ingress_list:
        return []
    rows = []
    for ing in ingress_list:
        meta = ing.get("metadata", {})
        lb_host = None
        lb = ing.get("status", {}).get("loadBalancer", {}).get("ingress", [])
        if lb:
            lb_host = lb[0].get("hostname") or lb[0].get("ip")
        if not lb_host:
            lb_host = meta.get("annotations", {}).get(
                "service.beta.kubernetes.io/load-balancer-name", "desconhecido"
            )
        rows.append({
            "name": meta.get("name"),
            "namespace": meta.get("namespace"),
            "lb_host": lb_host,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    main_lb = df["lb_host"].value_counts().idxmax()
    return df[df["lb_host"] != main_lb].to_dict("records")

def detect_service_anomalies(svcs):
    if not svcs:
        return []
    rows = []
    for svc in svcs:
        meta = svc.get("metadata", {})
        rows.append({
            "name": meta.get("name"),
            "namespace": meta.get("namespace"),
            "svc_type": svc.get("spec", {}).get("type", "ClusterIP"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    common = df["svc_type"].value_counts().idxmax()
    return df[df["svc_type"] != common].to_dict("records")

def analyze_k8s_resources_yaml(fpath):
    if not os.path.exists(fpath):
        return {}, "Arquivo não encontrado."
    with open(fpath, "r", encoding="utf-8") as f:
        docs = load_yaml_all(f)
    docs = [d for d in docs if isinstance(d, dict)]
    ingress = [
        d for d in docs
        if d.get("kind") == "Ingress"
        and d.get("metadata", {}).get("namespace") != "velero"
    ]
    services = [
        d for d in docs
        if d.get("kind") == "Service"
        and d.get("metadata", {}).get("namespace") != "velero"
    ]
    return {
        "Ingress": detect_ingress_anomalies(ingress),
        "Service": detect_service_anomalies(services),
    }, None

# ========================================
# =========== Parse & logs utils =========
# ========================================
def extract_timestamp(line):
    m = re.search(r"(\\d{4}-\\d{2}-\\d{2}[T\\s]\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?Z?)", line)
    if not m:
        return None
    ts = m.group(1)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            pass
    return None

def extract_log_features(line):
    return {
        "line": line,
        "timestamp": extract_timestamp(line),
        "is_error": any(k in line for k in ERROR_KEYWORDS),
        "uppercase_count": sum(c.isupper() for c in line),
        "possible_stacktrace": bool(re.search(r"\\bat\\b|\\bline\\b|File ", line)),
        "line_len": len(line),
    }

def load_all_lines_from_logs():
    rows = []
    for root, _, files in os.walk(LOGS_DIR):
        for fn in files:
            if fn.endswith(".log"):
                p = os.path.join(root, fn)
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for ln in f:
                            feat = extract_log_features(ln.rstrip("\\n"))
                            feat["file"] = fn
                            rows.append(feat)
                except Exception:
                    pass
    return rows

def store_error_logs():
    df = pd.DataFrame(load_all_lines_from_logs())
    if df.empty or df[df["is_error"]].empty:
        st.info("Nenhum log de erro encontrado.")
        return
    df[df["is_error"]].to_csv(ERROR_LOGS_FILE, index=False)
    st.success(f"Logs de erro salvos em {ERROR_LOGS_FILE}")

# ========================================
# == análise detalhada de logs por pod ===
# ========================================
def get_namespace_resources(ns):
    res = {"Ingress": [], "Service": []}
    if not os.path.exists(ALL_RESOURCES_FILE):
        return res
    with open(ALL_RESOURCES_FILE, "r", encoding="utf-8") as f:
        docs = load_yaml_all(f)
    for d in docs:
        if isinstance(d, dict) and d.get("metadata", {}).get("namespace") == ns:
            k = d.get("kind")
            if k in res:
                res[k].append(d.get("metadata", {}).get("name"))
    return res

def analyze_pod_errors():
    df_logs = pd.DataFrame(load_all_lines_from_logs())
    if df_logs.empty:
        st.info("Nenhum log carregado.")
        return
    df_err = df_logs[df_logs["is_error"]].copy()
    if df_err.empty:
        st.success("Nenhum erro nos pods.")
        return

    def split_ns(fn):
        parts = fn.split("__")
        return parts[0], parts[1] if len(parts) > 1 else ("unknown", fn)

    df_err["namespace"], df_err["pod"] = zip(*df_err["file"].apply(split_ns))
    df_err["error_summary"] = df_err["line"].str[:60]

    grouped = (
        df_err.groupby(["namespace", "pod", "error_summary"])
        .size().reset_index(name="count")
    )
    grouped["recursos"] = grouped["namespace"].apply(lambda n: str(get_namespace_resources(n)))

    st.dataframe(grouped)

# ========================================
# ========= Status geral dos pods ========
# ========================================
def get_pod_statuses():
    try:
        out = run_kubectl("get", "pods", "--all-namespaces", "-o", "json").decode()
        items = json.loads(out)["items"]
        rows = []
        for it in items:
            meta = it["metadata"]
            stt = it.get("status", {})
            rows.append({
                "namespace": meta["namespace"],
                "pod": meta["name"],
                "phase": stt.get("phase"),
                "reason": stt.get("reason", ""),
                "message": stt.get("message", ""),
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Erro ao obter status: {e}")
        return pd.DataFrame()

# ========================================
# ======== Clusterização de erros ========
# ========================================
def cluster_error_logs():
    df = pd.DataFrame(load_all_lines_from_logs())
    if df.empty:
        st.info("Sem logs para clusterizar.")
        return
    errs = df[df["is_error"]]
    if errs.empty:
        st.info("Sem erros para clusterizar.")
        return

    vec = TfidfVectorizer(stop_words="english", max_features=1000)
    X = vec.fit_transform(errs["line"])
    k = 5
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    errs["cluster"] = km.labels_

    st.bar_chart(errs["cluster"].value_counts().sort_index())

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    errs["x"], errs["y"] = coords[:, 0], coords[:, 1]
    fig = px.scatter(
        errs, x="x", y="y", color="cluster", hover_data=["line"],
        title="Clusters de erros (PCA 2D)"
    )
    st.plotly_chart(fig, use_container_width=True)

# ========================================
# ========== Pipeline principal ==========
# ========================================
def analyze_resources_and_logs():
    st.header("📄 Análise de Recursos")

    if not os.path.exists(ALL_RESOURCES_FILE):
        st.warning("Execute a coleta de recursos primeiro.")
    else:
        with open(ALL_RESOURCES_FILE, "r", encoding="utf-8") as f:
            content = f.read()

        # IPs/região proibidos
        ips = {ip for pat in PROHIBITED_IP_PATTERNS for ip in re.findall(pat, content)}
        if ips:
            st.error(f"IPs proibidos encontrados: {ips}")
        else:
            st.success("Nenhum IP proibido encontrado.")

        if PROHIBITED_REGION in content:
            st.error(f"Região '{PROHIBITED_REGION}' detectada!")
        else:
            st.success("Região proibida não encontrada.")

        # formato
        fmt_lines = [
            l for fe in FORMAT_ERRORS for l in content.splitlines() if fe in l
        ]
        if fmt_lines:
            st.warning(f"Erros de formatação: {len(fmt_lines)} linhas.")
        else:
            st.success("Formato YAML OK.")

        anomalies, _ = analyze_k8s_resources_yaml(ALL_RESOURCES_FILE)
        for kind, lst in anomalies.items():
            if lst:
                st.warning(f"{kind}: {len(lst)} anomalia(s).")
            else:
                st.success(f"{kind}: sem anomalias.")

    # ---------------- Logs ----------------
    st.header("📑 Análise de Logs")
    if not os.path.exists(LOGS_DIR):
        st.warning("Colete os logs primeiro.")
        return

    df_logs = pd.DataFrame(load_all_lines_from_logs())
    if df_logs.empty:
        st.info("Nenhum log encontrado.")
        return
    st.write(f"{len(df_logs)} linhas carregadas.")
    st.write("Erros:", int(df_logs["is_error"].sum()))

    # distribuição erros vs normais
    fig_bar = px.bar(
        df_logs["is_error"].value_counts().rename({True: "Erro", False: "Normal"}),
        labels={"value": "Linhas", "index": "Tipo"},
        title="Erros vs Normais"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # evolução temporal
    if df_logs["timestamp"].notna().any():
        df_ts = df_logs.dropna(subset=["timestamp"]).sort_values("timestamp")
        df_ts["cum"] = df_ts["is_error"].cumsum()
        st.line_chart(df_ts.set_index("timestamp")["cum"])

    st.subheader("Detalhamento por pod")
    analyze_pod_errors()

    st.subheader("Status dos pods")
    df_status = get_pod_statuses()
    if not df_status.empty:
        st.dataframe(df_status)

    st.subheader("Clusterização de erros")
    cluster_error_logs()

# ========================================
# ===============  STREAMLIT  ============
# ========================================
def main():
    global KUBE_CONTEXT

    st.set_page_config("K8s Cluster Auditor", layout="wide")
    st.title("🔍 Kubernetes Cluster Auditor & Log Analyzer")

    contexts = get_available_contexts()
    if not contexts:
        st.stop()

    idx = 0
    if "selected_ctx" in st.session_state:
        try:
            idx = contexts.index(st.session_state["selected_ctx"])
        except ValueError:
            pass

    KUBE_CONTEXT = st.sidebar.selectbox(
        "Contexto Kubernetes", contexts, index=idx
    )
    st.session_state["selected_ctx"] = KUBE_CONTEXT
    st.sidebar.write(f"Contexto ativo: `{KUBE_CONTEXT}`")

    st.sidebar.markdown("---")
    action = st.sidebar.radio(
        "Ação",
        (
            "Coletar recursos (YAML)",
            "Executar cluster-info dump",
            "Coletar logs dos pods",
            "Analisar recursos e logs",
            "Armazenar logs de erro",
        ),
    )

    if action == "Coletar recursos (YAML)":
        if st.button("Iniciar coleta"):
            collect_all_resources()
    elif action == "Executar cluster-info dump":
        if st.button("Executar"):
            cluster_info_dump()
    elif action == "Coletar logs dos pods":
        if st.button("Coletar"):
            collect_pod_logs()
    elif action == "Analisar recursos e logs":
        if st.button("Analisar"):
            analyze_resources_and_logs()
    elif action == "Armazenar logs de erro":
        if st.button("Armazenar"):
            store_error_logs()

if __name__ == "__main__":
    main()
