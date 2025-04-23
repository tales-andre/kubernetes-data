import streamlit as st
import subprocess
import os
import re
import yaml
import pandas as pd
import numpy as np
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

# padrões de IPs proibidos: qualquer IP começando com 10.35. e qualquer ocorrência de "10-35"
PROHIBITED_IP_PATTERNS = [r"10\.35\.\d+\.\d+", r"10-35"]
PROHIBITED_REGION = "sa-east-1"

FORMAT_ERRORS = [", ,", ",,", ", , ", ",,,", " , ,", ",, ,"]

ERROR_KEYWORDS = ["ERROR", "Error", "Exception", "Failed", "Fail"]
KUBE_CONTEXT: str | None = None


# ────────────────────────────────────────────────────────────────────────────────
# Helper functions to work with kubectl taking the active context into account
# ────────────────────────────────────────────────────────────────────────────────

def kube_cmd(*args: str) -> list[str]:
    """
    Assemble a kubectl command, injecting --context if the
    global KUBE_CONTEXT variable is set.
    """
    cmd = ["kubectl"]
    if KUBE_CONTEXT:
        cmd.extend(["--context", KUBE_CONTEXT])
    cmd.extend(args)
    return cmd


def run_kubectl(*args: str, **kwargs):
    """Shortcut for subprocess.check_output(kubectl …) respecting context."""
    return subprocess.check_output(kube_cmd(*args), **kwargs)

# ========================================
# ============ FUNÇÕES UTILITY ===========
# ========================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def get_available_contexts() -> list[str]:
    """Return the list of contexts configured in the local kube-config."""
    try:
        contexts = subprocess.check_output(
            ["kubectl", "config", "get-contexts", "-o", "name"]
        )
        return contexts.decode().split()
    except Exception as e:
        st.error(f"Erro ao listar contexts: {e}")
        return []

# ========================================
# ============ COLETA DE DADOS ===========
# ========================================

def collect_all_resources():
    ensure_output_dir()
    with st.spinner("Coletando recursos do cluster…"):
        try:
            with open(ALL_RESOURCES_FILE, "w", encoding="utf-8") as outf:
                outf.write("# Dump de TODOS os recursos\n")

            resources_ns = run_kubectl(
                "api-resources", "--verbs=list", "--namespaced=true", "-o", "name"
            ).decode().split()
            resources_non_ns = run_kubectl(
                "api-resources", "--verbs=list", "--namespaced=false", "-o", "name"
            ).decode().split()

            all_resources = sorted(set(resources_ns + resources_non_ns))
            for resource in all_resources:
                with open(ALL_RESOURCES_FILE, "a", encoding="utf-8") as outf:
                    outf.write(f"\n# Recurso: {resource}\n")
                try:
                    result = run_kubectl(
                        "get", resource, "--all-namespaces", "-o", "yaml"
                    )
                    with open(ALL_RESOURCES_FILE, "ab") as outf_bin:
                        outf_bin.write(result)
                except subprocess.CalledProcessError:
                    pass
                except Exception as e:
                    st.warning(f"Falha ao coletar {resource}: {e}")

            st.success(f"Coleta concluída. Arquivo gerado: {ALL_RESOURCES_FILE}")
        except Exception as e:
            st.error(f"Erro ao coletar recursos: {e}")

def cluster_info_dump():
    ensure_output_dir()
    with st.spinner("Executando cluster-info dump…"):
        try:
            result = run_kubectl("cluster-info", "dump", "--all-namespaces")
            with open(CLUSTER_INFO_DUMP, "wb") as f:
                f.write(result)
            st.success(f"Arquivo gerado: {CLUSTER_INFO_DUMP}")
        except Exception as e:
            st.error(f"Não foi possível executar cluster-info dump: {e}")

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

        # ignora namespace velero
        namespaces = [ns for ns in namespaces if ns.lower() != "velero"]

        for namespace in namespaces:
            try:
                pods = run_kubectl(
                    "get",
                    "pods",
                    "-n",
                    namespace,
                    "-o", "jsonpath={.items[*].metadata.name}",
                ).decode().split()
            except Exception as e:
                st.warning(f"Erro ao listar pods no namespace {namespace}: {e}")
                continue

            for pod in pods:
                # Logs do pod em si
                try:
                    log_file = os.path.join(LOGS_DIR, f"{namespace}__{pod}.log")
                    logs = run_kubectl("logs", pod, "-n", namespace, stderr=subprocess.STDOUT)
                    with open(log_file, "wb") as lf:
                        lf.write(logs)
                except Exception:
                    pass

                # Logs de containers adicionais (se houver)
                try:
                    containers = run_kubectl(
                        "get",
                        "pod",
                        pod,
                        "-n",
                        namespace,
                        "-o", "jsonpath={.spec.containers[*].name}",
                    ).decode().split()
                    if len(containers) > 1:
                        for container in containers:
                            try:
                                container_log_file = os.path.join(
                                    LOGS_DIR, f"{namespace}__{pod}__{container}.log"
                                )
                                logs_cont = run_kubectl(
                                    "logs",
                                    pod,
                                    "-n",
                                    namespace,
                                    "-c",
                                    container,
                                    stderr=subprocess.STDOUT,
                                )
                                with open(container_log_file, "wb") as clf:
                                    clf.write(logs_cont)
                            except Exception:
                                pass
                except Exception:
                    pass
        st.success(f"Logs coletados em: {LOGS_DIR}")
# ========================================
# ============ ANÁLISE DE RESOURCES ======
# ========================================

def detect_ingress_anomalies(ingress_list):
    if not ingress_list:
        return []
    info = []
    for ing in ingress_list:
        metadata = ing.get("metadata", {})
        name = metadata.get("name", "unknown")
        namespace = metadata.get("namespace", "default")
        lb_host = None
        status = ing.get("status", {})
        lb = status.get("loadBalancer", {})
        lb_ingress = lb.get("ingress", [])
        if lb_ingress:
            lb_host = lb_ingress[0].get("hostname") or lb_ingress[0].get("ip")
        annotations = metadata.get("annotations", {})
        if not lb_host:
            lb_host = annotations.get("service.beta.kubernetes.io/load-balancer-name", "desconhecido")
        info.append({"name": name, "namespace": namespace, "lb_host": lb_host})
    df_ing = pd.DataFrame(info)
    if df_ing.empty:
        return []
    counts = df_ing["lb_host"].value_counts()
    anomalies = []
    if len(counts) > 1:
        total_ingress = df_ing.shape[0]
        main_lb = counts.index[0]
        for lb_val, lb_count in counts.items():
            if lb_val != main_lb and lb_count < total_ingress * 0.05:
                outliers = df_ing[df_ing["lb_host"] == lb_val]
                anomalies.extend(outliers.to_dict("records"))
    return anomalies

def detect_service_anomalies(svc_list):
    if not svc_list:
        return []
    info = []
    for svc in svc_list:
        metadata = svc.get("metadata", {})
        spec = svc.get("spec", {})
        name = metadata.get("name", "unknown")
        namespace = metadata.get("namespace", "default")
        svc_type = spec.get("type", "ClusterIP")
        info.append({"name": name, "namespace": namespace, "svc_type": svc_type})
    df_svc = pd.DataFrame(info)
    if df_svc.empty:
        return []
    counts = df_svc["svc_type"].value_counts()
    anomalies = []
    if len(counts) > 1:
        total_svcs = df_svc.shape[0]
        main_type = counts.index[0]
        for tp_val, tp_count in counts.items():
            if tp_val != main_type and tp_count < total_svcs * 0.1:
                outliers = df_svc[df_svc["svc_type"] == tp_val]
                anomalies.extend(outliers.to_dict("records"))
    return anomalies

def analyze_k8s_resources_yaml(file_path):
    if not os.path.exists(file_path):
        return {}, "Arquivo de recursos não existe."
    with open(file_path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    docs = [d for d in docs if d is not None and isinstance(d, dict)]
    # ignora resources do namespace velero
    ingress_list = [resource for resource in docs if resource.get("kind", "") == "Ingress" and resource.get("metadata", {}).get("namespace") != "velero"]
    service_list = [resource for resource in docs if resource.get("kind", "") == "Service" and resource.get("metadata", {}).get("namespace") != "velero"]
    ingress_anomalies = detect_ingress_anomalies(ingress_list)
    svc_anomalies = detect_service_anomalies(service_list)
    results = {"Ingress": ingress_anomalies, "Service": svc_anomalies}
    return results, None

# ============================================
# ======= PARSE E ANÁLISE DE LOGS ============
# ============================================

def extract_timestamp(line):
    match = re.search(r"(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?)", line)
    if match:
        ts_str = match.group(1)
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"]:
            try:
                return datetime.strptime(ts_str, fmt)
            except:
                pass
    return None

def extract_log_features(line):
    is_error = any(kw in line for kw in ERROR_KEYWORDS)
    uppercase_count = sum(1 for c in line if c.isupper())
    possible_stacktrace = bool(re.search(r"\bat\b|\bline\b|File ", line))
    line_len = len(line)
    timestamp = extract_timestamp(line)
    return {
        "line": line,
        "timestamp": timestamp,
        "is_error": is_error,
        "uppercase_count": uppercase_count,
        "possible_stacktrace": possible_stacktrace,
        "line_len": line_len
    }

def load_all_lines_from_logs():
    all_rows = []
    for root, dirs, files in os.walk(LOGS_DIR):
        for fname in files:
            if fname.endswith(".log"):
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8") as lf:
                        lines = lf.readlines()
                    for line in lines:
                        line = line.strip()
                        feats = extract_log_features(line)
                        feats["file"] = fname
                        all_rows.append(feats)
                except Exception:
                    pass
    return all_rows

def store_error_logs():
    all_lines = load_all_lines_from_logs()
    if not all_lines:
        st.info("Nenhum log encontrado para armazenamento.")
        return
    df_logs = pd.DataFrame(all_lines)
    error_logs = df_logs[df_logs["is_error"] == True]
    if not error_logs.empty:
        error_logs.to_csv(ERROR_LOGS_FILE, index=False)
        st.success(f"Logs de erro armazenados em: {ERROR_LOGS_FILE}")
    else:
        st.info("Nenhum log de erro identificado para armazenamento.")

# ================================================
# ======= ANÁLISE DETALHADA DOS ERROS DOS PODS =========
# ================================================

def get_namespace_resources(namespace):
    resources = {"Ingress": [], "Service": []}
    if not os.path.exists(ALL_RESOURCES_FILE):
        return resources
    with open(ALL_RESOURCES_FILE, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    for doc in docs:
        if doc and isinstance(doc, dict):
            meta = doc.get("metadata", {})
            ns = meta.get("namespace", "default")
            if ns == namespace:
                kind = doc.get("kind", "")
                if kind in resources:
                    resources[kind].append(meta.get("name", "unknown"))
    return resources

def analyze_pod_errors():
    all_lines = load_all_lines_from_logs()
    if not all_lines:
        st.info("Nenhum log encontrado para análise de erros dos pods.")
        return
    df_logs = pd.DataFrame(all_lines)
    df_errors = df_logs[df_logs["is_error"] == True].copy()
    if df_errors.empty:
        st.info("Nenhum erro encontrado nos logs dos pods.")
        return

    def extract_namespace_pod(fname):
        parts = fname.split("__")
        if len(parts) >= 2:
            return parts[0], parts[1]
        else:
            return "unknown", fname

    df_errors["namespace"], df_errors["pod"] = zip(*df_errors["file"].apply(extract_namespace_pod))
    df_errors["error_summary"] = df_errors["line"].apply(lambda x: x[:50].strip())
    df_errors["auth_issue"] = df_errors["line"].apply(lambda x: "sim" if ("auth" in x.lower() or "denied" in x.lower() or "autenticação" in x.lower()) else "não")
    df_grouped = df_errors.groupby(["namespace", "pod", "error_summary", "auth_issue"]).size().reset_index(name="error_count")
    def resources_for_namespace(ns):
        res = get_namespace_resources(ns)
        ingress = res.get("Ingress", [])
        svc = res.get("Service", [])
        return f"Ingress: {', '.join(ingress) if ingress else 'N/A'}; Services: {', '.join(svc) if svc else 'N/A'}"
    df_grouped["recursos"] = df_grouped["namespace"].apply(resources_for_namespace)
    st.subheader("Erros dos Pods")
    st.dataframe(df_grouped)
    df_error_group = df_errors.groupby("error_summary").agg(
        pods=("pod", lambda x: ", ".join(set(x))),
        total_erros=("line", "count")
    ).reset_index()
    st.subheader("Agrupamento de Erros Semelhantes")
    st.dataframe(df_error_group)

def get_pod_statuses():
    try:
        output = run_kubectl("get", "pods", "--all-namespaces", "-o", "json").decode()
        data = json.loads(output)
        items = data.get("items", [])
        records = []
        for item in items:
            metadata = item.get("metadata", {})
            status = item.get("status", {})
            namespace = metadata.get("namespace", "default")
            pod_name = metadata.get("name", "unknown")
            phase = status.get("phase", "Unknown")
            reason = status.get("reason", "")
            message = ""
            conditions = status.get("conditions", [])
            for condition in conditions:
                if condition.get("status") == "False":
                    reason = condition.get("reason", reason)
                    message = condition.get("message", "")
                    break
            container_statuses = status.get("containerStatuses", [])
            waiting_reasons = []
            for cs in container_statuses:
                state = cs.get("state", {})
                if "waiting" in state:
                    waiting_reasons.append(state["waiting"].get("reason", ""))
                    if not message and state["waiting"].get("message"):
                        message = state["waiting"].get("message")
                elif "terminated" in state:
                    waiting_reasons.append(state["terminated"].get("reason", ""))
                    if not message and state["terminated"].get("message"):
                        message = state["terminated"].get("message")
                last_state = cs.get("lastState", {})
                if "waiting" in last_state:
                    if not message and last_state["waiting"].get("message"):
                        message = last_state["waiting"].get("message")
            waiting_reasons = ", ".join(waiting_reasons)
            records.append(
                {
                    "namespace": namespace,
                    "pod": pod_name,
                    "phase": phase,
                    "reason": reason,
                    "message": message,
                    "waiting_reasons": waiting_reasons,
                }
            )
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Erro ao obter status dos pods: {e}")
        return pd.DataFrame()

def get_pod_description(namespace, pod):
    try:
        output = run_kubectl("describe", "pod", pod, "-n", namespace, stderr=subprocess.STDOUT).decode()
        return output
    except Exception as e:
        return f"Erro ao descrever o pod: {e}"

# ========================================
# ========== CLUSTERIZAÇÃO DE ERROS ==========
# ========================================

def cluster_error_logs():
    """
    Filtra os logs de erro, realiza uma vetorização TF-IDF dos textos e utiliza KMeans para clusterizá-los.
    Em seguida, usa PCA para reduzir a dimensionalidade e exibe gráficos interativos que auxiliam na identificação de padrões.
    """
    all_lines = load_all_lines_from_logs()
    if not all_lines:
        st.info("Nenhum log disponível para análise de erros.")
        return
    df_logs = pd.DataFrame(all_lines)
    df_errors = df_logs[df_logs["is_error"] == True].copy()
    if df_errors.empty:
        st.info("Nenhum erro encontrado para clusterização.")
        return

    error_texts = df_errors["line"].tolist()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(error_texts)
    n_clusters = 5  # Número de clusters (pode ser ajustado ou determinado automaticamente)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    df_errors["cluster"] = labels

    st.subheader("Distribuição dos Clusters de Erros")
    cluster_counts = df_errors["cluster"].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    pca = PCA(n_components=2, random_state=42)
    X_reduced = pca.fit_transform(X.toarray())
    df_errors["pca1"] = X_reduced[:,0]
    df_errors["pca2"] = X_reduced[:,1]
    fig = px.scatter(df_errors, x="pca1", y="pca2", color="cluster", hover_data=["line"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Exemplos de Erros por Cluster")
    for cluster in range(n_clusters):
        st.markdown(f"**Cluster {cluster} (Total: {cluster_counts[cluster]})**")
        sample_texts = df_errors[df_errors["cluster"] == cluster]["line"].head(3).tolist()
        for txt in sample_texts:
            st.write(txt)

# ========================================
# ========== ANÁLISE PRINCIPAL ==========
# ========================================

def analyze_resources_and_logs():
    st.subheader("Análise de Recursos (YAML)")
    if not os.path.exists(ALL_RESOURCES_FILE):
        st.warning("Arquivo de recursos não existe. Execute a coleta primeiro.")
    else:
        with open(ALL_RESOURCES_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        ip_found = re.findall(r"10\.35\.\d+\.\d+", content)
        if "10.35." in content:
            ip_found.append("10.35.* (match genérico)")
        region_found = re.findall(r"sa-east-1", content)
        if ip_found:
            st.error(f"IPs proibidos encontrados: {set(ip_found)}")
        else:
            st.success("Nenhuma menção a 10.35. encontrada no YAML de recursos.")
        if region_found:
            st.error("Região 'sa-east-1' encontrada no YAML de recursos!")
        else:
            st.success("Nenhuma menção à região 'sa-east-1' encontrada no YAML.")
        format_issues = []
        for fe in FORMAT_ERRORS:
            lines = [line for line in content.split("\n") if fe in line]
            if lines:
                format_issues.extend(lines)
        if format_issues:
            st.error("Foram encontradas possíveis falhas de formatação no YAML.")
        else:
            st.success("Formato do YAML aparentemente OK.")
        anomalies_dict, err_msg = analyze_k8s_resources_yaml(ALL_RESOURCES_FILE)
        if err_msg:
            st.warning(err_msg)
        else:
            st.subheader("Anomalias de Configuração")
            ing_anoms = anomalies_dict.get("Ingress", [])
            svc_anoms = anomalies_dict.get("Service", [])
            if ing_anoms:
                st.warning(f"{len(ing_anoms)} anomalia(s) em Ingress.")
            else:
                st.success("Nenhuma anomalia em Ingress.")
            if svc_anoms:
                st.warning(f"{len(svc_anoms)} anomalia(s) em Services.")
            else:
                st.success("Nenhuma anomalia em Services.")
            with open(ALL_RESOURCES_FILE, "r", encoding="utf-8") as f:
                all_docs = list(yaml.safe_load_all(f))
            ing_list = [d for d in all_docs if d and d.get("kind") == "Ingress"]
            data_ing = []
            for ing in ing_list:
                meta = ing.get("metadata", {})
                name = meta.get("name", "unknown")
                namespace = meta.get("namespace", "default")
                lb_host = None
                status = ing.get("status", {}).get("loadBalancer", {})
                lb_ingr = status.get("ingress", [])
                if lb_ingr:
                    lb_host = lb_ingr[0].get("hostname") or lb_ingr[0].get("ip")
                annotations = meta.get("annotations", {})
                if not lb_host:
                    lb_host = annotations.get("service.beta.kubernetes.io/load-balancer-name", "desconhecido")
                data_ing.append({"name": name, "namespace": namespace, "lb_host": lb_host})
            df_all_ing = pd.DataFrame(data_ing)
            if not df_all_ing.empty:
                lb_counts = df_all_ing["lb_host"].value_counts().reset_index()
                lb_counts.columns = ["lb_host", "count"]
                fig_ing_dist = px.bar(lb_counts, x="lb_host", y="count", title="Distribuição de LB nos Ingress")
                st.plotly_chart(fig_ing_dist, use_container_width=True)
    st.subheader("Análise de Logs (Pods)")
    if not os.path.exists(LOGS_DIR):
        st.warning("Diretório de logs não existe. Execute a coleta de logs primeiro.")
        return
    all_lines = load_all_lines_from_logs()
    if not all_lines:
        st.info("Nenhum log disponível para análise.")
        return
    df_logs = pd.DataFrame(all_lines)
    st.write(f"Foram carregadas {len(df_logs)} linhas de log.")
    df_logs["possible_stacktrace_int"] = df_logs["possible_stacktrace"].astype(int)
    df_logs["is_error_int"] = df_logs["is_error"].astype(int)
    error_lines = df_logs[df_logs["is_error"] == True]
    if not error_lines.empty:
        st.error(f"{len(error_lines)} linhas de log contêm ERROS!")
        st.dataframe(error_lines.tail(10))
    else:
        st.success("Nenhum erro encontrado nos logs.")
    st.write("### Distribuição de Linhas por Arquivo")
    log_counts_by_file = df_logs["file"].value_counts().reset_index()
    log_counts_by_file.columns = ["file", "count"]
    fig_logs_by_file = px.bar(log_counts_by_file, x="file", y="count", title="Linhas de Log por Arquivo")
    st.plotly_chart(fig_logs_by_file, use_container_width=True)
    st.write("### Distribuição Geral: Erros vs Normais")
    counts_global = df_logs["is_error"].value_counts()
    normal_count = counts_global.get(False, 0)
    error_count = counts_global.get(True, 0)
    df_plot_bar = pd.DataFrame({
        "Tipo": ["Linha Normal", "Linha de Erro"],
        "Quantidade": [normal_count, error_count]
    })
    fig_bar = px.bar(df_plot_bar, x="Tipo", y="Quantidade", title="Logs: Erros vs Normais")
    st.plotly_chart(fig_bar, use_container_width=True)
    if df_logs["timestamp"].notnull().any():
        df_logs_ts = df_logs.dropna(subset=["timestamp"]).copy().sort_values("timestamp")
        df_logs_ts["error_cumulative"] = df_logs_ts["is_error"].cumsum()
        fig_line = px.line(df_logs_ts, x="timestamp", y="error_cumulative",
                           title="Evolução de Erros ao Longo do Tempo",
                           labels={"error_cumulative": "Erros (acumulados)", "timestamp": "Tempo"})
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Não foi possível plotar a evolução no tempo (falta timestamp).")
    st.subheader("Detalhamento dos Erros dos Pods")
    analyze_pod_errors()
    st.subheader("Status dos Pods")
    df_pods_status = get_pod_statuses()
    if not df_pods_status.empty:
        st.dataframe(df_pods_status)
    else:
        st.info("Não foi possível obter o status dos pods.")
    st.subheader("Pods em Pending: Motivo")
    if not df_pods_status.empty:
        df_pending = df_pods_status[df_pods_status["phase"] == "Pending"]
        if not df_pending.empty:
            st.dataframe(df_pending)
        else:
            st.success("Nenhum pod em Pending.")
    else:
        st.info("Não foi possível obter status dos pods.")
    st.subheader("Clusterização dos Erros")
    cluster_error_logs()
    st.success("Análise de logs concluída!")

# ========================================
# ============== STREAMLIT APP ===========
# ========================================

def main():
    global KUBE_CONTEXT  # pylint: disable=global-statement

    st.set_page_config(page_title="K8s Cluster Auditor", layout="wide")
    st.title("🔍 Kubernetes Cluster Auditor & Log Analyzer")

    # ── Context selector ───────────────────────────────────────────────────────
    contexts = get_available_contexts()
    if not contexts:
        st.stop()

    default_index = 0
    if "selected_ctx" in st.session_state:
        try:
            default_index = contexts.index(st.session_state["selected_ctx"])
        except ValueError:
            pass
    selected_ctx = st.sidebar.selectbox(
        "Selecionar contexto Kubernetes", contexts, index=default_index
    )
    st.session_state["selected_ctx"] = selected_ctx
    KUBE_CONTEXT = selected_ctx
    st.sidebar.markdown(f"**Contexto ativo:** `{KUBE_CONTEXT}`")

    # ── Action menu ───────────────────────────────────────────────────────────
    st.sidebar.markdown("---")
    acao = st.sidebar.radio(
        "Menu de Ações",
        (
            "Coletar recursos (YAML)",
            "Executar cluster-info dump",
            "Coletar logs dos pods",
            "Analisar recursos e logs",
            "Armazenar logs de erro",
        ),
    )

    if acao == "Coletar recursos (YAML)":
        if st.button("Iniciar Coleta de Recursos"):
            collect_all_resources()
    elif acao == "Executar cluster-info dump":
        if st.button("Executar cluster-info dump"):
            cluster_info_dump()
    elif acao == "Coletar logs dos pods":
        if st.button("Coletar Logs dos Pods"):
            collect_pod_logs()
    elif acao == "Analisar recursos e logs":
        if st.button("Analisar"):
            analyze_resources_and_logs()
    elif acao == "Armazenar logs de erro":
        if st.button("Armazenar Logs de Erro"):
            store_error_logs()


if __name__ == "__main__":
    main()