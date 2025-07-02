import os
import re
import pandas as pd

try:
    from transformers import pipeline
except Exception:  # noqa: E722 - handle missing frameworks gracefully
    pipeline = None

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:  # noqa: E722 - any ImportError
    HAS_TORCH = False

LOGS_DIR = os.path.join("eks_data", "pod_logs")

ERROR_HELP = {
    "imagepullbackoff": "Verifique se a imagem existe e suas credenciais de registry.",
    "crashloopbackoff": "O container reiniciou diversas vezes. Inspecione logs do container para detalhes.",
    "oomkilled": "O processo foi finalizado por falta de memória. Avalie aumentar os recursos da aplicação.",
}


def load_logs_df():
    rows = []
    if not os.path.isdir(LOGS_DIR):
        return pd.DataFrame(rows)
    for fname in os.listdir(LOGS_DIR):
        if not fname.endswith(".log"):
            continue
        path = os.path.join(LOGS_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append({"file": fname, "line": line})
    return pd.DataFrame(rows)


def main():
    print("Carregando logs...")
    df = load_logs_df()
    if df.empty:
        print("Nenhum log encontrado em", LOGS_DIR)
        return
    corpus = "\n".join(df["line"].tolist())[-100000:]
    print(f"{len(df)} linhas carregadas.")
    qa = None
    if pipeline is not None and HAS_TORCH:
        print("Carregando modelo de linguagem (distilbert)...")
        try:
            qa = pipeline(
                "question-answering",
                model="distilbert-base-uncased-distilled-squad",
            )
        except Exception:
            qa = None
            print(
                "Não foi possível carregar o modelo. Continuando apenas com dicas básicas."
            )
    else:
        print(
            "Bibliotecas de machine learning não disponíveis. Continuando apenas com dicas básicas."
        )
    print("Chatbot pronto. Digite 'sair' para finalizar.")
    while True:
        try:
            question = input("Você: ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"sair", "exit", "quit"}:
            break
        lower_q = question.lower()
        for key, advice in ERROR_HELP.items():
            if key in lower_q:
                print("Bot:", advice)
                break
        else:
            if qa is not None:
                try:
                    ans = qa(question=question, context=corpus)
                    print("Bot:", ans.get("answer"))
                except Exception:
                    print("Bot: Não consegui gerar uma resposta agora.")
            else:
                print(
                    "Bot: Não consigo responder perguntas complexas sem o modelo de linguagem."
                )
    print("Encerrado.")


if __name__ == "__main__":
    main()
