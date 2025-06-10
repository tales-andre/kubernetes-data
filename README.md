# Kubernetes Data Tools

Este repositório contém utilitários para coleta e análise de dados de clusters EKS.

## Scripts

### `app2.py`
Interface web (Streamlit) para coleta de recursos e logs do cluster, além de visualização rápida de erros.

### `eks_chatbot.py`
Chatbot simples em linha de comando para consultas sobre erros comuns do EKS. Ele varre arquivos em `eks_data/pod_logs` e procura mensagens de erro relevantes.

Uso:
```bash
python eks_chatbot.py
```
Digite a pergunta quando solicitado. O script retorna causas e soluções conhecidas para o erro encontrado.
