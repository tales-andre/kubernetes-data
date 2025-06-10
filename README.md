# K8s Cluster Auditor

Este projeto contém scripts para coletar e analisar recursos de um cluster Kubernetes (EKS) e um chatbot de apoio para consulta de logs.

## Pré-requisitos

- Python 3.10 ou superior
- `kubectl` configurado com acesso ao cluster
- Permissões para listar recursos e logs

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

### Auditoria e coleta

Execute a aplicação Streamlit para coletar recursos e logs ou visualizar análises rápidas:

```bash
streamlit run app2.py
```

Selecione a ação desejada na barra lateral e clique em **Executar**.

### Chatbot de troubleshooting

Após coletar os logs (`Coletar logs` na aplicação), é possível conversar com o chatbot:

```bash
python eks_chatbot.py
```

Digite sua pergunta e o bot utilizará um modelo de linguagem para buscar respostas nos logs e em um pequeno dicionário de erros comuns.

**Observação:** o primeiro uso pode baixar os pesos do modelo da biblioteca `transformers`.
