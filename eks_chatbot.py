import os
import re

# Diretório com logs processados (app2.py) ou logs brutos
LOG_DIR = os.path.join('eks_data', 'pod_logs')

# Dicionário de erros comuns do EKS
ERROR_KB = {
    'CrashLoopBackOff': {
        'cause': 'O contêiner falhou repetidamente ao iniciar.',
        'solution': 'Verifique os logs do pod e a configuração da imagem.'
    },
    'ImagePullBackOff': {
        'cause': 'Falha ao puxar a imagem do contêiner.',
        'solution': 'Confirme se a imagem existe e se as credenciais estão corretas.'
    },
    'ErrImagePull': {
        'cause': 'Erro ao baixar a imagem especificada.',
        'solution': 'Verifique se o registro está acessível e se o nome da imagem está correto.'
    },
    'FailedScheduling': {
        'cause': 'O agendador não conseguiu alocar o pod em nenhum nó.',
        'solution': 'Verifique recursos disponíveis (CPU/memória) ou restrições de afinidade.'
    },
}


def load_error_lines() -> list[str]:
    """Carrega linhas de log que contenham palavras-chave de erro."""
    lines: list[str] = []
    if not os.path.isdir(LOG_DIR):
        return lines

    keywords = [k.lower() for k in ERROR_KB]
    for fname in os.listdir(LOG_DIR):
        if not fname.endswith('.log'):
            continue
        path = os.path.join(LOG_DIR, fname)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                lower = line.lower()
                if any(key in lower for key in keywords):
                    lines.append(line.strip())
    return lines


def search_error(question: str):
    """Retorna conhecimento sobre erros mencionados na pergunta."""
    question = question.lower()
    hits = []
    for err, data in ERROR_KB.items():
        if err.lower() in question:
            hits.append((err, data))
    return hits


def main():
    error_lines = load_error_lines()
    print('Pergunte sobre um erro do EKS (ex: CrashLoopBackOff):')
    question = input('> ')

    matches = search_error(question)
    if not matches:
        print('Nenhum erro conhecido detectado na pergunta.')
        return

    for err, info in matches:
        print(f"\n### {err}")
        example = next((l for l in error_lines if err.lower() in l.lower()), None)
        if example:
            print(f"Exemplo de log: {example}")
        print('Possível causa :', info['cause'])
        print('Possível solução:', info['solution'])


if __name__ == '__main__':
    main()
