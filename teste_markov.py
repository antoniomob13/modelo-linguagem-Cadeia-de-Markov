import re
import urllib.request
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import nltk

# ==============================================================================
# CONFIGURAÇÕES GRAMATICAIS E CLASSES RESERVADAS (ESTADOS 0 A 6)
# ==============================================================================
CLASSES_FUNCIONAIS = {
    "artigo": {
        "o", "a", "os", "as", "um", "uma", "uns", "umas",
    },
    "preposicao": {
        "ao", "aos", "à", "às", "ante", "após", "até",
        "com", "contra", "de", "do", "dos", "da", "das", "desde",
        "em", "no", "nos", "na", "nas", "entre", "para", "por",
        "per", "sem", "sob", "sobre", "trás", "pelo", "pela", "pelos", "pelas",
    },
    "pronome": {
        "eu", "tu", "ele", "ela", "nós", "vós", "eles", "elas",
        "me", "te", "se", "lhe", "lhes", "mim", "ti", "si",
        "comigo", "contigo", "consigo", "meu", "minha", "meus", "minhas",
        "seu", "sua", "seus", "suas", "este", "esta", "estes", "estas",
        "esse", "essa", "esses", "essas", "isto", "isso", "aquilo",
        "quem", "qual", "quais", "cujo", "cuja", "cujos", "cujas",
    },
    "conjuncao": {
        "e", "nem", "mas", "porém", "contudo", "todavia", "ou",
        "ora", "porque", "pois", "que", "como", "quando",
        "enquanto", "embora", "ainda", "senão",
    },
    "auxiliar": {
        "ser", "sou", "é", "era", "eram", "foi", "foram", "estar",
        "está", "estava", "estavam", "ter", "tem", "tinha", "tinham",
        "haver", "há", "havia", "poder", "pode", "podia", "dever",
        "deve", "devia",
    },
}

CLASSES_PONTUACAO = {
    "pausa": {",", ";", ":"},
    "final": {".", "!", "?"},
}

TODOS_SINAIS_PONTUACAO = CLASSES_PONTUACAO["pausa"] | CLASSES_PONTUACAO["final"]

MARCADORES_ESTRANGEIROS = {
    "bonjour", "dans", "des", "elle", "elles", "il", "ils", "la", "las",
    "les", "on", "sont", "une", "vous", "mon", "ma", "mes", "avec", "pour",
    "the", "and", "of", "to", "in", "that", "is",
}

RUIDO_EDITORIAL = {
    "vol", "aguilar", "capítulo", "capitulo", "ii", "iii", "iv", "vi", "vii",
    "viii", "ix", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii",
    "xix", "xx", "pág", "pag", "ed", "edição", "nota", "notas", "cap",
}

# Palavras que NUNCA devem ficar imediatamente antes de uma vírgula ou ponto final
PROIBIDO_ANTES_DE_PONTUACAO = (
    CLASSES_FUNCIONAIS["artigo"]
    | CLASSES_FUNCIONAIS["preposicao"]
    | CLASSES_FUNCIONAIS["conjuncao"]
    | {"meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
       "este", "esta", "esse", "essa", "aquele", "aquela", "muito", "mais",
       "menos", "tão", "não", "já", "ainda", "também", "só", "bem"}
)


def classe_funcional(palavra):
    for classe, palavras_classe in CLASSES_FUNCIONAIS.items():
        if palavra in palavras_classe:
            return classe
    return None


def classe_especial(palavra):
    classe = classe_funcional(palavra)
    if classe is not None:
        return classe
    for nome, tokens in CLASSES_PONTUACAO.items():
        if palavra in tokens:
            return nome
    return None


def formatar_tokens_limpos(tokens):
    """Formata a lista de tokens garantindo espaçamento e maiúsculas corretas."""
    texto = ""
    maiuscula = True
    for token in tokens:
        if token in TODOS_SINAIS_PONTUACAO:
            texto = texto.rstrip() + token
            if token in CLASSES_PONTUACAO["final"]:
                maiuscula = True
        else:
            palavra = token.capitalize() if maiuscula else token
            texto += (" " if texto else "") + palavra
            maiuscula = False
    return texto


# ==============================================================================
# PASSO 1: Carregar e Limpar as Frases de Machado de Assis
# ==============================================================================
def carregar_machado():
    print("1. Carregando textos em português...")
    textos = []
    try:
        nltk.download("machado", quiet=True)
        from nltk.corpus import machado
        arquivos = [f for f in machado.fileids() if f.startswith(("romance/", "contos/", "cronica/"))]
        if not arquivos:
            arquivos = machado.fileids()
        for arquivo in arquivos:
            textos.append(machado.raw(arquivo))
        print(f"   -> {len(textos)} obras de Machado carregadas pelo NLTK!")
    except Exception as erro:
        print(f"   -> NLTK indisponível ({erro}); baixando Dom Casmurro...")
        url = "https://www.gutenberg.org/cache/epub/55752/pg55752.txt"
        requisicao = urllib.request.Request(url, headers={"User-Agent": "markov-machado/1.0"})
        resposta = urllib.request.urlopen(requisicao, timeout=60)
        textos.append(resposta.read().decode("utf-8", errors="ignore"))

    try:
        nltk.download("mac_morpho", quiet=True)
        from nltk.corpus import mac_morpho

        frases_jornalisticas = []
        for sentenca in mac_morpho.sents():
            palavras_sentenca = []
            for token in sentenca:
                palavra, separador, _ = token.rpartition("_")
                palavras_sentenca.append(palavra if separador else token)
            frases_jornalisticas.append(" ".join(palavras_sentenca))
        textos.append(" ".join(frases_jornalisticas))
        print(
            f"   -> {len(frases_jornalisticas):,} frases do Mac-Morpho adicionadas!"
        )
    except Exception as erro:
        print(f"   -> Mac-Morpho indisponível; seguindo apenas com Machado: {erro}")

    texto_completo = " ".join(textos).lower()
    frases = []
    for trecho in re.findall(r"[^.!?]+[.!?]", texto_completo):
        tokens = re.findall(
            r"[a-záàâãéêíóôõúç]+(?:-[a-záàâãéêíóôõúç]+)*|[,.!?;:]",
            trecho,
        )
        if any(t in RUIDO_EDITORIAL or t in MARCADORES_ESTRANGEIROS for t in tokens):
            continue

        # Limpa pontuações duplicadas ou no início da frase
        tokens_limpos = []
        for t in tokens:
            if t in TODOS_SINAIS_PONTUACAO:
                if not tokens_limpos or tokens_limpos[-1] in TODOS_SINAIS_PONTUACAO:
                    continue
                if tokens_limpos[-1] in PROIBIDO_ANTES_DE_PONTUACAO:
                    continue
            tokens_limpos.append(t)

        palavras_frase = [t for t in tokens_limpos if t not in TODOS_SINAIS_PONTUACAO]
        if 6 <= len(palavras_frase) <= 45:
            if tokens_limpos[-1] not in CLASSES_PONTUACAO["final"]:
                tokens_limpos.append(".")
            frases.append(tokens_limpos)

    palavras = [palavra for frase in frases for palavra in frase]
    print(f"   -> Total de tokens válidos lidos: {len(palavras):,} em {len(frases):,} frases.")
    return palavras, frases


# ==============================================================================
# PASSO 2: Vetorização Contínua (PPMI + SVD), K-Means e Indexação Sem Pontuação Cega
# ==============================================================================
def agrupar_palavras_kmeans(palavras, frases, tam_vocab=25000, dim_vetor=50, K=80):
    print(f"\n2. Criando vetores ({dim_vetor}D) e agrupando {tam_vocab} palavras em K={K} clusters...")

    contagem = Counter(palavras)
    vocab = [
        p for p, _ in contagem.most_common()
        if len(p) > 1 or p in {"a", "e", "o", "à", "é"} or p in TODOS_SINAIS_PONTUACAO
    ][:tam_vocab]
    palavra_para_id = {p: i for i, p in enumerate(vocab)}
    tamanho_vocab = len(vocab)

    seq_ids = [palavra_para_id[p] for p in palavras if p in palavra_para_id]

    # Filtra frases completas onde todas as palavras estão no vocabulário ativo
    frases_ids = []
    # Desacoplamos as palavras puras da pontuação para que a vírgula NÃO apague a memória!
    ngramas_palavras = {1: defaultdict(Counter), 2: defaultdict(Counter), 3: defaultdict(Counter), 4: defaultdict(Counter)}
    prob_pausa_apos = defaultdict(Counter)  # Registra quando Machado realmente usa vírgula/ponto após um par (w_{i-1}, w_i)
    aberturas_por_palavra = defaultdict(list)  # Permite iniciar o texto pela palavra escolhida no input()

    cooc = defaultdict(float)
    freq_uni = np.zeros(tamanho_vocab, dtype=np.float64)

    for frase in frases:
        if not all(p in palavra_para_id for p in frase):
            continue
        f_ids = [palavra_para_id[p] for p in frase]
        frases_ids.append(f_ids)

        # Extrai apenas a sequência de PALAVRAS (sem pontuação) para manter o nexo gramatical intacto
        apenas_palavras_ids = []
        for idx_tok, pid in enumerate(f_ids):
            token_str = vocab[pid]
            if token_str not in TODOS_SINAIS_PONTUACAO:
                apenas_palavras_ids.append(pid)
                # Verifica se logo após esta palavra veio uma pausa ou ponto na frase original
                if len(apenas_palavras_ids) >= 2:
                    par_atual = (apenas_palavras_ids[-2], apenas_palavras_ids[-1])
                    prox_tok = vocab[f_ids[idx_tok + 1]] if idx_tok + 1 < len(f_ids) else ""
                    if prox_tok in CLASSES_PONTUACAO["pausa"]:
                        prob_pausa_apos[par_atual][","] += 1
                    elif prox_tok in CLASSES_PONTUACAO["final"]:
                        prob_pausa_apos[par_atual]["."] += 1
                    else:
                        prob_pausa_apos[par_atual]["NENHUMA"] += 1

        if len(apenas_palavras_ids) >= 5:
            # Indexa a abertura real da frase pela 1ª palavra (e também por qualquer palavra onde ela apareça)
            aberturas_por_palavra[apenas_palavras_ids[0]].append(tuple(apenas_palavras_ids[:3]))

            n_pal = len(apenas_palavras_ids)
            for idx, atual in enumerate(apenas_palavras_ids):
                freq_uni[atual] += 1.0
                # Coocorrência semântica em janela de ±3 palavras reais
                for j in range(max(0, idx - 3), min(n_pal, idx + 4)):
                    if j != idx:
                        cooc[(atual, apenas_palavras_ids[j])] += 1.0 / abs(j - idx)

                # Indexa N-gramas de palavras puras (ordens 1 a 4 de histórico)
                for ordem in (1, 2, 3, 4):
                    if idx >= ordem:
                        hist = tuple(apenas_palavras_ids[idx - ordem:idx])
                        ngramas_palavras[ordem][hist][atual] += 1

    # Também contabiliza frequência dos sinais de pontuação para a matriz PPMI não ter linhas zeradas
    for pid in seq_ids:
        if vocab[pid] in TODOS_SINAIS_PONTUACAO:
            freq_uni[pid] += 1.0

    esparsidade_palavras = 1.0 - (
        sum(len(v) for v in ngramas_palavras[1].values()) / float(tamanho_vocab * tamanho_vocab)
    )

    # Matriz PPMI + TruncatedSVD (30 dimensões)
    total_cooc = sum(cooc.values())
    prob_uni = freq_uni / np.sum(freq_uni)
    linhas, colunas, dados = [], [], []
    for (i, j), val in cooc.items():
        p_ij = val / total_cooc
        pmi = np.log2(max(p_ij / (prob_uni[i] * prob_uni[j] + 1e-12), 1e-12))
        if pmi > 0:
            linhas.append(i)
            colunas.append(j)
            dados.append(pmi)

    matriz_ppmi = csr_matrix((dados, (linhas, colunas)), shape=(tamanho_vocab, tamanho_vocab))
    svd = TruncatedSVD(n_components=dim_vetor, random_state=42)
    vetores_palavras = normalize(svd.fit_transform(matriz_ppmi))

    # Reserva os 7 primeiros estados para classes funcionais/pontuação e agrupa conteúdo via K-Means
    classes = list(CLASSES_FUNCIONAIS) + list(CLASSES_PONTUACAO)
    num_funcionais = len(classes)
    classe_para_cluster = {classe: indice for indice, classe in enumerate(classes)}

    ids_conteudo = [i for i, p in enumerate(vocab) if classe_especial(p) is None]
    kmeans = KMeans(n_clusters=K - num_funcionais, random_state=42, n_init=10)
    clusters_conteudo = kmeans.fit_predict(vetores_palavras[ids_conteudo])

    cluster_da_palavra = np.empty(tamanho_vocab, dtype=np.int64)
    for indice, cluster in zip(ids_conteudo, clusters_conteudo):
        cluster_da_palavra[indice] = cluster + num_funcionais
    for indice, palavra in enumerate(vocab):
        classe = classe_especial(palavra)
        if classe is not None:
            cluster_da_palavra[indice] = classe_para_cluster[classe]

    return (
        seq_ids,
        vocab,
        palavra_para_id,
        vetores_palavras,
        cluster_da_palavra,
        esparsidade_palavras,
        ngramas_palavras,
        prob_pausa_apos,
        aberturas_por_palavra,
        frases_ids,
    )


# ==============================================================================
# PASSO 3: Construir a Matriz de Transição P (KxK) da Cadeia de Markov
# ==============================================================================
def construir_matriz_P(seq_ids, vocab, cluster_da_palavra, K=80, alpha=0.001, frases_ids=None):
    print("\n3. Construindo a Matriz de Transição P entre os clusters...")

    sequencias = frases_ids if frases_ids is not None else [seq_ids]
    contagem_P = np.zeros((K, K), dtype=np.float64)
    for frase in sequencias:
        seq_estados = [int(cluster_da_palavra[id_p]) for id_p in frase]
        for i in range(len(seq_estados) - 1):
            contagem_P[seq_estados[i], seq_estados[i + 1]] += 1.0

    esparsidade_clusters = np.sum(contagem_P == 0) / float(K * K)
    soma_linhas = np.sum(contagem_P, axis=1, keepdims=True)
    P = (contagem_P + alpha) / (soma_linhas + K * alpha)

    palavras_por_cluster = {k: Counter() for k in range(K)}
    for id_p in seq_ids:
        c = int(cluster_da_palavra[id_p])
        palavras_por_cluster[c][vocab[id_p]] += 1

    emissao = {}
    prob_emissao_id = np.zeros(len(vocab), dtype=np.float64)
    palavra_para_id = {w: i for i, w in enumerate(vocab)}

    for k in range(K):
        total = sum(palavras_por_cluster[k].values())
        mais_comuns = palavras_por_cluster[k].most_common()
        lista_p = [w for w, _ in mais_comuns]
        probs_p = np.array([cont / total for _, cont in mais_comuns], dtype=np.float64)
        emissao[k] = (lista_p, probs_p)
        for w, cont in mais_comuns:
            prob_emissao_id[palavra_para_id[w]] = cont / total

    return P, emissao, prob_emissao_id, esparsidade_clusters


# ==============================================================================
# PASSO 4: Calcular π (Estacionária), Retorno (M_ii) e Entropia (H)
# ==============================================================================
def calcular_metricas_desempenho(P):
    print("\n4. Calculando Distribuição Estacionária (pi), Retorno (M_ii) e Entropia (H)...")
    K = P.shape[0]

    A = np.vstack([P.T - np.eye(K), np.ones(K)])
    b = np.zeros(K + 1)
    b[-1] = 1.0
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    pi = np.maximum(pi, 1e-15)
    pi = pi / np.sum(pi)

    retorno_medio = 1.0 / pi
    entropia_por_estado = -np.sum(P * np.log2(P), axis=1)
    H = float(np.sum(pi * entropia_por_estado))

    return pi, retorno_medio, H


# ==============================================================================
# PASSO 5: Validação Monte Carlo e Geração com Coerência Vetorial + Pontuação Limpa
# ==============================================================================
def simular_monte_carlo(P, pi_teorico, n_passos=100000):
    print(f"\n5. Validando na prática com Simulação de Monte Carlo ({n_passos:,} passos)...")
    K = P.shape[0]
    visitas = np.zeros(K)
    estado = 0
    rng = np.random.default_rng(42)

    for _ in range(n_passos):
        visitas[estado] += 1
        estado = rng.choice(K, p=P[estado])

    pi_simulado = visitas / n_passos
    erro_medio = np.mean(np.abs(pi_teorico - pi_simulado))
    print(f"   -> Diferença média entre Teoria (Sistema Linear) e Simulação: {erro_medio:.6f}")
    return pi_simulado


def escolher_abertura_por_input(
    palavra_usuario,
    palavra_para_id,
    vocab,
    aberturas_por_palavra,
    ngramas_palavras,
    rng,
):
    """
    Inicializa a sequência exatamente a partir da palavra (ou frase curta)
    definida pelo usuário via input(), ancorando o contexto inicial.
    """
    if palavra_usuario:
        tokens_usr = re.findall(r"[a-záàâãéêíóôõúç]+(?:-[a-záàâãéêíóôõúç]+)*", palavra_usuario.lower())
        ids_usr = [palavra_para_id[t] for t in tokens_usr if t in palavra_para_id]
        if ids_usr:
            # Se o usuário digitou 2 ou mais palavras conhecidas, usa todas elas como início!
            if len(ids_usr) >= 2:
                return ids_usr
            pid_inicial = ids_usr[0]
            # Se existe uma abertura de frase real com essa palavra em Machado de Assis, sorteia uma
            if aberturas_por_palavra.get(pid_inicial):
                opcoes = aberturas_por_palavra[pid_inicial]
                escolhida = opcoes[int(rng.integers(0, len(opcoes)))]
                return list(escolhida[:2])
            # Caso a palavra só apareça no meio das frases, pega a continuação mais natural dela
            conts = ngramas_palavras[1].get((pid_inicial,))
            if conts:
                prox_id = conts.most_common(1)[0][0]
                return [pid_inicial, prox_id]
            return [pid_inicial]
        else:
            print(f"   [Aviso] A palavra '{palavra_usuario}' não está no vocabulário ativo. Usando início padrão.")

    # Fallback: escolhe uma abertura clássica frequente nos romances
    todas_aberturas = [ab for lista in aberturas_por_palavra.values() for ab in lista]
    escolhida = todas_aberturas[int(rng.integers(0, len(todas_aberturas)))]
    return list(escolhida[:2])


def gerar_texto(
    P,
    emissao,
    prob_emissao_id,
    vetores_palavras,
    ngramas_palavras,
    prob_pausa_apos,
    aberturas_por_palavra,
    vocab,
    palavra_para_id,
    cluster_da_palavra,
    palavra_inicial=None,
    tamanho=35,
    temperatura=0.85,
    semente=42,
    continuo=False,
):
    """
    Gera texto com alto nexo combinando:
    1. Memória lexical sem bloqueio por vírgula (Stupid Backoff de ordem 4 -> 3 -> 2 -> 1);
    2. Coerência temática por Similaridade de Cosseno no espaço vetorial SVD (30D);
    3. Probabilidade de transição da Cadeia de Markov de Clusters P[estado_atual, c_cand];
    4. Máquina de estados para inserção limpa de vírgulas e pontos finais.
    """
    rng = np.random.default_rng(semente)

    historico_palavras = escolher_abertura_por_input(
        palavra_inicial, palavra_para_id, vocab, aberturas_por_palavra, ngramas_palavras, rng
    )
    tokens_saida = [vocab[pid] for pid in historico_palavras]
    estado_atual = int(cluster_da_palavra[historico_palavras[-1]])

    # Inicializa o Vetor de Tópico (centróide semântico da frase no espaço 30D)
    vetor_topico = np.zeros(vetores_palavras.shape[1], dtype=np.float64)
    for pid in historico_palavras:
        if classe_especial(vocab[pid]) is None:
            vetor_topico += vetores_palavras[pid]
    norma = np.linalg.norm(vetor_topico)
    if norma > 1e-9:
        vetor_topico /= norma

    palavras_desde_pausa = len(historico_palavras)
    palavras_na_oracao = len(historico_palavras)

    while len(historico_palavras) < tamanho:
        ultima_id = historico_palavras[-1]
        ultima_str = vocab[ultima_id]

        # 1. Busca Hierárquica de Contexto (prioriza ordens longas que preservam o sentido completo)
        candidatos_pesos = {}
        for ordem, fator_backoff in ((4, 1.0), (3, 0.6), (2, 0.25), (1, 0.05)):
            if len(historico_palavras) >= ordem:
                chave = tuple(historico_palavras[-ordem:])
                contagens = ngramas_palavras[ordem].get(chave)
                if contagens:
                    # Filtra candidatos que não repetem palavras recentes nem quebram sintaxe
                    validos = {}
                    for pid, cont in contagens.items():
                        cand_str = vocab[pid]
                        if pid == ultima_id:
                            continue
                        # Evita repetir a mesma palavra de conteúdo na janela recente
                        if classe_funcional(cand_str) is None and pid in historico_palavras[-7:]:
                            continue
                        # Evita colisão de duas palavras da mesma classe funcional fechada
                        c_ult = classe_funcional(ultima_str)
                        c_cand = classe_funcional(cand_str)
                        if c_ult and c_ult == c_cand and c_ult in {"artigo", "preposicao", "conjuncao"}:
                            continue
                        validos[pid] = cont

                    if validos:
                        total = float(sum(validos.values()))
                        for pid, cont in validos.items():
                            # Desconto absoluto leve para ordens baixas (evita ruído de contagem 1)
                            c_desc = max(cont - 0.35, 0.05) if ordem <= 2 else float(cont)
                            candidatos_pesos[pid] = candidatos_pesos.get(pid, 0.0) + fator_backoff * (c_desc / total)
                        # Se já temos contexto sólido de 3 ou 4 palavras, não poluímos com ordem 1!
                        if ordem >= 2 and len(candidatos_pesos) >= 3:
                            break

        if candidatos_pesos:
            candidatos = np.array(list(candidatos_pesos.keys()), dtype=np.int64)
            p_ctx = np.array([candidatos_pesos[pid] for pid in candidatos], dtype=np.float64)
            clusters_cand = cluster_da_palavra[candidatos]

            # 2. Peso da Cadeia de Markov de Clusters: P[estado_atual, cluster_candidato]
            p_markov = P[estado_atual, clusters_cand]

            # 3. Coerência Temática: Similaridade de Cosseno com o Vetor de Tópico da frase
            if np.linalg.norm(vetor_topico) > 1e-9:
                sim_cosseno = vetores_palavras[candidatos] @ vetor_topico
                # Transforma similaridade [-1, 1] em multiplicador suave [0.7, 1.6]
                bonus_topico = np.clip(1.0 + 0.6 * sim_cosseno, 0.7, 1.6)
            else:
                bonus_topico = 1.0

            scores = (p_ctx ** 0.75) * (p_markov ** 0.25) * bonus_topico

            # 4. Amostragem Top-k / Top-p (Nucleus) controlada por Temperatura
            top_k = min(8, len(candidatos))
            idx_top = np.argsort(scores)[::-1][:top_k]
            candidatos = candidatos[idx_top]
            scores = scores[idx_top]

            log_s = np.log(np.maximum(scores, 1e-15)) / max(temperatura, 0.15)
            log_s -= np.max(log_s)
            probs = np.exp(log_s)
            probs /= np.sum(probs)

            prox_id = int(rng.choice(candidatos, p=probs))
        else:
            # Backoff pela Cadeia de Markov de Clusters caso a palavra seja rara
            probs_c = np.power(P[estado_atual], 1.0 / max(temperatura, 0.2))
            probs_c /= np.sum(probs_c)
            prox_cluster = int(rng.choice(P.shape[0], p=probs_c))
            palavras_k, probs_k = emissao[prox_cluster]
            palavra_sorteada = str(rng.choice(palavras_k[:15], p=probs_k[:15] / np.sum(probs_k[:15])))
            prox_id = palavra_para_id[palavra_sorteada]

        prox_str = vocab[prox_id]
        historico_palavras.append(prox_id)
        tokens_saida.append(prox_str)
        estado_atual = int(cluster_da_palavra[prox_id])
        palavras_desde_pausa += 1
        palavras_na_oracao += 1

        # Atualiza o Vetor de Tópico se a nova palavra for de conteúdo (substantivo/verbo/adjetivo)
        if classe_especial(prox_str) is None:
            vetor_topico = 0.85 * vetor_topico + 0.15 * vetores_palavras[prox_id]
            norma = np.linalg.norm(vetor_topico)
            if norma > 1e-9:
                vetor_topico /= norma

        # 5. Controle Inteligente de Pontuação (Cadência Real de Machado de Assis)
        if prox_str not in PROIBIDO_ANTES_DE_PONTUACAO and len(historico_palavras) >= 2:
            par_recente = (historico_palavras[-2], historico_palavras[-1])
            cont_pausa = prob_pausa_apos.get(par_recente)
            if cont_pausa:
                total_ocorrencias = float(sum(cont_pausa.values()))
                taxa_ponto = cont_pausa["."] / total_ocorrencias
                taxa_virgula = cont_pausa[","] / total_ocorrencias

                # Fecha a oração com ponto final apenas se já tiver tamanho completo (>= 9 palavras)
                if palavras_na_oracao >= 9 and (taxa_ponto >= 0.35 or (palavras_na_oracao >= 18 and taxa_ponto >= 0.15)):
                    tokens_saida.append(".")
                    estado_atual = int(cluster_da_palavra[palavra_para_id["."]])
                    palavras_desde_pausa = 0
                    palavras_na_oracao = 0
                    if not continuo or len(historico_palavras) >= tamanho - 4:
                        break
                # Insere vírgula apenas quando o par pede pausa natural e não houve vírgula recente
                elif palavras_desde_pausa >= 5 and taxa_virgula >= 0.40 and len(historico_palavras) < tamanho - 2:
                    tokens_saida.append(",")
                    estado_atual = int(cluster_da_palavra[palavra_para_id[","]])
                    palavras_desde_pausa = 0

    # Garante fechamento limpo sem terminar a frase em preposição, artigo ou conjunção
    while tokens_saida and tokens_saida[-1] in (PROIBIDO_ANTES_DE_PONTUACAO | CLASSES_PONTUACAO["pausa"]):
        tokens_saida.pop()
    if not tokens_saida or tokens_saida[-1] not in CLASSES_PONTUACAO["final"]:
        tokens_saida.append(".")

    return formatar_tokens_limpos(tokens_saida)


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    palavras, frases = carregar_machado()
    (
        seq_ids,
        vocab,
        palavra_para_id,
        vetores_palavras,
        cluster_da_palavra,
        esp_palavras,
        ngramas_palavras,
        prob_pausa_apos,
        aberturas_por_palavra,
        frases_ids,
    ) = agrupar_palavras_kmeans(
        palavras, frases, tam_vocab=25000, dim_vetor=50, K=80
    )
    P, emissao, prob_emissao_id, esp_clusters = construir_matriz_P(
        seq_ids,
        vocab,
        cluster_da_palavra,
        K=80,
        alpha=0.001,
        frases_ids=frases_ids,
    )
    pi, retorno, H = calcular_metricas_desempenho(P)
    simular_monte_carlo(P, pi, n_passos=100000)

    print("\n" + "="*75)
    print("RESULTADOS PARA COLOCAR NA TABELA DO ARTIGO:")
    print("="*75)
    print(f"Esparsidade na Matriz de Palavras (25000x25000): {esp_palavras*100:.2f}% de zeros")
    print(f"Esparsidade na Matriz de Clusters (80x80)      : {esp_clusters*100:.2f}% de zeros")
    print(f"Taxa de Entropia de Shannon da Cadeia (H)     : {H:.2f} bits/passo")

    print("\nTOP 5 ESTADOS (CLUSTERS) MAIS VISITADOS NO LONGO PRAZO:")
    mais_visitados = np.argsort(pi)[::-1]
    for pos, k in enumerate(mais_visitados[:5], 1):
        exemplos = ", ".join(emissao[k][0][:6])
        print(f" {pos}º) Cluster {k:02d} | pi = {pi[k]:.4f} ({pi[k]*100:.1f}%) "
              f"| Retorno M_ii = {retorno[k]:.2f} passos | Ex: [{exemplos}]")

    # ==========================================================================
    # ENTRADA INTERATIVA DA PALAVRA INICIAL DEFINIDA PELO USUÁRIO
    # ==========================================================================
    print("\n" + "="*75)
    palavra_input = input("Digite uma palavra (ou início de frase) para começar o texto [ex: capitu, olhos, amor, casa]: ").strip()
    if not palavra_input:
        palavra_input = "capitu"

    print("="*75)
    print(f"EXEMPLOS DE TEXTO INICIADOS POR '{palavra_input.upper()}' COM DIFERENTES TEMPERATURAS:")
    print("="*75)
    for idx_t, temp in enumerate([0.7, 1.0, 1.3]):
        texto_gerado = gerar_texto(
            P,
            emissao,
            prob_emissao_id,
            vetores_palavras,
            ngramas_palavras,
            prob_pausa_apos,
            aberturas_por_palavra,
            vocab,
            palavra_para_id,
            cluster_da_palavra,
            palavra_inicial=palavra_input,
            tamanho=35,
            temperatura=temp,
            semente=42 + idx_t * 11,
            continuo=True,
        )
        print(f"\nTemperatura T = {temp}:\n\"{texto_gerado}\"")

    print("\n" + "="*75)
    print(f"TEXTO LONGO INICIADO POR '{palavra_input.upper()}':")
    print("="*75)
    texto_longo = gerar_texto(
        P,
        emissao,
        prob_emissao_id,
        vetores_palavras,
        ngramas_palavras,
        prob_pausa_apos,
        aberturas_por_palavra,
        vocab,
        palavra_para_id,
        cluster_da_palavra,
        palavra_inicial=palavra_input,
        tamanho=140,
        temperatura=0.7,
        semente=2026,
        continuo=True,
    )
    print(texto_longo)