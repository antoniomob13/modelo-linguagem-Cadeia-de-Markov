# Modelagem Estocástica de Linguagem via Cadeias de Markov de Classes e K-Means

Repositório oficial com a implementação prática do artigo **"Modelagem Estocástica de Linguagem via Cadeias de Markov de Classes e K-Means na Obra de Machado de Assis"**, desenvolvido para a disciplina de **Avaliação de Desempenho** no curso de Ciência da Computação da **Universidade Federal do Oeste do Pará (UFOPA)**.

## 👥 Autores
* **Antônio Mário Oliveira Barbosa** (`antonio.mob@discente.ufopa.edu.br`)
* **Fabiélen da Silva Nascimento** (`fabielen.nascimento@discente.ufopa.edu.br`)

---

## 📌 Sobre o Projeto

Modelos de linguagem baseados em Cadeias de Markov tradicionais que operam diretamente sobre palavras isoladas sofrem com a **esparsidade de dados** (matrizes gigantescas preenchidas quase inteiramente por zeros) e alto consumo de memória RAM ($O(\vert{}V\vert{}^2)$).

Para solucionar esse gargalo computacional, este projeto implementa uma **Cadeia de Markov de Tempo Discreto (DTMC)** híbrida baseada em classes semânticas e sintáticas:
1. **Corpus Textual:** Combina 171 obras narrativas (romances, contos e crônicas) de **Machado de Assis** com 51.397 sentenças do corpus **Mac-Morpho** (via biblioteca `nltk`), totalizando **1.993.276 tokens válidos** em **100.864 frases**.
2. **Projeção em Espaço Vetorial Contínuo ($D = 50$):** As $P = 25.000$ palavras mais frequentes do vocabulário são representadas em uma matriz de coocorrência ponderada por Informação Mútua Pontual Positiva (**PPMI**) e reduzidas para vetores densos de 50 dimensões via Decomposição em Valores Singulares Truncada (`TruncatedSVD`).
3. **Agrupamento Híbrido ($K = 80$ Estados):** O vocabulário é particionado em 80 estados, sendo **7 estados fixos** para classes gramaticais fechadas e pontuação (*artigos, preposições, pronomes, conjunções, verbos auxiliares, pausas* e *ponto final*) e **73 clusters semânticos** gerados via algoritmo **K-Means**.
4. **Avaliação de Desempenho e Geração:** O script constrói a matriz de transição $P$ ($80 \times 80$), calcula a **Distribuição Estacionária ($\pi$)**, o **Tempo Médio de Retorno ($M_{ii}$)** e a **Taxa de Entropia de Shannon ($H$)**, valida o equilíbrio teórico via **Simulação de Monte Carlo** ($100.000$ passos) e gera textos controlados por **Temperatura ($T$)** a partir de uma palavra escolhida pelo usuário (`input`).

---

## 📊 Principais Resultados Obtidos

| Métrica Analisada | Cadeia de Palavras ($P = 25.000$) | Cadeia de Clusters ($K = 80$) | Impacto Prático |
| :--- | :---: | :---: | :--- |
| **Quantidade de Estados ($\Vert{}S\Vert{}$)** | $25.000$ estados | $80$ estados | $312\times$ menos estados |
| **Tamanho da Matriz $P$ ($\Vert{}S\Vert{}^2$)** | $625.000.000$ células | $6.400$ células | $-99,999\%$ de memória |
| **Esparsidade (% de zeros)** | $99,95\%$ | $25,23\%$ | Redução drástica de zeros |
| **Taxa de Entropia de Shannon ($H$)** | — | $4,17$ bits/passo | Ramificação efetiva equilibrada |
| **Erro Médio (Teoria vs. Monte Carlo)** | — | $0,000148$ | Validação da convergência ergódica |

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
Certifique-se de ter o **Python 3.10+** instalado. Instale as bibliotecas necessárias executando no terminal:

```bash
pip install numpy scipy scikit-learn nltk
