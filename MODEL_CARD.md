# 📋 Model Card — ModaVision AI (`melhor_modelo_modavision.keras`)

**Projeto:** Atividade Prática 4 — NeuroStartup (Inteligência Artificial e Machine Learning)  
**Startup:** ModaVision AI  
**Arquitetura em Produção:** Rede Neural Convolucional (`CNN_ModaVision`)  
**Arquivo do Modelo:** `melhor_modelo_modavision.keras`  
**Semente de Reprodutibilidade:** `SEED = 42`

### 👥 Autores
- **Allan Reis da Conceição** — RA: 1539160
- **Gabriel Noriler Souza** — RA: 1642714
- **Mayara Franciele Santos da Silva** — RA: 1873454

---

## 1. 🎯 Uso Pretendido (*Intended Use*)
O modelo **`CNN_ModaVision`** foi projetado para **automatizar a categorização visual de peças de vestuário e calçados** no fluxo de cadastro de produtos em plataformas de e-commerce e marketplaces de moda (como Mercado Livre, Shein, Shopee e Dafiti).
- **Público-alvo:** Sistemas de *onboarding* de catálogo, equipes de operação de marketplaces e lojistas (*sellers*) que cadastram grandes volumes de peças por foto.
- **Funcionamento:** Ao receber a imagem de uma peça, o modelo retorna a distribuição de probabilidades entre **10 categorias padronizadas** (*Camiseta/Top, Calça, Suéter, Vestido, Casaco, Sandália, Camisa, Tênis, Bolsa e Bota*), permitindo o preenchimento automático da categoria quando a confiança é alta ou sugerindo as **Top-3 categorias** para validação rápida do lojista.

---

## 2. 🚫 Não Usar Para (*Out-of-Scope / Misuse*)
Este modelo **NÃO deve ser utilizado** nos seguintes cenários:
1. **Controle de qualidade ou inspeção de defeitos de fabricação:** A resolução `28x28` em tons de cinza não permite identificar rasgos, manchas, costuras defeituosas ou desgaste do tecido.
2. **Detecção de falsificações, pirataria ou autenticidade de marca:** O modelo avalia apenas a silhueta global da peça e não reconhece logotipos, etiquetas ou padrões de autenticidade.
3. **Decisões punitivas automatizadas contra vendedores:** O modelo não deve ser usado para bloquear anúncios, aplicar multas ou suspender contas de lojistas sem revisão humana (*Human-in-the-Loop*).
4. **Classificação de fotos com múltiplas peças simultâneas ou modelos humanos de corpo inteiro em cenários complexos:** O modelo espera uma única peça isolada e centralizada por imagem.

---

## 3. 🗂️ Dados Utilizados (*Training & Evaluation Data*)
O modelo foi desenvolvido sobre o dataset público **Fashion MNIST** (`keras.datasets.fashion_mnist.load_data()`), disponibilizado pela Zalando Research:
- **Volume total:** `70.000` imagens em tons de cinza (`1` canal), resolução `28 x 28` pixels (valores originais de `0` a `255`, normalizados para `[0.0, 1.0]` via divisão por `255.0`).
- **Balanceamento:** Perfeitamente balanceado entre as **10 categorias** (`7.000` imagens por classe no total).
- **Aplicação Rigorosa da Regra do Cofre:**
  - **Treino (`X_treino, y_treino`):** `48.000` imagens (80% do conjunto de desenvolvimento, amostragem estratificada com `random_state=42`).
  - **Validação (`X_valid, y_valid`):** `12.000` imagens (20% do conjunto de desenvolvimento, usadas para seleção de arquitetura e `EarlyStopping`).
  - **Cofre / Teste (`X_teste, y_teste`):** `10.000` imagens oficiais de teste, mantidas intocadas até a avaliação final única no Passo 3.

---

## 4. 📊 Desempenho no Teste Comparado ao Baseline (*Performance Metrics*)

### 4.1. Histórico de Experimentos na Validação (`12.000` imagens)
| ID | Modelo | Tipo | Arquitetura | Parâmetros | Épocas | Acc Treino | Acc Validação | Val Loss |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: |
| **5** | **`CNN_ModaVision`** | **Rede Convolucional (CNN)** | `Conv2D(32)->Pool->Conv2D(64)->Pool->Flatten->Dense(128)->Dense(10)` | **421.642** | **25** | **96,12%** | **93,13%** | **0,2018** |
| 4 | `MLP_Profunda_256_128_64` | Rede Densa (MLP) | `Dense(256)->Dense(128)->Dense(64)->Dense(10)` + `Dropout(0.2)` | 242.762 | 28 | 92,72% | 89,86% | 0,3029 |
| 3 | `MLP_Media_128_64` | Rede Densa (MLP) | `Dense(128)->Dense(64)->Dense(10)` + `Dropout(0.2)` | 109.386 | 23 | 91,50% | 89,39% | 0,3055 |
| 2 | `MLP_Simples_64` | Rede Densa (MLP) | `Dense(64)->Dense(10)` + `Dropout(0.2)` | 50.890 | 25 | 90,99% | 89,13% | 0,3091 |
| 1 | `Baseline (LogisticRegression)` | Linear (`scikit-learn`) | `Flatten(784) -> Linear(10)` | 7.850 | 1 | 87,69% | 86,12% | — |

### 4.2. Resultado Oficial no Cofre — Conjunto de Teste (`10.000` imagens)
| Modelo Avaliado no Cofre | Acurácia no Teste | Acertos a cada 1.000 Produtos | Erros a cada 1.000 Produtos | Impacto vs. Baseline |
| :--- | :-: | :-: | :-: | :-: |
| `Baseline (LogisticRegression)` | 84,27% | 843 | 157 | Referência inicial |
| **`CNN_ModaVision` (Produção)** | **92,27%** | **923** | **77** | **+8,00 p.p. (-51,0% de erros)** |

---

## 5. ⚠️ Limitações e Recomendações para Produção (*Limitations*)
1. **Resolução e Ausência de Cor (`28x28` em tons de cinza):** As imagens do Fashion MNIST não possuem canais RGB e têm baixa definição. Em categorias cujo diferencial depende de textura fina, botões ou recortes de gola (especialmente entre **Camisa**, **Camiseta/Top**, **Suéter** e **Casaco**), ocorre a maior parte das confusões residuais do modelo.
2. **Fundo Padronizado e Peça Isolada:** Todas as imagens de treino possuem fundo neutro e peça centralizada. Fotos reais tiradas por lojistas em ambientes domésticos, manequins ou com iluminação irregular sofreriam queda de desempenho (*domain shift*).
3. **Próximo Passo para Escala Real:** Para deploy em ambiente produtivo com fotos reais de smartphones de lojistas, recomenda-se aplicar *Transfer Learning* (ex.: `EfficientNetB0` ou `MobileNetV3`) utilizando imagens RGB em resolução `224x224` com *Data Augmentation* (rotação, brilho, contraste e recorte de fundo).
