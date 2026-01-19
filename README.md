# 🧪 Neural Fusion Lab: Multimodal Agentic Financial System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-blue?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-GraphRAG-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **"Exploring the Frontier where Quantitative Finance meets Agentic AI & Multimodal Learning."**

## 📖 About This Lab

**Neural Fusion Lab**은 금융공학(Financial Engineering)과 최신 인공지능(Modern AI) 기술의 융합을 탐구하는 **실험적 연구 공간(Research Sandbox)**입니다.

이 레포지토리는 완성된 프로덕트가 아닌, **"Evolution (진화)"**을 목표로 하는 일련의 **실험(Experiments)**들을 기록합니다. 기존의 정통적인 퀀트 모델을 베이스라인(Legacy)으로 삼아, 최신 SOTA 아키텍처와 새로운 패러다임(Multi-Agent, GraphRAG)을 적용했을 때의 효용성을 검증합니다.

---

## 🧪 Research Tracks & Experiments

현재 본 연구실에서는 아래 3가지 핵심 트랙을 중심으로 실험을 진행하고 있습니다.

### Track 1. 🧬 Evolution of Legacy Models (모델 고도화 실험)
> **Hypothesis:** "딥러닝 기반의 생성형 모델이 전통적 통계 모델의 경직성을 극복할 수 있는가?"

* **Legacy (Baseline):** 기존의 `ARIMA`, `LSTM`, `PPO` 기반의 단일 모델 접근법.
* **Evolution (Experimental):**
    * **TimeSeries Foundation Models:** Chronos, TimeGPT 등을 활용한 제로샷 예측 성능 검증 실험.
    * **Deep Hedging:** 고전적 델타 헷징을 넘어선 강화학습(RL) 기반의 비선형 헷징 전략 연구.
    * **Neural SDEs:** 데이터를 통해 확률 미분방정식(SDE)을 직접 학습하는 생성형 시계열 모델링 실험.

### Track 2. 🤖 Agentic Workflow (에이전트 오케스트레이션 실험)
> **Hypothesis:** "단일 LLM을 넘어선 전문 에이전트 협업 체계가 금융 분석의 신뢰도를 높일 수 있는가?"

* **Legacy (Baseline):** 사람이 직접 피처를 가공하고 모델을 트리거하는 수동 프로세스.
* **Evolution (Experimental):**
    * **Multi-Agent Systems:** LangGraph를 활용해 데이터 수집, 차트 분석, 리스크 관리를 분담하는 에이전트 협업 시스템 구축 및 상태 관리(State Management) 실험.
    * **Auto-Quant:** 투자 가설 설정부터 백테스팅 코드 작성까지 스스로 수행하는 **자율형 퀀트(Autonomous Quant)** 에이전트 프로토타이핑.

### Track 3. 🧠 Knowledge-Driven AI (지식 기반 추론 실험)
> **Hypothesis:** "단순 검색(Search)을 넘어선 구조적 추론(Reasoning)이 리스크 탐지에 유효한가?"

* **Legacy (Baseline):** 단순 텍스트 유사도 기반의 검색(Simple RAG)이나 키워드 매칭.
* **Evolution (Experimental):**
    * **GraphRAG:** 기업 지배구조, 공급망 등 복잡한 관계를 **지식 그래프(Knowledge Graph)**로 시각화하고 2차 파급 효과를 추론하는 실험.
    * **Multimodal Analysis:** 재무제표(텍스트)와 차트(이미지)를 동시에 이해하는 VLM(Vision-Language Model) 기반 분석 실험.

---

## 🔬 Deep Dive: Active Experiment
**현재 중점적으로 진행 중인 개별 연구 주제입니다.**

### [Exp 03] Structural Analysis of Financial Graphs

**Objective:**
단순한 노드 연결 분석을 넘어, **그래프 이론(Graph Theory)**과 **딥러닝**을 결합하여 금융 네트워크 내의 **구조적 동형성(Isomorphism)**을 판별합니다. 자금 세탁 패턴(Money Laundering Ring)이나 순환 출자(Circular Shareholding)와 같은 특이 구조를 탐지하는 것이 목표입니다.

**Theoretical Background:**
* **Graph Isomorphism (GI):** 두 그래프가 구조적으로 동일한지 판별하는 난제 (NP-intermediate).
* **Subgraph Isomorphism:** 거대 그래프 내에서 특정 패턴 그래프와 동형인 부분집합을 찾는 문제 (NP-Complete).

**Methodology:**
1.  **Exact Matching (Hard Approach):**
    * **Algorithm:** VF2 Algorithm (via NetworkX)
    * **Description:** 엄밀한 수학적 정의(Bijective mapping)에 기반하여, 정의된 금융 사기 패턴과 정확히 일치하는 부분 그래프를 전수 조사.
2.  **Neural Matching (Soft Approach):**
    * **Model:** Graph Isomorphism Network (GIN)
    * **Description:** Weisfeiler-Lehman (WL) Test 기반의 GNN을 활용하여, 노이즈가 섞인 데이터에서도 두 그래프 구조의 **유사도(Cosine Similarity)**를 산출하여 스코어링.

---

## 🏗️ Experimental Architecture

이 프로젝트는 단일 스크립트가 아닌, **Stateful Multi-Agent Architecture**를 실험적으로 채택하여 에이전트 간의 메시지 흐름을 제어합니다.

```mermaid
graph TD
    User((User)) -->|Input Ticker| Supervisor[🕵️ Supervisor Agent]
    
    subgraph "Experimental Agentic Workflow"
        Supervisor -->|Route| Chart[📈 Chart Analyst\n(GPT-4o Vision)]
        Supervisor -->|Route| Knowledge[🕸️ Knowledge Analyst\n(Neo4j GraphRAG)]
        Supervisor -->|Route| Quant[🧮 Quant Analyst\n(Technical Indicators)]
        
        Chart -->|Analysis| Supervisor
        Knowledge -->|Insight| Supervisor
        Quant -->|Metrics| Supervisor
    end
    
    Supervisor -->|Final Decision| User
```

## 📂 Lab Structure (예정)

이 레포지토리는 안정적인 **핵심 모듈(Modules)**과 다양한 시도를 수행하는 **실험실(Experiments)**로 구분됩니다.

```bash
Neural-Fusion-Lab/
├── 📂 modules/               # [Stable] 재사용 가능한 핵심 컴포넌트
│   ├── agents.py             # LangGraph Nodes & Supervisor Logic
│   ├── multimodal.py         # VLM Engine (Image Encoding & Prompting)
│   ├── graph_rag.py          # Neo4j Connector & Cypher Query Engine
│   └── tools.py              # Market Data Fetcher (yfinance wrapper)
│
├── 📂 experiments/           # [Sandbox] 주제별 연구 및 실험 코드
│   ├── 🧪 exp_01_advanced_rag/        # (Completed) Financial Text Analysis
│   ├── 🧪 exp_02_multimodal_chart/    # (Completed) VLM based Technical Analysis
│   ├── 🧪 exp_03_structural_analysis/ # (Active) Graph Isomorphism & GIN
│   └── 🧪 exp_04_neural_sde/          # (Planned) SDE learning from Data
│
├── 📂 data/                  # Experiment Data (Generated Charts, CSVs)
├── 📂 notebooks/             # EDA & Prototyping Jupyter Notebooks
├── main.py                   # System Entry Point
├── config.yaml               # Experiment Configuration
└── requirements.txt
```

## 🚀 Key Features & Implementation

### 1. 👁️ Multimodal Technical Analysis (`modules/multimodal.py`)
* **Dynamic Visualization:** `matplotlib`를 사용하여 실시간으로 주가 차트(SMA, Bollinger Bands 포함)를 생성.
* **Vision AI:** 생성된 차트를 이미지로 인코딩하여 **GPT-4o(VLM)**에 주입. 단순 수치로 파악하기 힘든 시각적 패턴(Head & Shoulders, Wedge 등)을 분석.

### 2. 🕸️ Knowledge Graph RAG (`modules/graph_rag.py`)
* **Neo4j Integration:** 기업 간 관계(지분 구조, 공급망, 경쟁사)를 저장한 그래프 데이터베이스와 연동.
* **Supply Chain Risk:** 단순 텍스트 뉴스가 아닌, 연결된 노드(Node)를 탐색하여 2차, 3차 파급 효과(Ripple Effect)를 추론.

### 3. 🤖 Orchestration with LangGraph (`modules/agents.py`)
* **State Management:** `TypedDict`를 활용하여 에이전트 간 공유 메모리(Context) 관리.
* **Router Logic:** 작업의 종류에 따라 적합한 에이전트를 호출하고 결과를 취합하는 중앙 제어 구조.

## 💻 Getting Started

### Prerequisites
* Python 3.10+
* Neo4j Database (Local or AuraDB)
* OpenAI API Key

### Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/neural-fusion-lab.git](https://github.com/your-username/neural-fusion-lab.git)
    cd neural-fusion-lab
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration**
    * `.env` 파일을 생성하고 API 키를 설정합니다.
        ```ini
        OPENAI_API_KEY=sk-proj-...
        NEO4J_URI=bolt://localhost:7687
        NEO4J_USER=neo4j
        NEO4J_PASSWORD=your_password
        ```
    * `config.yaml`에서 모델 버전 및 파라미터를 조정할 수 있습니다.

### Usage
```bash
```
python main.py 실행 시 main.py에 설정된 종목(예: NVDA)에 대해 분석을 시작하며, data/ 폴더에 차트 이미지가 생성되고 터미널에 최종 분석 리포트가 출력됩니다.

## 🔮 Future Research (Roadmap)

이 프로젝트는 지속적으로 고도화될 예정입니다.

* [x] **Phase 1 (Completed):** Multi-Agent System 구축 및 Multimodal/GraphRAG 연동.
* [ ] **Phase 2 (In Progress):**
    * **Fine-tuning LLaVA:** 금융 차트 특화 Vision Model 파인튜닝.
    * **Text-to-Cypher:** 자연어 질의를 정교한 그래프 쿼리로 변환하는 모델 학습.
* [ ] **Phase 3 (Planned):**
    * **Deep Hedging (PPO):** 강화학습 기반의 포트폴리오 최적화 모듈 탑재.
    * **Auto-Backtest:** 에이전트가 제안한 전략을 즉시 검증하는 백테스팅 엔진 연동.

## 🛠️ Tech Stack & Tools

* **Core AI:** PyTorch, Stable-Baselines3, Hugging Face Transformers
* **LLM & Agents:** LangChain, LangGraph, LlamaIndex
* **Database & Graph:** Neo4j, Vector DB (Chroma/FAISS)
* **Finance:** yfinance, TA-Lib, QuantLib

---

### 👨‍💻 Maintainer

* **Jae Yeon Park**
* M.S. in Mathematical Sciences (Financial Mathematics)
* Focus: DRL in Finance, Time-Series Analysis, Agentic AI

---

*Disclaimer: This repository is for research and educational purposes. Not financial advice.*
