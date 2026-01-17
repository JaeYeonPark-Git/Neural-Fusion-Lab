# 🧪 Neural Fusion Lab: Multimodal Agentic Financial System

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-blue?logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-GraphRAG-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **"Exploring the Frontier where Quantitative Finance meets Agentic AI & Multimodal Learning."**

## 📖 About This Lab

**Neural Fusion Lab**은 금융공학(Financial Engineering)과 최신 인공지능(Modern AI) 기술의 융합을 탐구하는 실험적 연구 공간(Sandbox)입니다.

이 레포지토리의 주된 목적은 **"Evolution (진화)"**입니다.
기존에 수행했던 정통적인 퀀트/시계열 프로젝트(ARIMA, PPO 등)를 그대로 두지 않고, **최신 SOTA(State-of-the-Art) 아키텍처**와 **새로운 패러다임(Multi-Agent, GraphRAG)**을 적용하여 한 단계 더 발전시키는 것을 목표로 합니다.

## 📖 Project Overview

**Neural Fusion Lab**은 정통 금융공학(Quantitative Finance)과 최신 에이전트 AI(Agentic AI) 기술을 결합한 차세대 금융 분석 시스템입니다.

기존의 단순 수치 분석을 넘어, **LangGraph** 기반의 멀티 에이전트 협업 시스템을 구축하여 다음과 같은 복합적인 추론을 수행합니다.

* **Vision Analyst:** 주가 차트 이미지를 시각적으로 분석(VLM)하여 기술적 패턴 식별.
* **Knowledge Analyst:** 기업 지배구조 및 공급망 데이터를 지식 그래프(Neo4j)로 탐색하여 리스크 진단.
* **Quant Analyst:** 시계열 데이터 및 기술적 지표(RSI, MACD 등)를 계산.
* **Supervisor:** 위 모든 정보를 종합하여 최적의 투자의견(Buy/Sell/Hold) 도출.

## 🚀 Research Direction (연구 방향)

현재 이 연구실은 다음과 같은 방향으로 기존 코드들을 리팩토링하고 확장할 계획입니다.

### 1. 🧬 Evolution of Legacy Models (기존 모델의 고도화)
* **Legacy:** 기존의 `ARIMA`, `LSTM`, `PPO` 기반의 단일 모델 접근법.
* **Evolution:**
    * **TimeSeries Foundation Models:** Chronos, TimeGPT 등을 활용한 제로샷 예측 성능 검증.
    * **Deep Hedging:** 고전적 델타 헷징을 넘어선 강화학습(RL) 기반의 비선형 헷징 전략 연구.
    * **Neural SDEs:** 데이터를 통해 미분방정식을 직접 학습하는 생성형 시계열 모델링.

### 2. 🤖 Agentic Workflow (에이전트 기반 워크플로우)
* **Legacy:** 사람이 직접 피처를 가공하고 모델을 돌리는 수동 프로세스.
* **Evolution:**
    * **Multi-Agent Systems:** 데이터 수집, 차트 분석, 리스크 관리를 각각 담당하는 AI 에이전트 협업 시스템 구축.
    * **Auto-Quant:** 투자 가설 설정부터 백테스팅까지 스스로 수행하는 자율형 퀀트 에이전트 실험.

### 3. 🧠 Knowledge-Driven AI (지식 기반 AI)
* **Legacy:** 단순 텍스트 검색(Simple RAG)이나 키워드 매칭.
* **Evolution:**
    * **GraphRAG:** 기업 지배구조, 공급망 등 복잡한 관계를 **지식 그래프(Knowledge Graph)**로 시각화하고 추론.
    * **Multimodal Analysis:** 재무제표(텍스트)와 차트(이미지)를 동시에 이해하는 멀티모달 모델 연구.

## 🏗️ System Architecture

이 프로젝트는 **Stateful Multi-Agent Architecture**를 채택하여, 에이전트 간의 메시지 흐름과 상태(State)를 엄격하게 관리합니다.

```mermaid
graph TD
    User((User)) -->|Input Ticker| Supervisor[🕵️ Supervisor Agent]
    
    subgraph "Agentic Workflow (LangGraph)"
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

이 레포지토리는 주제별 실험(Experiment) 단위로 구성될 예정입니다.

```bash
Neural-Fusion-Lab/
├── 📂 data/                  # Generated charts & Raw financial data
├── 📂 modules/               # Core Logic Modules
│   ├── __init__.py
│   ├── agents.py             # LangGraph Nodes & Supervisor Logic
│   ├── multimodal.py         # VLM Engine (Image Encoding & Prompting)
│   ├── graph_rag.py          # Neo4j Connector & Cypher Query Engine
│   └── tools.py              # Market Data Fetcher (yfinance wrapper)
├── 📂 notebooks/             # EDA & Prototype Experiments
├── main.py                   # Entry Point (Graph Compilation & Execution)
├── config.yaml               # Model Configs & Hyperparameters
├── requirements.txt          # Python Dependencies
└── README.md                 # Project Documentation
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
