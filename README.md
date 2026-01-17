# 🧪 Neural Fusion Lab: Advanced AI & Quantitative Research

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

> **"Exploring the Frontier where Quantitative Finance meets Agentic AI & Multimodal Learning."**

## 📖 About This Lab

**Neural Fusion Lab**은 금융공학(Financial Engineering)과 최신 인공지능(Modern AI) 기술의 융합을 탐구하는 실험적 연구 공간(Sandbox)입니다.

이 레포지토리의 주된 목적은 **"Evolution (진화)"**입니다.
기존에 수행했던 정통적인 퀀트/시계열 프로젝트(ARIMA, PPO 등)를 그대로 두지 않고, **최신 SOTA(State-of-the-Art) 아키텍처**와 **새로운 패러다임(Multi-Agent, GraphRAG)**을 적용하여 한 단계 더 발전시키는 것을 목표로 합니다.

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

## 📂 Lab Structure (예정)

이 레포지토리는 주제별 실험(Experiment) 단위로 구성될 예정입니다.

```bash

Neural-Fusion-Lab/
├── 📂 data/                  # 원천 데이터 (PDF, Images, CSV)
├── 📂 modules/               # 핵심 모듈
│   ├── __init__.py
│   ├── agents.py             # Agent 정의 (Supervisor, Chart, News, Quant)
│   ├── multimodal.py         # VLM(Vision Language Model) 처리 로직
│   ├── 🧪 exp_01_advanced_hedging/  # (Planned) Deep Hedging with RL
│   ├── 🧪 exp_02_graph_rag/         # (Planned) Financial Knowledge Graph
│   ├── graph_rag.py          # Neo4j 연결 및 Graph Traversal
│   └── tools.py              # 외부 API (yfinance, Tavily 등) 도구 모음
├── 📂 models/                # Fine-tuned LoRA weights 저장소
├── 📂 notebooks/             # 실험용 주피터 노트북 (EDA)
├── main.py                   # 실행 진입점 (Orchestrator)
├── config.yaml               # API Key 및 하이퍼파라미터
└── requirements.txt
```

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

```

```
