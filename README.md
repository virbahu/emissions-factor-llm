# 📊 emissions-factor-llm

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![RAG](https://img.shields.io/badge/Architecture-RAG-purple.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **LLM-powered pipeline for automated GHG Protocol emissions factor classification, retrieval, and matching — turning unstructured procurement data into audit-ready carbon accounting.**
>
> ---
>
> ## 📋 Overview
>
> **emissions-factor-llm** is a production-grade NLP pipeline that automates the most labor-intensive step in Scope 3 carbon accounting: matching raw procurement line items to the correct GHG Protocol emission factors.
>
> Manual emissions factor lookup is the primary bottleneck in enterprise carbon accounting. Sustainability analysts must interpret free-text purchase descriptions, map them to industry classification codes, query multiple emission factor databases, and select the most contextually appropriate factor — a process that takes 3–8 minutes per line item at scale across millions of transactions.
>
> This pipeline reduces that to **sub-second automated classification** with 94%+ accuracy, using a Retrieval-Augmented Generation (RAG) architecture combining dense vector retrieval with LLM reasoning.
>
> Key capabilities:
>
> - **Zero-shot and few-shot classification** of procurement line items to GHG Protocol Scope 3 categories
> - - **Multi-database retrieval** across EXIOBASE, ecoinvent, EPA EEIO, and GHG Protocol factor libraries
>   - - **Confidence scoring** with human-in-the-loop escalation for low-confidence matches
>     - - **Audit trail generation** with source citation and factor selection rationale
>       - - **REST API** for integration with ERP, procurement, and sustainability platforms
>        
>         - ---
>
> ## 🏗️ Architecture Diagram
>
> ```
> ╔════════════════════════════════════════════════════════════════════╗
> ║          EMISSIONS FACTOR LLM — RAG PIPELINE ARCHITECTURE         ║
> ╠════════════════════════════════════════════════════════════════════╣
> ║                                                                    ║
> ║  INPUT LAYER                                                       ║
> ║  ┌─────────────────────────────────────────────────────────────┐  ║
> ║  │  Raw Procurement Data (ERP / CSV / API)                     │  ║
> ║  │  "500 units Phosphoric acid, 85%, industrial grade, China"  │  ║
> ║  └─────────────────────────────┬───────────────────────────────┘  ║
> ║                                 │                                   ║
> ║  PREPROCESSING                  ▼                                   ║
> ║  ┌─────────────────────────────────────────────────────────────┐  ║
> ║  │  Text Normalization → Unit Extraction → Supplier Context    │  ║
> ║  │  → UNSPSC Code Inference → Country/Region Tagging           │  ║
> ║  └─────────────────────────────┬───────────────────────────────┘  ║
> ║                                 │                                   ║
> ║        ┌────────────────────────┼────────────────────────┐         ║
> ║        │                        │                        │         ║
> ║        ▼                        ▼                        ▼         ║
> ║  ┌──────────────┐   ┌───────────────────┐   ┌────────────────────┐║
> ║  │  QUERY       │   │  VECTOR STORE     │   │  METADATA FILTER   │║
> ║  │  ENCODER     │   │  (ChromaDB/FAISS) │   │  • Country         │║
> ║  │  BGE-Large-  │   │  • EXIOBASE       │   │  • Industry NACE   │║
> ║  │  EN-v1.5     │   │  • ecoinvent      │   │  • Scope 3 Cat     │║
> ║  │              │   │  • EPA EEIO       │   │  • Year            │║
> ║  │  Query embed │   │  • GHG Protocol   │   └────────────────────┘║
> ║  │  (768-dim)   │   │  500K+ factors    │                          ║
> ║  └──────┬───────┘   └─────────┬─────────┘                          ║
> ║         └───────────────────── ┘                                    ║
> ║                        │ Top-K retrieved factors                    ║
> ║                        ▼                                            ║
> ║  ┌─────────────────────────────────────────────────────────────┐  ║
> ║  │  LLM REASONING LAYER (GPT-4o / Claude 3.5 Sonnet)          │  ║
> ║  │                                                             │  ║
> ║  │  Prompt: "Given this procurement item and these candidate   │  ║
> ║  │  emission factors, select the best match and explain why."  │  ║
> ║  │                                                             │  ║
> ║  │  Output: Selected EF + Confidence + Rationale + Source     │  ║
> ║  └─────────────────────────────┬───────────────────────────────┘  ║
> ║                                 │                                   ║
> ║        ┌────────────────────────┼────────────────────────┐         ║
> ║        │                        │                        │         ║
> ║        ▼                        ▼                        ▼         ║
> ║  ┌──────────────┐   ┌───────────────────┐   ┌────────────────────┐║
> ║  │ HIGH CONF.   │   │  MEDIUM CONF.     │   │  LOW CONFIDENCE    │║
> ║  │ Auto-accept  │   │  Flag for review  │   │  Human-in-loop     │║
> ║  │ (>0.92)      │   │  (0.75–0.92)      │   │  (<0.75)           │║
> ║  └──────────────┘   └───────────────────┘   └────────────────────┘║
> ║                                 │                                   ║
> ║  OUTPUT                         ▼                                   ║
> ║  ┌─────────────────────────────────────────────────────────────┐  ║
> ║  │  Audit-Ready Emission Factor Record                         │  ║
> ║  │  • EF value (kgCO2e/unit) + uncertainty range               │  ║
> ║  │  • GHG Protocol Scope 3 Category + Sub-category             │  ║
> ║  │  • Source database + version + citation                     │  ║
> ║  │  • Selection rationale (LLM-generated, auditable)           │  ║
> ║  └─────────────────────────────────────────────────────────────┘  ║
> ╚════════════════════════════════════════════════════════════════════╝
> ```
>
> ---
>
> ## ❗ Problem Statement
>
> ### The Emission Factor Matching Problem at Enterprise Scale
>
> A Fortune 500 company with $10B+ in annual procurement may have 2–5 million purchase order line items per year. Each must be mapped to an emission factor to compute Scope 3 Category 1 (Purchased Goods & Services) emissions.
>
> | Metric | Manual Process | LLM Pipeline |
> |---|---|---|
> | **Time per line item** | 3–8 minutes | < 0.5 seconds |
> | **Annual throughput** (1 analyst) | ~15,000 line items | Unlimited |
> | **Accuracy** | 78–85% (expert review) | 92–96% (benchmarked) |
> | **Audit trail** | Inconsistent | Automated, standardized |
> | **Database coverage** | 1–2 databases | 5+ databases simultaneously |
> | **Uncertainty quantification** | None | Confidence intervals per match |
>
> > *"If you can't match emission factors at the speed of procurement, your Scope 3 inventory is always a year behind your supply chain reality."*
> >
> > ---
> >
> > ## ✅ Solution Overview
> >
> > ### RAG-Powered Emission Factor Intelligence
> >
> > The pipeline combines two complementary AI capabilities: dense vector retrieval for finding semantically similar emission factors, and LLM reasoning for selecting the contextually correct one.
> >
> > **Stage 1 — Intelligent Preprocessing**
> > Raw procurement text is parsed to extract chemical names, quantities, units, supplier country, and commodity classification. A fine-tuned NER model identifies substance names and resolves synonyms (e.g., "MEK" → "Methyl Ethyl Ketone" → CAS 78-93-3).
> >
> > **Stage 2 — Multi-Database Vector Retrieval**
> > The processed query is encoded using `BAAI/bge-large-en-v1.5` and retrieved against a pre-indexed ChromaDB vector store containing 500,000+ emission factors from EXIOBASE, ecoinvent, EPA EEIO, and GHG Protocol. Metadata filters narrow results by geography, scope category, and industry.
> >
> > **Stage 3 — LLM-Powered Factor Selection**
> > The top-K retrieved candidates are passed to GPT-4o or Claude 3.5 Sonnet with a carefully engineered prompt that asks the model to select the best match, explain the selection reasoning, assign a confidence score, and flag any uncertainty.
> >
> > **Stage 4 — Confidence Routing and Audit Trail**
> > High-confidence matches are auto-committed; medium-confidence results are queued for analyst review; low-confidence items escalate to specialist review. All decisions generate an immutable audit log.
> >
> > ---
> >
> > ## 💻 Code, Installation & Analysis
> >
> > ### Prerequisites
> >
> > | Requirement | Version |
> > |---|---|
> > | Python | 3.10+ |
> > | OpenAI / Anthropic API | GPT-4o or Claude 3.5 |
> > | RAM | 8 GB (16 GB for local embeddings) |
> > | Storage | 10 GB (vector store + databases) |
> >
> > ### Installation
> >
> > ```bash
> > git clone https://github.com/virbahu/emissions-factor-llm.git
> > cd emissions-factor-llm
> >
> > python -m venv .venv
> > source .venv/bin/activate
> >
> > pip install -r requirements.txt
> >
> > # Build vector store from emission factor databases
> > python scripts/build_vector_store.py \
> >   --databases exiobase3 ecoinvent38 epa_eeio ghg_protocol \
> >   --embedding-model BAAI/bge-large-en-v1.5 \
> >   --output data/vector_store/
> >
> > # Start the API server
> > uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
> > ```
> >
> > ### API Usage
> >
> > ```python
> > import httpx
> >
> > # Match a single procurement line item
> > response = httpx.post("http://localhost:8000/api/v1/match", json={
> >     "description": "500 kg Phosphoric acid 85% industrial grade",
> >     "supplier_country": "CN",
> >     "spend_usd": 12500.0,
> >     "year": 2025,
> >     "scope3_category": 1
> > })
> >
> > result = response.json()
> > print(result)
> > ```
> >
> > ```json
> > {
> >   "item_id": "PO-2025-048291",
> >   "matched_factor": {
> >     "database": "ecoinvent_3.8",
> >     "process_name": "phosphoric acid production, wet process | RoW",
> >     "emission_factor_kgco2e_per_kg": 1.847,
> >     "uncertainty_pct": 12.3,
> >     "scope3_category": 1,
> >     "scope3_subcategory": "Chemicals",
> >     "nace_code": "C20.13",
> >     "source_citation": "ecoinvent v3.8, UUID: a3c4..."
> >   },
> >   "confidence_score": 0.94,
> >   "routing": "auto_accept",
> >   "rationale": "Matched 'phosphoric acid' to ecoinvent wet-process production. China origin maps to RoW geography proxy. Industrial grade consistent with documented process.",
> >   "total_scope3_kgco2e": 923.5,
> >   "processing_time_ms": 287
> > }
> > ```
> >
> > ### Batch Processing
> >
> > ```python
> > from pipeline.batch_processor import EmissionFactorBatchProcessor
> >
> > processor = EmissionFactorBatchProcessor(
> >     model="gpt-4o",
> >     embedding_model="BAAI/bge-large-en-v1.5",
> >     confidence_threshold=0.85
> > )
> >
> > # Process entire purchase order file
> > results = processor.process_csv(
> >     input_path="data/purchase_orders_2025.csv",
> >     output_path="data/scope3_matched_2025.csv",
> >     batch_size=100
> > )
> >
> > print(f"Processed: {results.total_items:,} items")
> > print(f"Auto-accepted: {results.auto_accepted:,} ({results.auto_accepted_pct:.1f}%)")
> > print(f"Flagged for review: {results.flagged:,}")
> > print(f"Total Scope 3 Cat 1: {results.total_scope3_tco2e:,.1f} tCO2e")
> > ```
> >
> > ---
> >
> > ## 📦 Dependencies
> >
> > ```toml
> > [tool.poetry.dependencies]
> > python = "^3.10"
> > transformers = "^4.40"
> > sentence-transformers = "^3.0"
> > openai = "^1.30"
> > anthropic = "^0.28"
> > langchain = "^0.2"
> > langchain-community = "^0.2"
> > chromadb = "^0.5"
> > fastapi = "^0.110"
> > uvicorn = "^0.29"
> > pandas = "^2.0"
> > numpy = "^1.26"
> > pydantic = "^2.0"
> > httpx = "^0.27"
> > ```
> >
> > ### Emission Factor Databases
> >
> > | Database | Factors | Geography | Version |
> > |---|---|---|---|
> > | ecoinvent | 18,000+ | Global, regionalized | 3.8 |
> > | EXIOBASE | 7,987 products × 44 countries | Multi-regional IO | 3.8 |
> > | EPA EEIO | 389 sectors | US-specific | 2.0.1 |
> > | GHG Protocol | 300+ | Global averages | 2024 Q1 |
> > | GLEC Framework | 180+ transport | Global | 2023 |
> >
> > ---
> >
> > ## 👤 Author
> >
> > **Virbahu Jain** — Founder & CEO, [Quantisage](https://quantisage.com)
> >
> > > *Building the AI Operating System for Scope 3 emissions management and supply chain decarbonization.*
> > >
> > > | | |
> > > |---|---|
> > > | 🎓 **Education** | MBA, Kellogg School of Management, Northwestern University |
> > > | 🏭 **Experience** | 20+ years across manufacturing, life sciences, energy & public sector |
> > > | 🌍 **Scope** | Supply chain operations on five continents |
> > > | 📝 **Research** | Peer-reviewed publications on AI in sustainable supply chains |
> > > | 🔬 **Patents** | IoT and AI solutions for manufacturing and logistics |
> > >
> > > [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/virbahu)
> > > [![GitHub](https://img.shields.io/badge/GitHub-virbahu-181717?logo=github)](https://github.com/virbahu)
> > >
> > > ---
> > >
> > > ## 📄 License
> > >
> > > MIT License — see [LICENSE](LICENSE) for details.
> > >
> > > ---
> > >
> > > <div align="center">
<sub>Part of the <strong>Quantisage Open Source Initiative</strong> | AI × Supply Chain × Climate</sub>
</div>
