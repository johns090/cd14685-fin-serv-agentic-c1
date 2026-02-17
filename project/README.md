# 🏦 Financial Services Agentic AI Project – SAR Processing System

This README guides you through the student-facing `starter/` workspace for building an AI-powered Suspicious Activity Report (SAR) processing system with multi-agent Chain-of-Thought (Risk Analyst) and ReACT (Compliance Officer) workflows.

## 📋 Project Instructions

- **What you build:** Two cooperating agents that detect suspicious activity and generate regulatory-compliant SAR narratives with human-in-the-loop decision gates.
- **Architecture:** `CSV data → Risk Analyst Agent (CoT) → Human review → Compliance Officer Agent (ReACT) → SAR filing` (see `project/PROJECT_DESCRIPTION.md` for the full brief).
- **Where to work:** All implementation happens in [`starter/src/`](starter/src:1) with notebooks in [`starter/notebooks/`](starter/notebooks:1) for guided development.

## 📂 Directory Structure (student view)

```
project/
├── PROJECT_DESCRIPTION.md          # Full assignment brief
├── README.md                       # This file
└── starter/                        # Your working directory
    ├── README.md                   # Detailed student guide
    ├── .env.template               # Vocareum/OpenAI env vars
    ├── requirements.txt            # Python dependencies
    ├── data/                       # customers/accounts/transactions CSVs
    ├── notebooks/                  # 01, 02, 03 notebooks (per phase)
    ├── src/                        # Implement your code here
    │   ├── foundation_sar.py
    │   ├── risk_analyst_agent.py
    │   └── compliance_officer_agent.py
    ├── tests/                      # Pytest suites (skip until implemented)
    └── outputs/                    # Audit logs & generated SARs
```

## 🚀 Getting Started

1) **Prerequisites**
   - Python 3.8+
   - Vocareum OpenAI API key from Udacity “Cloud Resources” (starts with `voc-`)
   - VS Code + Jupyter (recommended)

2) **Set up environment**
```bash
cd project/starter
pip install -r requirements.txt
cp .env.template .env
# Edit .env and set OPENAI_API_KEY=voc-your-key-here
```

3) **Open notebooks**
   - Phase 1: `notebooks/01_data_exploration.ipynb`
   - Phase 2-3: `notebooks/02_agent_development.ipynb`
   - Phase 4: `notebooks/03_workflow_integration.ipynb`

## 🧠 Implementation Checklist (by phase)

### Phase 1 – Foundation & Data Modeling
- Implement Pydantic schemas, logger, and loader in [`foundation_sar.py`](starter/src/foundation_sar.py:1).
- Validate CSV data, create `CaseData`, and log audit trails.
- Run `python -m pytest tests/test_foundation.py -v`.

### Phase 2 – Risk Analyst Agent (Chain-of-Thought)
- Build CoT prompt + analysis in [`risk_analyst_agent.py`](starter/src/risk_analyst_agent.py:1).
- Classify 5 types (Structuring, Sanctions, Fraud, Money_Laundering, Other) with confidence + key indicators.
- Run `python -m pytest tests/test_risk_analyst.py -v`.

### Phase 3 – Compliance Officer Agent (ReACT)
- Build ReACT prompt + narrative generation in [`compliance_officer_agent.py`](starter/src/compliance_officer_agent.py:1).
- Enforce ≤120-word narratives with citations and completeness checks.
- Run `python -m pytest tests/test_compliance_officer.py -v`.

### Phase 4 – Workflow Integration
- Orchestrate end-to-end flow in `notebooks/03_workflow_integration.ipynb`.
- Human decision gates, SAR JSON outputs, audit logs, cost/efficiency metrics.

## 🧪 Testing

From `project/starter`:
```bash
# Phase-specific
python -m pytest tests/test_foundation.py -v > tests/test_results/foundation_test_results.txt
python -m pytest tests/test_risk_analyst.py -v > tests/test_results/risk_analyst_test_results.txt
python -m pytest tests/test_compliance_officer.py -v tests/test_results/compliance_officer_test_results.txt

# Full suite
python -m pytest tests/ > tests/test_results/full_test_results.txt
```
Notes:
- Tests skip if a module isn’t implemented yet (expected during development).
- Focus on one phase at a time and use failures as guidance.


### Test Coverage

**30 comprehensive tests** across 3 modules:
- **Foundation SAR (10 tests)**: Data validation, case creation, audit logging
- **Risk Analyst Agent (10 tests)**: Agent initialization, case analysis, error handling  
- **Compliance Officer Agent (10 tests)**: Narrative generation, regulatory compliance

### Test Results

Complete test results and validation proof available in:
- `project/starter/test_results/` - Consolidated test results 
- All tests pass with 100% success rate (3.17s execution time)
- Validates production readiness and regulatory compliance

### Break Down Tests

**Foundation Tests**: Validate core data structures and utilities
- Customer/Account/Transaction data validation
- Case aggregation and schema compliance
- CSV data loading and audit logging

**Risk Analyst Tests**: Validate Chain-of-Thought analysis workflow  
- OpenAI API integration and response parsing
- JSON extraction from various response formats
- Error handling for malformed responses

**Compliance Officer Tests**: Validate ReACT regulatory narrative generation
- 120-word narrative limit enforcement
- Regulatory citation and terminology validation
- Multi-format response parsing and validation
## Project Instructions

This section should contain all the student deliverables for this project.

## ✅ Success Criteria

- Schemas validate provided CSV data; DataLoader builds consistent `CaseData` objects.
- Risk Analyst outputs structured JSON with 5 classifications, confidence scores, key indicators, and risk levels.
- Compliance Officer outputs ≤120-word narratives with citations and completeness checks.
- Audit logs captured; human gates demonstrated; SAR documents generated in `outputs/filed_sars/`.


## 🛠️ Built With

- Python, Pydantic, pandas, OpenAI API, python-dotenv
- Jupyter, pytest, matplotlib/seaborn (optional for exploration)

## 📜 License

This project is covered by the root [`LICENSE.md`](../LICENSE.md:1).
