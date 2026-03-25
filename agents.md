# MXene-Informatics — Planned Agents (Layer 4: Agentic Interface)

## Status: NOT STARTED — Future Work

---

## Agent 1: XPS Reference Assignment Agent

**Purpose:** Dynamically assign literature references to XPS chemical state assignments instead of hardcoded references.

**Problem:**
- Current `XPS_REFERENCES` in `src/analysis/xps_analysis.py` has hardcoded references (DOI links) tied to Ti₃C₂Tₓ MXene
- If the pipeline is used for a different material (V₂CTₓ, Mo₂CTₓ, Nb₂CTₓ), the same references would incorrectly appear
- References should be assigned dynamically based on the actual data and material system

**Proposed Workflow:**
1. Peak fitting runs → identifies peaks at specific binding energies
2. Agent receives: element, binding energy, chemical state assignment
3. Agent queries:
   - **NIST XPS Database** (srdata.nist.gov/xps) for matching BE values
   - **Literature corpus** (RAG over MXene papers) for papers reporting similar assignments
   - **CrossRef API** for DOI resolution and citation metadata
4. Agent returns: top 1-3 matching references with DOI links, ranked by relevance
5. References are displayed in the dashboard with clickable links

**Technical Requirements:**
- RAG pipeline: embed MXene literature PDFs → vector store (FAISS/ChromaDB)
- NIST XPS API access or scraped reference database
- LLM (Claude API or local) for reasoning over matches
- Fallback: use hardcoded references if agent fails

**Priority:** Medium — enhances credibility and extensibility but not blocking for current paper

---

## Agent 2: Literature Q&A Agent (RAG)

**Purpose:** Natural language querying over MXene literature and experimental data.

**Proposed Capabilities:**
- "What is the typical c-lattice parameter for HF-etched Ti₃C₂Tₓ?"
- "Which papers report Seebeck coefficient for MXene composites?"
- "Compare my XRD pattern with published Ti₃C₂Tₓ patterns"

**Technical Requirements:**
- PDF ingestion pipeline for thesis + published papers
- Embedding model (sentence-transformers or OpenAI)
- Vector store (FAISS or ChromaDB)
- LLM for answer generation with citations
- Streamlit chat interface

**Priority:** Low — nice-to-have for the dashboard, not needed for paper

---

## Agent 3: Synthesis Optimization Agent

**Purpose:** Suggest optimal synthesis parameters based on characterization results.

**Proposed Capabilities:**
- Analyze XRD phase purity → suggest etching time/temperature adjustments
- Analyze XPS termination ratios → suggest post-treatment modifications
- Compare with literature synthesis conditions

**Technical Requirements:**
- Synthesis parameter database from literature
- Decision logic or ML model linking parameters → outcomes
- LLM for natural language recommendations

**Priority:** Low — aspirational, requires substantial literature data collection

---

## Implementation Notes
- All agents should be in `src/agents/` directory
- Each agent should have a standalone test script
- Agents should gracefully degrade (return hardcoded fallback if API/LLM unavailable)
- Dashboard integration via new "AI Assistant" page in app.py
