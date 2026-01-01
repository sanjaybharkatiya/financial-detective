# Technical Design Document: The Financial Detective

**Version:** 2.2  
**Date:** January 2025  
**Author:** Architecture Team  
**Status:** Final

---

## 1. Executive Summary

The Financial Detective is an LLM-powered extraction pipeline that transforms unstructured financial text—such as excerpts from public company annual reports and enterprise financial disclosures—into a validated, structured Knowledge Graph. The system addresses the fundamental challenge of extracting entities (companies, risk factors, monetary amounts) and their relationships from natural language financial disclosures without relying on brittle regex or rule-based pattern matching.

The solution leverages pluggable LLM providers (OpenAI GPT-4o, Google Gemini, and local Ollama models) with a carefully designed prompt that enforces JSON-only output conforming to a strict Pydantic schema. Provider selection is driven by environment variables and implemented via a factory pattern, enabling seamless switching between cloud and local inference without code changes. All extraction is performed through LLM reasoning; no post-processing, cleanup, or pattern matching is applied. The output is a validated JSON Knowledge Graph accompanied by visual graph representations (Mermaid diagrams with interactive HTML viewer featuring dark theme, zoom/pan controls, and optional NetworkX PNG) for human verification.

**Key Constraints:**
- No regex or pattern-based extraction
- All entity and relationship extraction via LLM reasoning
- Strict schema enforcement: optional fields allowed only if explicitly defined in schema (e.g., `confidence`, `context`); undocumented fields remain forbidden via `extra="forbid"`
- Auto-repair for invalid relationships; removal of orphan/meaningless nodes
- Fail-fast on invalid data; no silent error suppression

---

## 2. Problem Statement

### The Challenge

Financial reports contain critical structured information embedded in unstructured prose:

- **Ownership structures** between corporate entities
- **Risk disclosures** tied to specific business units
- **Monetary figures** associated with revenue, liabilities, and investments

This information is essential for compliance analysis, investment research, and regulatory reporting. However, it exists in narrative form with:

- Inconsistent terminology across documents
- Complex sentence structures with nested clauses
- Cross-references and coreference (pronouns referring to earlier entities)
- Implicit relationships requiring contextual inference

### Why Regex Fails

Rule-based extraction using regular expressions is fundamentally unsuitable for this domain:

| Challenge | Regex Limitation |
|-----------|------------------|
| **Phrasing Variation** | "$1.2 billion," "1,200 million dollars," "revenue of $9.5 billion" require separate patterns |
| **Contextual Meaning** | "The company" requires coreference resolution to identify the referent |
| **Negation Handling** | "No material risk was identified" creates false positives for naive patterns |
| **Relationship Extraction** | Multi-sentence relationships ("Parent owns Subsidiary. Subsidiary reported...") cannot be captured |
| **Maintenance Burden** | Each new document format requires pattern updates |

Regex approaches exhibit O(n) maintenance cost where n is the number of document variations encountered. This is unsustainable for production financial analysis.

### Design Decision

The system uses LLM-based extraction exclusively. This trades computational cost (API calls) for:

1. Zero maintenance of extraction rules
2. Generalization across phrasing variations
3. Semantic understanding of relationships
4. Contextual disambiguation

---

## 3. High-Level Architecture

### Pipeline Overview

The system follows a strict linear pipeline with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE FLOW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────┐
  │ raw_report  │
  │   .txt      │
  └──────┬──────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                        input_loader.py                          │
  │  • Read file as single string                                   │
  │  • Preserve all formatting                                      │
  │  • No parsing, cleaning, or normalization                       │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                      extractor/ package                         │
  │  ┌───────────────────────────────────────────────────────────┐  │
  │  │  factory.py: Environment-driven provider selection        │  │
  │  │    ├── OpenAI GPT-4o (cloud)                              │  │
  │  │    ├── Google Gemini (cloud)                              │  │
  │  │    └── Ollama (local)                                     │  │
  │  ├───────────────────────────────────────────────────────────┤  │
  │  │  __init__.py: Chunking orchestration                      │  │
  │  │    ├── chunker.py: Text splitting                         │  │
  │  │    └── graph_merger.py: Merge & deduplicate               │  │
  │  ├───────────────────────────────────────────────────────────┤  │
  │  │  All providers share:                                     │  │
  │  │  • Unified system prompt with 18 relation types           │  │
  │  │  • Anti-hallucination instructions                        │  │
  │  │  • JSON-only output enforcement                           │  │
  │  │  • Temperature = 0 for determinism                        │  │
  │  │  • JSON repair and relation normalization                 │  │
  │  └───────────────────────────────────────────────────────────┘  │
  │  • Parse JSON response                                          │
  │  • Validate against Pydantic schema                             │
  │  • Save intermediate results after each chunk                   │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
                                 ▼
                        ┌───────────────────┐
                        │  KnowledgeGraph   │
                        │    (Pydantic)     │
                        └─────────┬─────────┘
                                  │
                                  ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                         validator.py                            │
  │  • Check: at least one node exists                              │
  │  • Check: all node IDs are unique                               │
  │  • Auto-repair: remove invalid relationships                    │
  │  • Remove orphan nodes (no connections)                         │
  │  • Raise ValueError on critical violations                      │
  └──────────────────────────────┬──────────────────────────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
  ┌───────────────────────┐             ┌───────────────────────┐
  │   graph_output.json   │             │  visualizer_mermaid   │
  │                       │             │  • Mermaid .mmd file  │
  │  Serialized via       │             │  • Interactive HTML   │
  │  model_dump_json()    │             │  • Dark theme + zoom  │
  └───────────────────────┘             └───────────┬───────────┘
                                                    │
                                                    ▼
                                        ┌───────────────────────┐
                                        │  visuals/graph.html   │
                                        │  visuals/graph.mmd    │
                                        └───────────────────────┘
```

### Component Isolation

Each module has a single responsibility:

| Module | Input | Output | Side Effects |
|--------|-------|--------|--------------|
| `input_loader.py` | File path | Raw string | None |
| `chunker.py` | Raw string | List of chunks | None |
| `graph_merger.py` | List of KnowledgeGraphs | Single KnowledgeGraph | None |
| `extractor/` | Raw string | KnowledgeGraph | LLM API call (provider-dependent) |
| `validator.py` | KnowledgeGraph | Repaired KnowledgeGraph | None |
| `visualizer.py` | KnowledgeGraph | None | Writes PNG file (optional on Python 3.14) |
| `visualizer_mermaid.py` | KnowledgeGraph | None | Writes .mmd and .html files (always available) |
| `main.py` | None | Exit code | Orchestrates pipeline, opens browser |

This separation enables independent testing, clear error attribution, and future component replacement without system-wide changes.

---

## 4. Entity & Relationship Modeling

### Knowledge Graph Model

The domain model captures three entity types and eighteen relationship types, chosen to represent the comprehensive information structure of financial disclosures:

```
                    ┌─────────────────┐
                    │     Company     │
                    │  (Organization) │
                    └────────┬────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     │                       │                       │
     ▼                       ▼                       ▼
┌──────────┐           ┌──────────┐           ┌──────────┐
│ Financial│           │   Risk   │           │Ownership │
│ Relations│           │ Relations│           │ Relations│
└────┬─────┘           └────┬─────┘           └────┬─────┘
     │                      │                      │
     ▼                      ▼                      ▼
REPORTS_AMOUNT         HAS_RISK               OWNS
RAISED_CAPITAL         IMPACTED_BY            OPERATES
INVESTED_IN            DECLINED_DUE_TO        PARTNERED_WITH
COMMITTED_CAPEX        SUPPORTED_BY           JOINT_VENTURE_WITH
TARGETS                SUBJECT_TO
PLANS_TO               COMPLIES_WITH
ON_TRACK_TO
COMMITTED_TO
```

### Node Types

| Type | Description | Examples |
|------|-------------|----------|
| **Company** | Corporate entities, subsidiaries, or business units | "Parent Corporation," "Subsidiary Holdings," "Retail Division" |
| **RiskFactor** | Disclosed risks, uncertainties, or adverse conditions | "Currency fluctuation risk," "Regulatory compliance risk" |
| **DollarAmount** | Monetary values with context explaining what they represent | "$1.2 billion (Revenue for FY 2024)," "€45 million (CAPEX commitment)" |

### Relationship Types (18 Total)

| Category | Relationships | Semantics |
|----------|--------------|-----------|
| **Financial** | REPORTS_AMOUNT, RAISED_CAPITAL, INVESTED_IN, COMMITTED_CAPEX | Entity associated with monetary figures |
| **Ownership** | OWNS, OPERATES | Parent-subsidiary or operational control |
| **Partnerships** | PARTNERED_WITH, JOINT_VENTURE_WITH | Business partnerships and JVs |
| **Risk** | HAS_RISK, IMPACTED_BY, DECLINED_DUE_TO, SUPPORTED_BY | Entity exposure to risk factors |
| **Strategy** | TARGETS, PLANS_TO, ON_TRACK_TO, COMMITTED_TO | Forward-looking statements |
| **Compliance** | COMPLIES_WITH, SUBJECT_TO | Regulatory and legal constraints |

### Rationale for Schema Design

The schema uses a balanced set of types:

1. **Three Entity Types** — Clear classification boundaries reduce LLM ambiguity
2. **Eighteen Relationship Types** — Comprehensive coverage of financial disclosure patterns
3. **Optional Context Field** — Nodes can include explanatory context (e.g., what a dollar amount represents)
4. **Optional Confidence Field** — Relationships may include confidence scores for downstream filtering
5. **Strict Validation** — `extra="forbid"` prevents schema drift

---

## 5. JSON Schema Design

### Why Pydantic v2

Pydantic v2 was selected for schema definition based on:

| Criterion | Pydantic v2 Advantage |
|-----------|----------------------|
| **Type Safety** | Native Python type hints with runtime validation |
| **JSON Schema Generation** | `model_json_schema()` produces OpenAPI-compatible schema |
| **Strict Mode** | `extra="forbid"` rejects unexpected fields |
| **Serialization** | `model_dump_json()` for consistent output |
| **Ecosystem** | Standard in FastAPI and modern Python tooling |

### Schema Definition

```python
class Node(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    type: Literal["Company", "RiskFactor", "DollarAmount"]
    name: str
    context: str | None = None  # Optional: explains what the node represents

class Relationship(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    target: str
    relation: Literal[
        "OWNS", "HAS_RISK", "REPORTS_AMOUNT", "OPERATES",
        "IMPACTED_BY", "DECLINED_DUE_TO", "SUPPORTED_BY",
        "PARTNERED_WITH", "JOINT_VENTURE_WITH",
        "RAISED_CAPITAL", "INVESTED_IN", "COMMITTED_CAPEX",
        "TARGETS", "PLANS_TO", "ON_TRACK_TO", "COMMITTED_TO",
        "COMPLIES_WITH", "SUBJECT_TO"
    ]
    confidence: float | None = None  # Optional: 0.0-1.0

class KnowledgeGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    nodes: list[Node]
    relationships: list[Relationship]
```

### Strictness as Contract

The `extra="forbid"` configuration serves as a contract between the LLM and the system:

1. **No Schema Drift** — LLM cannot introduce unexpected fields
2. **Fail-Fast Validation** — Invalid outputs are rejected immediately
3. **Deterministic Parsing** — Same schema applies to extraction and downstream consumers

This strictness is intentional: we prefer a failed extraction over a partially correct one with unvalidated fields.

### Optional Fields

The schema supports two optional fields:

| Field | Location | Purpose |
|-------|----------|---------|
| `context` | Node | Explains what the node represents (e.g., "Revenue for FY 2024") |
| `confidence` | Relationship | Confidence score (0.0–1.0) for downstream filtering |

Both fields are explicitly defined in the schema; the `extra="forbid"` constraint still prevents any undocumented fields.

---

## 6. Multi-Provider LLM Extraction Strategy

The system supports multiple LLM providers through a factory pattern, enabling environment-driven provider selection without code changes.

### Provider Overview

| Provider | Model | Use Case |
|----------|-------|----------|
| **OpenAI** | GPT-4o | Production-grade cloud extraction with strong instruction following |
| **Google Gemini** | gemini-2.0-flash | Cloud extraction with large context windows (1M+ tokens) |
| **Ollama** | llama3:latest, qwen2.5:7b, Mistral, etc. | Local inference, offline capable, no API costs |

### OpenAI GPT-4o

GPT-4o was selected as the default cloud provider based on:

| Factor | Consideration |
|--------|---------------|
| **Instruction Following** | Superior adherence to JSON-only output constraints |
| **Context Window** | 128K tokens accommodates large financial documents |
| **Reasoning Quality** | Strong performance on entity disambiguation |
| **Availability** | Production-grade API with SLA |

### Google Gemini

Gemini offers an alternative cloud option with:

| Factor | Consideration |
|--------|---------------|
| **Context Window** | Up to 1M tokens for very large documents |
| **Model Flexibility** | Multiple models (gemini-2.0-flash, gemini-1.5-pro) configurable via environment |
| **JSON Mode** | Native `response_mime_type="application/json"` enforcement |

### Ollama (Local Inference)

Ollama enables fully local, offline extraction:

| Factor | Consideration |
|--------|---------------|
| **Privacy** | Documents never leave the local machine |
| **Cost** | No API fees; unlimited extractions |
| **Latency** | Network-independent; suitable for air-gapped environments |

**Recommended Ollama Models:**

| Model | Strengths | Use Case |
|-------|-----------|----------|
| `llama3:latest` | Good general-purpose, fast inference | Default choice for most extractions |
| `qwen2.5:7b` | Excellent structured JSON output, strong instruction following | Recommended for knowledge graph extraction |
| `mistral` | Lightweight, fast | Quick testing and development |

### Shared Prompt Constraints

All providers share identical prompt constraints and schema validation:

- Unified system prompt with 18 relationship types
- Ownership direction rules (subsidiary cannot own parent)
- Anti-hallucination instructions
- JSON-only output enforcement
- Temperature = 0 for determinism
- Context requirement for every node
- JSON repair and relation normalization functions

### Prompt Design Principles

The system prompt follows four principles:

#### 1. Anti-Hallucination

```
DO NOT hallucinate or infer information not explicitly stated in the text.
DO NOT add entities or relationships that are not directly mentioned.
ONLY extract information that is explicitly present in the provided text.
```

#### 2. JSON-Only Output

```
Output ONLY valid JSON matching the schema below—no explanations, no markdown.
Respond with ONLY the JSON object. No additional text.
```

#### 3. Ownership Rules

```
OWNERSHIP RULES:
- Subsidiary or JV cannot own parent.
- If A is subsidiary/JV of B, relation is B → A.
- If unclear, use PARTNERED_WITH or JOINT_VENTURE_WITH.
```

#### 4. Deterministic Behavior

```python
temperature=0
```

### JSON Repair and Normalization

All providers include robust error handling:

1. **JSON Extraction** — Strips markdown code fences and preamble text
2. **JSON Repair** — Fixes common syntax errors (missing colons, commas)
3. **Relation Normalization** — Maps invalid relation types to valid ones (e.g., "SUBSIDIARY" → "OWNS")

---

## 7. Large Document Chunking

The system supports automatic chunking for documents that exceed LLM context windows.

### Architecture

When chunking is enabled and text exceeds the configured chunk size:

```
┌─────────────┐
│  Raw Text   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         chunker.py                               │
│  • estimate_tokens(text) — chars/4 approximation                │
│  • split_text(text, chunk_size, overlap)                        │
│  • Paragraph-aware splitting with sentence fallback             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐     ┌──────────┐     ┌──────────┐
        │ Chunk 1  │     │ Chunk 2  │     │ Chunk N  │
        └────┬─────┘     └────┬─────┘     └────┬─────┘
             │                │                │
             ▼                ▼                ▼
        ┌──────────┐     ┌──────────┐     ┌──────────┐
        │ Extract  │     │ Extract  │     │ Extract  │
        │ Graph 1  │     │ Graph 2  │     │ Graph N  │
        └────┬─────┘     └────┬─────┘     └────┬─────┘
             │                │                │
             │     Save intermediate results   │
             └────────────────┼────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       graph_merger.py                            │
│  • merge_graphs(graphs) — combine nodes/relationships           │
│  • Deduplicate nodes by (name, type) key                        │
│  • Renumber IDs to prevent collisions                           │
│  • Preserve schema_version from first graph                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                      ┌───────────────────┐
                      │  KnowledgeGraph   │
                      │   (Unified)       │
                      └───────────────────┘
```

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `CHUNK_ENABLED` | Enable/disable chunking | `true` |
| `CHUNK_SIZE_TOKENS` | Target tokens per chunk | `4000` |
| `CHUNK_OVERLAP_TOKENS` | Overlap between chunks | `200` |

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Token estimation via chars/4** | Avoids external tokenizer dependency; sufficient for chunking purposes |
| **Paragraph-aware splitting** | Preserves semantic coherence within chunks |
| **Configurable overlap** | Preserves context for entities spanning chunk boundaries |
| **Node deduplication** | Merges duplicate entities by (name.lower(), type) key |
| **ID renumbering** | Generates globally unique IDs (company_1, amount_1, etc.) |
| **Continue on chunk error** | Processing continues even if individual chunks fail |
| **Iterative saves** | Results saved after each chunk for live progress viewing |

### Constraints

The chunking implementation adheres to strict constraints:

1. **No schema changes** — KnowledgeGraph model unchanged
2. **No cross-chunk inference** — Chunks processed independently
3. **Deduplication by name/type** — Similar entities merged across chunks
4. **Backward compatible** — `CHUNK_ENABLED=false` preserves original behavior

---

## 8. Validation & Reliability

### Two-Layer Validation

The system implements validation at two distinct levels:

#### Layer 1: Schema Validation (Pydantic)

Enforces structural correctness:
- All required fields present
- Field types match (str, Literal enum values)
- No extra fields (`extra="forbid"`)

This validation occurs during `KnowledgeGraph.model_validate()`.

#### Layer 2: Graph Integrity Validation with Auto-Repair

Enforces semantic correctness with automatic repair:
- At least one node exists
- All node IDs are unique
- Removes relationships referencing non-existent nodes
- Removes relationships violating type constraints (e.g., HAS_RISK targeting a Company)
- Removes orphan nodes (nodes with no relationships)

This validation occurs in `validator.py` after Pydantic parsing succeeds.

### Auto-Repair Behavior

Instead of failing on invalid relationships, the system automatically removes them:

| Violation | Action |
|-----------|--------|
| HAS_RISK target is not RiskFactor | Remove relationship |
| REPORTS_AMOUNT target is not DollarAmount | Remove relationship |
| OWNS between non-Company nodes | Remove relationship |
| Relationship references non-existent node | Remove relationship |
| Node has no relationships | Remove node (orphan cleanup) |

### Fail-Fast Philosophy

The system surfaces critical errors immediately:

| Condition | Behavior |
|-----------|----------|
| Missing `OPENAI_API_KEY` | `ValueError` raised before API call |
| LLM returns non-JSON | `json.JSONDecodeError` raised |
| JSON doesn't match schema | Pydantic `ValidationError` raised |
| Duplicate node IDs | `ValueError` with specific IDs listed |
| No nodes extracted | `ValueError` raised |

---

## 9. Visualization Strategy

The system produces three visualization outputs: Mermaid diagrams with interactive HTML viewer (always available), and optional NetworkX PNG.

### Technology Selection

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Mermaid Diagrams** | Native text output | Lightweight, CI-friendly, GitHub-renderable |
| **HTML Viewer** | Mermaid.js CDN | Self-contained, dark-themed, interactive with zoom/pan controls |
| **Graph Library** | NetworkX | Standard Python graph library with rich algorithms |
| **PNG Rendering** | matplotlib | Ubiquitous, no external dependencies beyond Python |

### Mermaid Diagram Visualization

Mermaid diagrams are always generated as a first-class output:

| Element | Mermaid Encoding |
|---------|------------------|
| **Company nodes** | Rectangle `["label"]` |
| **RiskFactor nodes** | Rounded `("label")` |
| **DollarAmount nodes** | Parallelogram `[/"label"/]` |
| **Relationships** | Labeled arrows `-->|RELATION|` |

Features:
- Special character escaping for Mermaid compatibility
- Context included in node labels where available
- Label truncation for very long text (60 char limit)

### Interactive HTML Viewer

The HTML viewer provides a modern, full-featured experience for exploring Knowledge Graphs of any size:

| Feature | Description |
|---------|-------------|
| **Dark Theme** | GitHub-inspired dark color scheme (`#0d1117`, `#161b22`) with blue accents |
| **Auto-Fit on Load** | Graph automatically scales to fit browser viewport |
| **Zoom Slider** | Continuous zoom control from 5% to 200% |
| **Preset Zoom Buttons** | Quick access to 10%, 25%, 50% zoom levels |
| **Quick-Zoom Panel** | Floating panel with +/−/Fit/Reset buttons |
| **Pan Navigation** | Drag-to-pan, arrow keys, touch support for mobile |
| **Keyboard Shortcuts** | `+/-` zoom, `F` fit, `0` reset, arrows scroll, `Home/End` jump |
| **Fixed Header** | Shows total nodes and relationships count |
| **Fixed Legend** | Always-visible node type shapes reference |
| **Responsive Layout** | Works on various screen sizes and browsers |
| **Mermaid Dark Theme** | Diagram rendered with dark theme variables for visual consistency |

### NetworkX PNG Visualization (Optional)

| Element | Encoding |
|---------|----------|
| **Company nodes** | Blue circles |
| **RiskFactor nodes** | Red circles |
| **DollarAmount nodes** | Green circles |
| **Relationships** | Gray directed edges with labels |
| **Layout** | Spring layout with fixed seed (42) for reproducibility |

**Note:** NetworkX visualization is optional and may be skipped on Python 3.14 due to upstream compatibility issues.

---

## 10. Testing Strategy

### Coverage Overview

| Module | Test File | Tests | Focus Areas |
|--------|-----------|-------|-------------|
| `validator.py` | `test_validator.py` | 21 | Validation rules; auto-repair functionality |
| `extractor/` | `test_extractor.py` | 13 | Delegation; chunking integration; error handling |
| `factory.py` | `test_factory.py` | 12 | OpenAI, Gemini, Ollama provider creation |
| `chunker.py` | `test_chunker.py` | 16 | Text splitting; token estimation; edge cases |
| `graph_merger.py` | `test_graph_merger.py` | 16 | ID renumbering; deduplication; relationship updates |
| `visualizer.py` | `test_visualizer.py` | 4 | NetworkX PNG generation (skipped on Python 3.14) |
| `visualizer_mermaid.py` | `test_visualizer_mermaid.py` | 24 | Mermaid diagrams; HTML generation; escaping |
| `config.py` | `test_config.py` | 20 | Environment variables; defaults; boolean parsing |
| `input_loader.py` | `test_input_loader.py` | 14 | File loading; UTF-8; error handling |
| `schema.py` | `test_schema.py` | 35 | Node, Relationship, KnowledgeGraph validation |
| **Total** | | **185** | **Comprehensive coverage of all modules** |

### Mocking External Calls

LLM API calls are fully mocked in tests:

```python
@patch("src.extractor.factory.create_extractor")
def test_extraction(self, mock_factory):
    mock_extractor = MagicMock()
    mock_extractor.extract.return_value = valid_graph
    mock_factory.return_value = mock_extractor
    # ...
```

This ensures:
- Tests run without API keys
- No cost incurred during CI/CD
- Deterministic test outcomes

---

## 11. Error Handling & Edge Cases

### Error Categories

| Category | Example | Handling |
|----------|---------|----------|
| **Configuration** | Missing `OPENAI_API_KEY` | `ValueError` before any processing |
| **Network** | API timeout | Exception propagates |
| **LLM Output** | Non-JSON response | `json.JSONDecodeError` |
| **Schema Mismatch** | Invalid node type | Pydantic `ValidationError` |
| **Graph Integrity** | Invalid relationships | Auto-removed; processing continues |
| **File System** | Unwritable output path | `OSError` from visualizer |

### Design Principle: Graceful Degradation with Fail-Fast Core

The system balances graceful degradation with fail-fast behavior:

1. **Core Validation Fails Fast** — Missing API keys, empty graphs, duplicate IDs
2. **Relationship Errors Auto-Repair** — Invalid relationships removed, processing continues
3. **Chunk Errors Continue** — Individual chunk failures don't stop entire extraction
4. **Non-Zero Exit** — `main.py` returns exit code 1 on critical errors

---

## 12. Limitations & Assumptions

### Assumptions

| Assumption | Implication |
|------------|-------------|
| **Explicit Relationships** | The system only extracts relationships stated in text; implicit relationships are not inferred |
| **English Language** | Prompts and extraction logic assume English-language documents |
| **Single Document** | Pipeline processes one document per run; batch processing is not implemented |
| **Cloud API Availability** | Production use requires reliable internet and API access (except Ollama) |
| **Schema Sufficiency** | The three node types and eighteen relationship types capture the relevant domain |

### Limitations

| Limitation | Impact | Mitigation Path |
|------------|--------|-----------------|
| **LLM Variability** | Minor output differences across API versions | Pin API version; monitor for drift |
| **Token Limits** | Documents exceeding context limits require chunking | Chunking enabled by default |
| **No Entity Resolution** | Slightly different names may create duplicate nodes within chunks | Post-processing entity resolution layer |
| **Cost per Cloud Extraction** | Each cloud run incurs API cost | Use Ollama for development; caching layer for repeated extractions |
| **No Incremental Updates** | Full re-extraction required for document changes | Delta extraction for document versions |

---

## 13. Future Enhancements

### Near-Term

| Enhancement | Effort | Value |
|-------------|--------|-------|
| **Mermaid → PNG Export** | Low | Generate PNG from Mermaid diagrams for non-browser environments |
| **Provider Benchmarking** | Low | Compare extraction quality across OpenAI, Gemini, and Ollama |
| **Confidence-Based Filtering** | Low | Filter relationships by confidence threshold in output |
| **Enhanced Entity Resolution** | Medium | Fuzzy matching for similar entity names across chunks |
| **Caching Layer** | Low | Reduce API costs for repeated extractions |

### Medium-Term

| Enhancement | Effort | Value |
|-------------|--------|-------|
| **Incremental Extraction** | High | Process document changes without full re-extraction |
| **Batch Document Processing** | Medium | Process multiple documents in a single run |
| **Interactive Graph Editor** | Medium | Browser-based editing of extracted graphs |

### Long-Term

| Enhancement | Effort | Value |
|-------------|--------|-------|
| **Graph Database Persistence** | High | Enable querying and relationship traversal at scale |
| **REST API Layer** | Medium | Integrate with external systems |
| **Multi-Document Knowledge Graph** | High | Build cumulative graph across document corpus |

---

## Appendix A: Module Dependency Graph

```
main.py
    │
    ├── src/input_loader.py
    │
    ├── src/extractor/                    # LLM extractor package
    │       ├── __init__.py               # Package entry point (with chunking orchestration)
    │       │       ├── src/chunker.py    # Text chunking
    │       │       ├── src/graph_merger.py  # Graph merging & deduplication
    │       │       └── src/config.py     # Configuration management
    │       ├── base.py                   # Abstract LLMExtractor interface
    │       ├── factory.py                # Environment-driven provider selection
    │       │       └── src/config.py
    │       ├── openai_llm.py             # OpenAI GPT-4o implementation
    │       │       └── JSON repair & relation normalization
    │       ├── gemini_llm.py             # Google Gemini implementation
    │       │       └── JSON repair & relation normalization
    │       └── ollama_llm.py             # Ollama local implementation
    │               └── JSON repair & relation normalization
    │               └── src/schema.py
    │
    ├── src/validator.py                  # Validation & auto-repair
    │       └── src/schema.py
    │
    ├── src/visualizer.py                 # NetworkX PNG (optional)
    │       └── src/schema.py
    │
    └── src/visualizer_mermaid.py         # Mermaid diagrams + interactive HTML viewer
            └── src/schema.py

clean_graph.py                            # Utility for post-processing cleanup
    └── src/schema.py
    └── src/visualizer_mermaid.py
```

All modules depend on `schema.py` as the single source of truth for data structures. Provider selection is driven by `config.py` which reads environment variables. Chunking is orchestrated in `extractor/__init__.py` and is transparent to callers.

---

## Appendix C: Example Output (Real Reliance Industries Annual Report)

To demonstrate the pipeline's capabilities on real-world large documents, the repository includes backup files from processing the [Reliance Industries Annual Report 2024-25](https://www.ril.com/reports/RIL-Integrated-Annual-Report-2024-25.pdf):

| Example File | Description | Size |
|--------------|-------------|------|
| `data/raw_report_ril.txt` | Raw text extracted from Reliance Annual Report PDF | 13,344 lines |
| `data/backup/graph_output_v1.json` | Full Knowledge Graph (1,198 nodes, 694 relationships) | 8,659 lines |
| `visuals/backup/graph_v1.mmd` | Mermaid diagram for the full graph | 1,457 lines |
| `visuals/backup/graph_v1.html` | Interactive HTML visualization | 2,028 lines |

These files demonstrate:
- Processing of 200+ page financial documents
- Automatic chunking and graph merging
- Interactive HTML visualization with zoom/pan controls
- Real-world entity and relationship extraction

---

## Appendix B: External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| `pydantic` | 2.x | Schema definition and validation |
| `openai` | 1.x | OpenAI GPT-4o API client |
| `google-genai` | 1.x | Google Gemini API client |
| `httpx` | 0.25.x | HTTP client (used by Ollama extractor for local API calls) |
| `python-dotenv` | 1.x | Environment variable loading from .env files |
| `networkx` | 3.x | Graph data structure and algorithms |
| `matplotlib` | 3.x | Graph rendering to PNG |
| `pytest` | 8.x | Testing framework |

### Runtime Dependencies

| Dependency | Required | Purpose |
|------------|----------|---------|
| **Ollama** | When `LLM_PROVIDER=ollama` | Local LLM inference server (HTTP-based) |

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | December 2024 | Architecture Team | Initial release |
| 1.1 | December 2024 | Architecture Team | Multi-provider LLM support (OpenAI, Gemini, Ollama); Mermaid visualization; optional confidence scores |
| 1.2 | December 2024 | Architecture Team | Removed company-specific references; made documentation generic |
| 1.3 | December 2024 | Architecture Team | Added automatic document chunking for large documents; simple merge strategy |
| 1.4 | December 2024 | Architecture Team | Added HTML viewer for browser-based Knowledge Graph visualization |
| 2.0 | December 2024 | Architecture Team | Major update: 18 relationship types; context field for nodes; node deduplication in merger; auto-repair for invalid relationships; orphan node removal; paginated HTML for large graphs; iterative extraction with live updates; clean_graph.py utility |
| 2.1 | December 2024 | Architecture Team | Enhanced HTML visualization: dark theme, auto-fit on load, zoom slider (5%-200%), preset zoom buttons, keyboard shortcuts, pan navigation, touch support |
| 2.2 | January 2025 | Architecture Team | Comprehensive test coverage (185 tests across 10 modules); added example output files from Reliance Industries Annual Report; updated testing strategy documentation |
