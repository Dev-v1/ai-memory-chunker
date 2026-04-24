# Chiron Memory Test — Full Session Recap

This file is a complete summary of everything built and decided in this session.
If you are reading this in a new chat window, hand this entire file to Claude and say:
"Continue from this recap."

---

## What This Project Is

A standalone Python backend that tests ONE thing:
Does the **Memory Smart Chunking Schema** produce better answers than asking `openai/gpt-oss-20b` directly with no schema?

Every test run makes two inference calls side by side:
- **Baseline** — raw question sent directly to gpt-oss-20b, no chunks, no schema
- **Schema run** — the same question processed through all 5 schema layers with the provided chunks

The response includes a `verdict` field: `schema_better`, `baseline_better`, or `equal`.

This project is completely separate from the main Chiron backend (`chiron-backend/`). It does not touch that folder.

---

## Project Location

```
/mnt/c/Users/srira/PycharmProjects/Chiron/chiron-memory-test/
```

The main Chiron backend is at:
```
/mnt/c/Users/srira/PycharmProjects/Chiron/chiron-backend/
```

These two folders are siblings. Nothing in chiron-memory-test touches chiron-backend.

---

## The Model

- **Model**: `openai/gpt-oss-20b`
- **Released**: August 2025 by OpenAI
- **License**: Apache 2.0
- **Parameters**: 21 billion total, 3.6 billion active per token (Mixture of Experts)
- **Context window**: 128,000 tokens
- **VRAM required**: ~40 GB in float16
- **Format**: SafeTensors (`trust_remote_code=False`)
- **HuggingFace username that owns the Chiron project**: `devthedevoloper`
- **Inference backend**: vLLM only — no transformers fallback
- **GPU**: A100 80GB on RunPod Serverless
- **Loading**: weights load directly into VRAM at container startup and stay there — no disk caching between requests
- `gpu_memory_utilization=0.95` — gives the model ~76 GB of the 80 GB
- `max_model_len=8192` — keeps KV cache small for test workloads

---

## The Memory Smart Chunking Schema — 5 Layers

### Layer 1: Mission Anchor
- Created once per session, never modified
- Contains: session_id (UUID), original_query, auto-generated success_criteria, constraints (be factual, cite sources, flag hallucinations, never invent facts), created_at
- Stored in memory as a frozen Python dataclass

### Layer 2: Working Memory
- Updated after every chunk is processed
- Contains: confirmed_facts (with source, confidence, cycle number), open_questions, hypothesis, confidence_score (0.0–1.0), contradiction_flags (with severity), cycles_completed
- Deduplication: new facts with cosine similarity >= 0.85 to existing facts are skipped
- Cosine similarity is bag-of-words based (no external embedding model needed)

### Layer 3: Chunk Processing Record
- Every processed chunk creates a permanent audit receipt
- Contains: cycle_number, chunk_source, chunk_text_preview (truncated to 500 chars), key_insights, relevance_score, hypothesis_changed, status (valid / invalid / flagged), processed_at
- Status is `invalid` if relevance < 0.2, `flagged` if hallucinations were detected, `valid` otherwise

### Layer 4: Episodic Memory
- Chunks with relevance >= 0.8 are stored with a retrieval_trigger (keyword list)
- When a subsequent chunk's text matches any trigger keyword, the stored entry is re-injected into the prompt alongside the new chunk

### Layer 5: Synthesis Gate
- Four conditions must ALL pass before the final answer is generated:
  1. confidence_score >= 0.70
  2. All open_questions are empty or contain the word "unresolvable"
  3. No unresolved contradiction_flags with severity "high"
  4. Hypothesis has >= 25% keyword overlap with the original query
- If gate fails: retry with next chunk. If no chunks remain: return best answer with `gate_passed: false`

---

## Complete File Structure

```
chiron-memory-test/
├── recap.md                   ← this file
├── README.md                  ← full setup and deployment guide
├── requirements.txt           ← Python dependencies
├── .env.example               ← environment variable template
├── Dockerfile                 ← RunPod Serverless container (CUDA 12.1)
├── handler.py                 ← RunPod entry point (wraps the pipeline)
├── rp_schema.json             ← RunPod input/output schema
├── main.py                    ← FastAPI app — POST /test and GET /health
├── model_loader.py            ← loads gpt-oss-20b into VRAM via vLLM
├── memory/
│   ├── __init__.py
│   ├── mission_anchor.py      ← Layer 1
│   ├── working_memory.py      ← Layer 2
│   ├── chunk_record.py        ← Layer 3
│   ├── episodic_memory.py     ← Layer 4
│   ├── synthesis_gate.py      ← Layer 5
│   └── schema.py              ← pipeline orchestrator (coordinates all 5 layers)
├── inference/
│   ├── __init__.py
│   ├── prompts.py             ← all prompt templates
│   └── runner.py              ← JSON enforcement: 3 retries + regex fallback
└── test_queries/
    ├── sample_chunks.json     ← 10 chunks on the 2008 financial crisis
    └── sample_questions.json  ← 5 multi-chunk test questions
```

---

## API Endpoints

### GET /health
Returns:
```json
{"status": "ok", "model_loaded": true, "model_name": "openai/gpt-oss-20b"}
```

### POST /test
Request body:
```json
{
  "question": "string",
  "chunks": [
    {"text": "string", "source": "string"}
  ]
}
```

Response:
```json
{
  "session_id": "uuid",
  "question": "string",
  "baseline": {
    "answer": "string",
    "confidence": 0.0,
    "sources_cited": [],
    "hallucination_risk": "low|medium|high"
  },
  "schema_result": {
    "answer": "string",
    "confidence": 0.0,
    "sources_cited": [],
    "hallucination_risk": "low|medium|high",
    "cycles_used": 3,
    "gate_passed": true,
    "reasoning_trace": []
  },
  "chunk_records": [],
  "working_memory_final": {},
  "episodic_memory": [],
  "verdict": "schema_better | baseline_better | equal"
}
```

Verdict logic:
- `schema_better`: schema confidence > baseline by more than 0.1 AND schema cited more sources
- `baseline_better`: baseline confidence > schema by more than 0.1 AND baseline cited more sources
- `equal`: neither condition met

---

## Environment Variables

| Variable | Value | Notes |
|---|---|---|
| `MODEL_NAME` | `openai/gpt-oss-20b` | Do not change |
| `HF_TOKEN` | `hf_your_token` | From huggingface.co/settings/tokens |
| `MODEL_CACHE_DIR` | removed | No longer used — model loads directly to VRAM |

`MODEL_CACHE_DIR` was removed in this session. The model loads directly into VRAM. The only variable you need is `HF_TOKEN`.

---

## What Was Changed During This Session

### model_loader.py — rewritten
Original version tried vLLM first, fell back to HuggingFace transformers if vLLM failed.
New version: **vLLM only, no fallback**. Raises immediately if the load fails.
Key settings:
- `dtype="float16"` — fits gpt-oss-20b in ~40 GB VRAM
- `gpu_memory_utilization=0.95` — uses ~76 GB of the A100's 80 GB
- `max_model_len=8192` — keeps KV cache small, saves VRAM
- No `download_dir` — weights go straight from HuggingFace into VRAM

### requirements.txt — simplified
Removed: `transformers`, `torch`, `accelerate`, `safetensors`, `sentencepiece`, `huggingface-hub`
These are all bundled inside vLLM. Only direct dependencies remain:
- `fastapi`, `uvicorn`, `pydantic`, `python-dotenv`, `vllm`, `runpod`

---

## Deployment — RunPod Serverless

### Step 1: Build Docker image
```bash
cd chiron-memory-test
docker build -t chiron-memory-test .
```

### Step 2: Push to Docker Hub
```bash
docker tag chiron-memory-test YOUR_DOCKERHUB_USERNAME/chiron-memory-test:latest
docker login
docker push YOUR_DOCKERHUB_USERNAME/chiron-memory-test:latest
```

### Step 3: Create endpoint on RunPod
1. Go to runpod.io/console/serverless
2. New Endpoint → Custom Container
3. Image: `YOUR_DOCKERHUB_USERNAME/chiron-memory-test:latest`
4. GPU: A100 80GB
5. Max workers: 1
6. Container disk: 50 GB

### Step 4: Set environment variables in RunPod dashboard
- `MODEL_NAME` = `openai/gpt-oss-20b`
- `HF_TOKEN` = your HuggingFace token

Note: Do NOT set `MODEL_CACHE_DIR` — it was removed. The model loads directly to VRAM.

### Step 5: Attach a Network Volume (strongly recommended)
- Go to Storage → Network Volumes → create 60 GB in the same region as your endpoint
- Mount path: `/runpod-volume`
- This caches the downloaded weights so cold starts after the first one are fast

### Step 6: Check model is loaded before testing
```bash
curl -X POST https://api.runpod.ai/v2/ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{"input": {"endpoint": "/health"}}'
```
Wait until `"model_loaded": true` before sending test jobs.

---

## How to Run the Automated Tests — Full Instructions

### What the test script does
It sends curl requests from your local machine to the deployed RunPod endpoint and checks that each response contains an expected value. No testing framework required. Just bash and curl.

### Step 1: Find your two required values

**ENDPOINT_ID**
- Log in to RunPod
- Click Serverless in the left menu
- Click your endpoint
- The ID is visible in the URL bar: `runpod.io/console/serverless/YOUR_ENDPOINT_ID`
- Copy just the ID portion

**API_KEY**
- In RunPod, click your profile icon (top right)
- Click API Keys
- Copy your key (starts with a long string of characters)

### Step 2: Create the test file

Open any text editor on your local machine. Create a new file called `run_tests.sh` and paste the following. Replace the two placeholder values at the top with your real values:

```bash
#!/bin/bash

ENDPOINT_ID="your_endpoint_id_here"
API_KEY="your_runpod_api_key_here"
BASE_URL="https://api.runpod.ai/v2/$ENDPOINT_ID/runsync"

PASS=0
FAIL=0

run_test() {
  local name="$1"
  local payload="$2"
  local expected="$3"

  response=$(curl -s -X POST "$BASE_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d "$payload")

  if echo "$response" | grep -q "$expected"; then
    echo "PASS — $name"
    PASS=$((PASS + 1))
  else
    echo "FAIL — $name"
    echo "       Expected: $expected"
    echo "       Got: $(echo $response | head -c 400)"
    FAIL=$((FAIL + 1))
  fi
}

run_test "health check" \
  '{"input": {"endpoint": "/health"}}' \
  '"model_loaded":true'

run_test "schema beats baseline" \
  '{"input": {"endpoint": "/test", "question": "What caused the 2008 financial crisis?", "chunks": [{"text": "Between 2001 and 2006 U.S. home prices rose 124 percent. Subprime mortgages grew to 20 percent of all new loans by 2005.", "source": "housing_bubble"}, {"text": "Banks bundled subprime mortgages into mortgage-backed securities. Rating agencies gave them AAA ratings incorrectly.", "source": "securitisation"}, {"text": "Lehman Brothers filed for bankruptcy on September 15 2008 with 613 billion dollars in debt.", "source": "lehman"}]}}' \
  '"verdict":"schema_better"'

run_test "synthesis gate passes" \
  '{"input": {"endpoint": "/test", "question": "What caused the 2008 financial crisis?", "chunks": [{"text": "Between 2001 and 2006 U.S. home prices rose 124 percent. Subprime mortgages grew to 20 percent of all new loans by 2005.", "source": "housing_bubble"}, {"text": "Banks bundled subprime mortgages into mortgage-backed securities. Rating agencies gave them AAA ratings incorrectly.", "source": "securitisation"}, {"text": "Lehman Brothers filed for bankruptcy on September 15 2008 with 613 billion dollars in debt.", "source": "lehman"}]}}' \
  '"gate_passed":true'

run_test "missing question returns error" \
  '{"input": {"endpoint": "/test", "chunks": [{"text": "some text", "source": "src"}]}}' \
  '"error"'

run_test "empty chunks returns error" \
  '{"input": {"endpoint": "/test", "question": "some question", "chunks": []}}' \
  '"error"'

echo ""
echo "Results: $PASS passed, $FAIL failed"
```

### Step 3: Save the file

Save it as `run_tests.sh`. You can save it anywhere — your Desktop is fine.

### Step 4: Open a terminal

- **Windows**: Open WSL (Windows Subsystem for Linux) or Git Bash. Do NOT use Command Prompt or PowerShell — they do not support bash scripts.
- **Mac**: Open Terminal (Applications → Utilities → Terminal)
- **Linux**: Open any terminal

### Step 5: Navigate to where you saved the file

```bash
cd ~/Desktop
```
Or wherever you saved it.

### Step 6: Make the script executable

```bash
chmod +x run_tests.sh
```
You only need to do this once.

### Step 7: Run the tests

```bash
./run_tests.sh
```

### Step 8: Read the results

A passing run looks like:
```
PASS — health check
PASS — schema beats baseline
PASS — synthesis gate passes
PASS — missing question returns error
PASS — empty chunks returns error

Results: 5 passed, 0 failed
```

A failing run shows what was expected and what the response actually contained:
```
FAIL — schema beats baseline
       Expected: "verdict":"schema_better"
       Got: {"output":{"verdict":"equal","schema_result":...
```

### Troubleshooting

| What you see | Fix |
|---|---|
| `model_loaded:false` | Model still loading. Wait 2 minutes, run again. |
| Timeout or no response | Container was cold (RunPod scaled to zero). Run the health check test first to wake it, then rerun all tests. |
| `verdict:equal` instead of `schema_better` | The model ran correctly — it just wasn't confident enough to cross the 0.1 threshold. This is a real result, not a code failure. Try adding more chunks. |
| `curl: command not found` | Install curl. On Windows use Git Bash which includes it. |
| `Permission denied` | You skipped Step 6. Run `chmod +x run_tests.sh` first. |
| `401 Unauthorized` | Your API_KEY is wrong. Copy it again from RunPod → API Keys. |
| `404` in response | Your ENDPOINT_ID is wrong. Check the URL in the RunPod dashboard. |

---

## Sample Test Data

The project includes pre-written test data at:
- `test_queries/sample_chunks.json` — 10 chunks about the 2008 financial crisis
- `test_queries/sample_questions.json` — 5 questions that require synthesising across multiple chunks

These are ready to paste into test payloads. The 5 questions are:
1. What were the root causes of the 2008 financial crisis?
2. How did the collapse of Lehman Brothers trigger a freeze in the broader financial system?
3. To what extent was the government response effective and what reforms followed?
4. How did a U.S. housing problem spread to cause sovereign debt crises in Europe?
5. What was the total human cost and how does it compare to the scale of the government response?

---

## What Has NOT Been Built Yet

- No fine-tuned specialist models (reasoning, legal, medical, prompt injection checker)
- No frontend (Next.js)
- No authentication (Clerk)
- No payments (Stripe)
- No database (PostgreSQL + pgvector)
- No Redis
- No enterprise features

This test backend is purely for validating the Memory Smart Chunking Schema on the base orchestrator model before the full system is built.

---

## Key Technical Decisions Made in This Session

1. **vLLM only** — removed the HuggingFace transformers fallback to keep the code simple and inference fast
2. **VRAM loading** — model loads directly into GPU memory, no disk cache between requests — saves RunPod quota because inference is faster
3. **No MODEL_CACHE_DIR** — removed from environment variables, removed from model_loader.py
4. **A100 80GB** — chosen GPU; `gpu_memory_utilization=0.95`, `max_model_len=8192`
5. **test.sh approach** — automated tests are plain bash curl scripts, not a Python testing framework, because the model runs on RunPod not locally
