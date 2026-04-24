# Chiron Memory Smart Chunking Schema — Test Backend

A minimal, standalone Python backend that answers one question:

> Does the Memory Smart Chunking Schema produce better, more grounded answers than asking `openai/gpt-oss-20b` directly without it?

Every test run makes **two** inference calls side by side and returns a `verdict` field (`schema_better`, `baseline_better`, or `equal`) with full reasoning traces so you can see exactly why one approach outperformed the other.

---

## What Is Being Tested

| | Baseline | Chiron Schema |
|---|---|---|
| Model | `openai/gpt-oss-20b` | `openai/gpt-oss-20b` |
| Context | Raw question only | Question + all 5 schema layers |
| Memory | None | Mission Anchor, Working Memory, Episodic Memory |
| Auditability | None | Full chunk-by-chunk audit trail |
| Confidence | Self-reported | Iteratively refined across cycles |
| Sources | Self-reported | Grounded against chunk sources |

The only variable that changes is the Memory Smart Chunking Schema. The base model is identical in both paths.

---

## Prerequisites

- Python 3.11+
- CUDA GPU with at least 16 GB VRAM (required to run `openai/gpt-oss-20b`)
- A HuggingFace account and access token with access to `openai/gpt-oss-20b`

---

## Install

```bash
# 1. Clone and enter the directory
cd chiron-memory-test

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install vLLM for fast inference (recommended if on CUDA 12.x)
pip install vllm>=0.4.3
# If vLLM fails to install, the server falls back to HuggingFace transformers automatically.
```

---

## Set Your HuggingFace Token

```bash
cp .env.example .env
```

Open `.env` and replace `hf_your_token_here` with your actual token:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get your token at https://huggingface.co/settings/tokens

---

## Start the Server Locally

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server will load `openai/gpt-oss-20b` on startup. This takes several minutes the first time (downloading weights) and is faster on subsequent starts if `MODEL_CACHE_DIR` already contains the weights.

Wait until you see:

```
INFO  model_loader — Model loaded via vllm backend
INFO  uvicorn — Application startup complete.
```

Then verify with:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status": "ok", "model_loaded": true, "model_name": "openai/gpt-oss-20b"}
```

---

## Run a Test

### Using the pre-written sample data

The `test_queries/` directory contains 10 chunks about the 2008 financial crisis and 5 questions that require synthesising across multiple chunks.

```bash
curl -X POST http://localhost:8000/test \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What were the root causes of the 2008 financial crisis, and how did each one connect to the others to produce a systemic collapse?",
    "chunks": [
      {
        "text": "Between 2001 and 2006, U.S. home prices rose by approximately 124 percent, fuelled by historically low interest rates...",
        "source": "housing_bubble_overview"
      },
      {
        "text": "The subprime mortgages were not held on bank balance sheets. Instead, banks bundled thousands of individual mortgages into mortgage-backed securities...",
        "source": "securitisation_mechanism"
      },
      {
        "text": "Post-crisis inquiries identified several systemic regulatory failures. The 1999 repeal of the Glass-Steagall Act...",
        "source": "regulatory_failures"
      }
    ]
  }'
```

### Using jq for readable output

```bash
curl -s -X POST http://localhost:8000/test \
  -H "Content-Type: application/json" \
  -d @- << 'EOF' | jq '{verdict, baseline_confidence: .baseline.confidence, schema_confidence: .schema_result.confidence, baseline_answer: .baseline.answer[0:200], schema_answer: .schema_result.answer[0:200]}'
{
  "question": "How did the collapse of Lehman Brothers trigger a freeze in the broader financial system?",
  "chunks": [
    {"text": "Lehman Brothers filed for Chapter 11 bankruptcy on September 15, 2008...", "source": "lehman_brothers_collapse"},
    {"text": "Lehman's collapse triggered an immediate freeze in the interbank lending market...", "source": "credit_freeze_interbank"},
    {"text": "U.S. GDP contracted 8.9 percent (annualised) in Q4 2008...", "source": "real_economy_impact"}
  ]
}
EOF
```

---

## Understanding the Response

```json
{
  "session_id": "uuid",
  "question": "...",
  "baseline": {
    "answer": "...",
    "confidence": 0.45,
    "sources_cited": [],
    "hallucination_risk": "medium"
  },
  "schema_result": {
    "answer": "...",
    "confidence": 0.82,
    "sources_cited": ["lehman_brothers_collapse", "credit_freeze_interbank"],
    "hallucination_risk": "low",
    "cycles_used": 3,
    "gate_passed": true,
    "reasoning_trace": [...]
  },
  "chunk_records": [...],
  "working_memory_final": {...},
  "episodic_memory": [...],
  "verdict": "schema_better"
}
```

### The `verdict` Field

| Value | Meaning |
|---|---|
| `schema_better` | Schema confidence is > 0.1 higher than baseline AND schema cited more sources |
| `baseline_better` | Baseline confidence is > 0.1 higher than schema AND baseline cited more sources |
| `equal` | Neither condition met — difference is within noise range |

### What a Good Result Looks Like

- `verdict: "schema_better"`
- `schema_result.confidence` noticeably higher than `baseline.confidence` (e.g. 0.80 vs 0.40)
- `schema_result.sources_cited` contains the chunk source labels (grounded)
- `baseline.sources_cited` is empty or contains invented sources
- `schema_result.gate_passed: true`
- `reasoning_trace` shows how confidence built up across cycles
- `baseline.hallucination_risk: "high"` vs `schema_result.hallucination_risk: "low"`

### What a Bad Result Looks Like

- `verdict: "baseline_better"` or `verdict: "equal"` when you expected schema to win
- `schema_result.gate_passed: false` — schema ran out of chunks before meeting confidence threshold
- `working_memory_final.confidence_score` below 0.70 — model found the chunks irrelevant
- `chunk_records` showing `status: "invalid"` for most chunks — the question and chunks don't match

The most common cause of a bad result is providing chunks that are not relevant to the question. The schema is only as good as the retrieval that feeds it.

---

## Inspect the Audit Trail

The `chunk_records` array in the response is the Layer 3 audit trail. Each entry shows:

```json
{
  "cycle_number": 1,
  "chunk_source": "housing_bubble_overview",
  "chunk_text_preview": "Between 2001 and 2006...",
  "key_insights": ["Home prices rose 124% 2001-2006", "20% of mortgages were subprime by 2005"],
  "relevance_score": 0.87,
  "hypothesis_changed": true,
  "status": "valid",
  "processed_at": "2025-09-01T14:23:11Z"
}
```

This is the full audit record an enterprise customer would see — exactly what was considered, what was extracted, and whether the model's hypothesis changed as a result.

---

## Deploy to RunPod Serverless

### Step 1: Build the Docker image locally

```bash
docker build -t chiron-memory-test .
```

Confirm the build succeeded:

```bash
docker run --rm chiron-memory-test python3 -c "import fastapi; import runpod; print('OK')"
```

### Step 2: Push to Docker Hub

```bash
# Tag the image (replace YOUR_DOCKERHUB_USERNAME with your Docker Hub username)
docker tag chiron-memory-test YOUR_DOCKERHUB_USERNAME/chiron-memory-test:latest

# Log in to Docker Hub
docker login

# Push
docker push YOUR_DOCKERHUB_USERNAME/chiron-memory-test:latest
```

Or push to GitHub Container Registry instead:

```bash
# Authenticate
echo $GITHUB_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Tag and push
docker tag chiron-memory-test ghcr.io/YOUR_GITHUB_USERNAME/chiron-memory-test:latest
docker push ghcr.io/YOUR_GITHUB_USERNAME/chiron-memory-test:latest
```

### Step 3: Create the RunPod Serverless endpoint

1. Go to [runpod.io/console/serverless](https://runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **Custom Container**
4. Enter your container image: `YOUR_DOCKERHUB_USERNAME/chiron-memory-test:latest`
5. GPU type: **NVIDIA A100 80GB** (recommended) or RTX 4090 (minimum for 20B model)
6. Max workers: **1** (increase once you've confirmed the model loads correctly)
7. Container disk: **50 GB** (to hold model weights)
8. Click **Deploy**

### Step 4: Set environment variables in RunPod dashboard

In your endpoint settings under **Environment Variables**, add:

| Key | Value |
|---|---|
| `MODEL_NAME` | `openai/gpt-oss-20b` |
| `HF_TOKEN` | `hf_xxxxxxxxxxxxxxxxxxxx` |
| `MODEL_CACHE_DIR` | `/runpod-volume/model-cache` |

### Step 5: Attach a Network Volume (recommended)

Without a Network Volume, the model (≈40 GB) re-downloads on every cold start.

1. In RunPod, go to **Storage > Network Volumes**
2. Create a volume of at least **60 GB** in the same region as your endpoint
3. In your endpoint settings, attach the volume and set the mount path to `/runpod-volume`
4. Set `MODEL_CACHE_DIR=/runpod-volume/model-cache` (already done in Step 4)

After the first run the weights are cached on the volume — subsequent cold starts load from disk, not the internet.

### Step 6: Test the deployed endpoint

RunPod Serverless endpoints use a different URL format but accept the identical JSON payload:

```bash
# Replace ENDPOINT_ID with your RunPod endpoint ID (shown in the dashboard)
# Replace YOUR_RUNPOD_API_KEY with your RunPod API key

curl -X POST https://api.runpod.ai/v2/ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{
    "input": {
      "endpoint": "/test",
      "question": "What were the root causes of the 2008 financial crisis?",
      "chunks": [
        {
          "text": "Between 2001 and 2006, U.S. home prices rose by approximately 124 percent...",
          "source": "housing_bubble_overview"
        },
        {
          "text": "The subprime mortgages were not held on bank balance sheets...",
          "source": "securitisation_mechanism"
        }
      ]
    }
  }'
```

The only difference from the local curl is:
- **URL**: `https://api.runpod.ai/v2/ENDPOINT_ID/runsync` instead of `http://localhost:8000/test`
- **Wrapper**: The payload is wrapped in `{"input": {...}}` for RunPod
- **Auth**: Bearer token header required

The response body inside the `output` field is **identical** to the local response.

To check health on RunPod:

```bash
curl -X POST https://api.runpod.ai/v2/ENDPOINT_ID/runsync \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -d '{"input": {"endpoint": "/health"}}'
```

### Step 7: Monitor costs

1. In the RunPod dashboard, go to **Billing > Usage**
2. A100 80GB costs approximately $2.21/hour when active
3. Serverless billing is per-second — you only pay when a job is running
4. An idle endpoint with no traffic costs nothing (workers scale to zero)
5. The first request after a cold start will take longer (model load) — use `/health` to pre-warm

### Model warm-up behaviour

On RunPod Serverless, `handler.py` loads `openai/gpt-oss-20b` once at container startup using `asyncio.get_event_loop().run_until_complete(load_model())`. This means:

- **First request**: model loads from the Network Volume (~2-5 minutes), then inference runs
- **Subsequent requests**: model is already in GPU memory — only inference time (seconds)
- **Scale-to-zero**: RunPod may spin down idle workers; next request triggers a new container load

Use the `/health` endpoint in your integration to check `model_loaded: true` before sending test queries.

---

## Project Structure

```
chiron-memory-test/
├── README.md               This file
├── requirements.txt        Python dependencies
├── .env.example            Environment variable template
├── Dockerfile              RunPod Serverless container
├── handler.py              RunPod entry point
├── rp_schema.json          RunPod input/output schema definition
├── main.py                 FastAPI app — /test and /health endpoints
├── model_loader.py         gpt-oss-20b loader (vLLM → transformers fallback)
├── memory/
│   ├── __init__.py
│   ├── mission_anchor.py   Layer 1: immutable session anchor
│   ├── working_memory.py   Layer 2: evolving brain state + deduplication
│   ├── chunk_record.py     Layer 3: per-chunk audit records
│   ├── episodic_memory.py  Layer 4: keyword-triggered recall
│   ├── synthesis_gate.py   Layer 5: four-condition pass check
│   └── schema.py           Pipeline orchestrator
├── inference/
│   ├── __init__.py
│   ├── prompts.py          All prompt templates
│   └── runner.py           JSON-enforced model caller (3 retries + regex fallback)
└── test_queries/
    ├── sample_chunks.json  10 chunks on the 2008 financial crisis
    └── sample_questions.json  5 multi-chunk test questions
```
