# 📰 Luyin Daily Digest System

A high-signal, AI-powered news aggregator and insight engine designed for EdTech, AI, and Business product teams. Luyin (陆音) transforms raw RSS feeds into a curated Slack digest featuring journalistic summaries and cross-cutting strategic insights.

---

## 🚀 Key Features

*   **Parallel Fetching:** High-speed ingestion of 20+ RSS feeds using `httpx` and `ThreadPoolExecutor`.
*   **Fuzzy Deduplication:** Smart filtering using `rapidfuzz` to catch similar stories across different publishers.
*   **Domain Quotas:** Balanced content selection (e.g., 4 EdTech, 2 AI, 1 Business) to prevent "news bloat."
*   **Gemini 2.0 Flash Integration:** 
    *   **Summarization:** Context-aware, 2-3 sentence summaries following strict "Sharp Editor" rules (no AI buzzwords).
    *   **Strategic Insights:** Generates 2-3 paragraphs of advisory notes connecting daily themes to product strategy.
*   **Self-Healing Sources:** An automated auditor (`audit_sources.py`) that detects broken feeds and uses Gemini to find current working URLs or replacements.
*   **Cloud Native:** Ready for deployment on **GCP (Cloud Functions + Cloud Scheduler)** or AWS Lambda.

---

## 🛠️ Tech Stack

*   **Language:** Python 3.12+
*   **AI:** Google Gemini (via `google-genai`)
*   **Networking:** `httpx` (with connection pooling)
*   **Parsing:** `feedparser`, `PyYAML`
*   **Matching:** `rapidfuzz` (Levenshtein distance)
*   **Resilience:** `tenacity` (exponential backoff retries)
*   **Testing:** `pytest` (50+ unit and integration tests)

---

## 📋 Prerequisites

*   Python 3.12 or higher.
*   A **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/)).
*   A **Slack Webhook URL** for the target channel.

---

## ⚙️ Configuration

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd luyin-daily-digest-system
    ```

2.  **Set up environment variables:**
    Create a `.env` file based on `.env.example`:
    ```env
    GEMINI_API_KEY=your_gemini_key_here
    SLACK_WEBHOOK_URL=your_slack_webhook_here
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Manage Sources:**
    Edit `sources.yaml` to add or remove RSS feeds and define their domains.

---

## 🏃 Usage

### Run the Digest Manually
```bash
python main.py
```

### Audit and Repair Feeds
Run the auditor to check for dead links and let Gemini suggest fixes:
```bash
python audit_sources.py
```

### Run Tests
```bash
pytest tests/
```

---

## ☁️ GCP Deployment (Step-by-Step)

### 1. Enable APIs
```bash
gcloud services enable cloudfunctions.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com cloudscheduler.googleapis.com
```

### 2. Store Secrets
```bash
echo -n "your-key" | gcloud secrets create GEMINI_API_KEY --data-file=-
echo -n "your-webhook" | gcloud secrets create SLACK_WEBHOOK_URL --data-file=-
```

### 3. Deploy Cloud Function (2nd Gen)
```bash
gcloud functions deploy luyin-digest \
  --gen2 \
  --runtime=python312 \
  --region=asia-southeast1 \
  --source=. \
  --entry-point=handler \
  --trigger-http \
  --set-secrets 'GEMINI_API_KEY=GEMINI_API_KEY:latest,SLACK_WEBHOOK_URL=SLACK_WEBHOOK_URL:latest' \
  --memory=512MB \
  --timeout=300s
```

### 4. Schedule Daily Runs
```bash
gcloud scheduler jobs create http luyin-daily-trigger \
  --schedule="0 9 * * *" \
  --uri="https://asia-southeast1-<project-id>.cloudfunctions.net/luyin-digest" \
  --location=asia-southeast1 \
  --oidc-service-account-email="<your-service-account>@<project-id>.iam.gserviceaccount.com"
```

---

## 🧪 Testing Philosophy

Luyin uses a tiered testing strategy:
1.  **Unit Tests:** Logic for filtering, quotas, and formatting.
2.  **Mock Integration:** Verifying AI prompt generation and Slack block construction.
3.  **Auditor Tests:** Verifying the "self-healing" YAML update logic.

Run the full suite with: `pytest tests/`

---

## ⚖️ License
MIT License - See [LICENSE](LICENSE) for details.
