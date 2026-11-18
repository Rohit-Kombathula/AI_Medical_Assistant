# AI Medical Assistant (Clinical Triage & Medical Report Explainer)

An autonomous, production-ready healthcare application powered by **Groq** (`llama-3.3-70b-versatile`) providing intelligent preliminary symptom triage, emergency red-flag detection, and plain-language medical lab report explanations with ultra-low latency.

---

## Key Features

1. **Groq-Powered Clinical Inference**:
   - Ultra-fast token generation using Groq's high-performance LPU inference engine.
   - Flagship model: `llama-3.3-70b-versatile` with automatic failover to `llama-3.1-70b-versatile` and `llama-3.1-8b-instant`.

2. **Clinical Symptom Triage**:
   - Structured, evidence-based preliminary triage framework.
   - Clinical considerations, urgency triage levels (Home Care, Routine Appointment, Urgent Care, Emergency).
   - Recommended next steps and patient question preparation for physician consultations.
   - Clear uncertainty notices and limitations of AI.

3. **Automated Emergency Red-Flag Screening**:
   - Real-time screening for critical conditions: acute chest pain, acute respiratory distress, stroke warning signs (FAST), severe hemorrhage, anaphylaxis, sudden neurological deficits, and crisis triggers.
   - Instant high-visibility emergency warning alerts advising immediate emergency contact (911 / 112 / 999).

4. **Medical & Laboratory Report Explainer**:
   - Translates dense laboratory and diagnostic reports (e.g., CBC, CMP, Lipid Panels, Urinalysis, Thyroid) into clear, accessible language.
   - Highlights out-of-range biomarkers, normal intervals, and contextual clinical correlations.
   - Generates targeted doctor consultation questions with one-click clipboard copying.

5. **Zero-Build, Lightweight Architecture**:
   - Python FastAPI backend serving asynchronous clinical endpoints.
   - Single-file responsive frontend built with Tailwind CSS via CDN and vanilla JavaScript.
   - Zero Node.js or npm dependencies required.

---

## File Structure

```text
.
├── main.py              # FastAPI server, Groq client, triage engine, and API routes
├── index.html           # Single-file modern health-tech SaaS UI
├── requirements.txt     # Python production dependencies (FastAPI, Groq, Pydantic)
├── .env.example         # Example environment variables template
├── .gitignore           # Git ignore configuration
└── README.md            # Setup, deployment, and API reference
```

---

## Local Setup & Quickstart

### Prerequisites
- Python 3.9 or higher
- Git
- Groq API Key (Free from [console.groq.com/keys](https://console.groq.com/keys))

### 1. Clone or Open the Workspace
```bash
cd Medical_Assistant_Chatbot
```

### 2. (Recommended) Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Your Groq API Key
Create a `.env` file in the root directory (or copy `.env.example`):
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

### 5. Start the Application
```bash
uvicorn main:app --reload
```
or run directly with Python:
```bash
python main.py
```

### 6. Open in Your Browser
Visit [http://localhost:8000](http://localhost:8000) in your web browser.

---

## Deployment Guide (Render)

This application is ready to deploy directly on [Render](https://render.com) as a Web Service.

### Render Configuration Settings

| Setting | Value |
| :--- | :--- |
| **Environment** | `Python` |
| **Region** | Choose closest to your audience |
| **Branch** | `main` (or `master`) |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

### Environment Variables on Render
Add your Groq API key under **Environment Variables** in the Render Dashboard:

| Key | Value |
| :--- | :--- |
| `GROQ_API_KEY` | `gsk_your_groq_api_key_here` |

---

## Connecting to GitHub & Deploying Live

Follow these exact terminal commands to push your local repository to GitHub:

### Step 1: Create a New Repository on GitHub
Go to [GitHub](https://github.com/new) and create a new repository (e.g., `ai-medical-assistant`). **Do not initialize with README, .gitignore, or license** (they are already created locally).

### Step 2: Push Your Local Repository to GitHub
Run the following commands in your project terminal:

```bash
# Rename current branch to main
git branch -M main

# Link to your remote GitHub repository
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<REPO_NAME>.git

# Push code live to GitHub
git push -u origin main
```

### Step 3: Connect to Render
1. Go to the Render Dashboard and click **New +** -> **Web Service**.
2. Select **Build and deploy from a Git repository**.
3. Select your newly pushed `ai-medical-assistant` repository.
4. Set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add the `GROQ_API_KEY` under Environment Variables.
6. Click **Create Web Service**. Your app will be live with an automatic HTTPS URL in a few minutes.

---

## API Reference

### 1. `GET /`
Serves the single-page application dashboard.

### 2. `GET /health`
Returns system status and model readiness.
- **Response**:
  ```json
  {
    "status": "healthy",
    "service": "AI Medical Assistant (Groq)",
    "model_configured": "llama-3.3-70b-versatile",
    "api_key_configured": true
  }
  ```

### 3. `POST /api/chat`
Performs clinical symptom triage.
- **Request Body**:
  ```json
  {
    "message": "Persistent dry cough for 3 days with low-grade fever",
    "history": []
  }
  ```
- **Response Body**:
  ```json
  {
    "response": "### 1. Preliminary Summary...",
    "is_emergency": false,
    "emergency_guidance": null
  }
  ```

### 4. `POST /api/analyze-report`
Parses and explains laboratory and diagnostic reports in patient-friendly terms.
- **Request Body**:
  ```json
  {
    "report_text": "COMPLETE BLOOD COUNT: WBC 12.4 HIGH, Hemoglobin 11.2 LOW..."
  }
  ```
- **Response Body**:
  ```json
  {
    "summary": "### 1. Report Overview...",
    "suggested_questions": [
      "What do these specific test results mean for my overall health?",
      "Are any of my values significantly out of range or requiring immediate follow-up?"
    ]
  }
  ```

---

## Clinical Safety & Disclaimer
This software is designed solely for educational, informational, and triage navigation purposes. It does not provide medical diagnoses, treatment plans, or clinical prescriptions. Patients should always consult a licensed physician or healthcare professional regarding any medical condition or symptoms. In case of acute medical distress, contact local emergency services immediately.
