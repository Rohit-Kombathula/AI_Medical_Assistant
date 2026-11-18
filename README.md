# AI Medical Assistant (India & Multilingual Edition)
### Clinical Triage & Medical Report Explainer

An autonomous, production-ready healthcare application tailored for **India's healthcare ecosystem** and powered by **Groq** (`qwen/qwen3.8-27b` with ultra-low latency LPU inference). Provides intelligent preliminary symptom triage, multilingual emergency red-flag screening (English and Telugu), and plain-language medical lab report explanations.

---

## Key Features

1. **India-Centric Healthcare Navigation**:
   - Integrates India's national emergency services: **108** (Emergency Ambulance), **112** (National Unified Emergency), and Tele-MANAS (**14416**) for mental health crisis support.
   - Clinical triage routing tailored for Indian infrastructure: Primary Health Centres (PHC), Community Health Centres (CHC), District Hospitals, Area Hospitals, and Hospital Casualty departments.
   - Evaluates symptoms with awareness of regional epidemiology (e.g., Dengue, Malaria, Typhoid, Chikungunya, viral fevers, heat exhaustion, and pollution-induced respiratory issues).

2. **Multilingual Emergency Red-Flag Screening (Telugu & English)**:
   - Real-time screening for critical conditions across **Telugu Script** (e.g., `గుండె నొప్పి`, `శ్వాస ఆడట్లేదు`, `పక్షవాతం`), **Romanized Telugu / Tanglish** (e.g., `gundelo noppi`, `oopiri aadatledu`), and **English** (chest pain, shortness of breath, stroke symptoms).
   - Generates instant emergency warning alerts in the patient's language advising immediate 108 / 112 contact.

3. **Multilingual Clinical Symptom Triage**:
   - Responds fluently in natural **Telugu (తెలుగు)** when the user prompts in Telugu or Romanized Telugu, and in English for English queries.
   - Clinical considerations, urgency triage levels (Home Care, Routine Appointment, Urgent Care, Casualty).
   - Doctor consultation questions tailored for Indian physicians.

4. **Medical & Laboratory Report Explainer**:
   - Translates dense laboratory and diagnostic reports (e.g., CBC, CMP, Lipid Panels, LFT, KFT/RFT, Thyroid, Dengue serology) from Indian diagnostic chains (Apollo, Dr Lal PathLabs, Vijaya, Metropolis, Thyrocare, etc.) into accessible language.
   - Generates targeted doctor consultation questions with one-click clipboard copying.

5. **Groq-Powered Ultra-Low Latency Inference**:
   - Powered by `qwen/qwen3.8-27b` with instantaneous multi-model failover (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`) ensuring zero downtime and rate-limit resilience.

6. **Zero-Build, Lightweight Architecture**:
   - Python FastAPI backend serving asynchronous clinical endpoints.
   - Single-file responsive frontend built with Tailwind CSS via CDN and vanilla JavaScript (zero npm/Node steps).

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
