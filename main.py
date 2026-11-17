import os
import re
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medical_assistant")

# Securely configure Gemini API key
# Defaults to the pre-configured key or reads from environment variable GEMINI_API_KEY
DEFAULT_API_KEY = "AQ.Ab8RN6IxuKMUULgS0LiHDfSchos3oU04B3QNzBLe8BuEyTZbIQ"
API_KEY = os.getenv("GEMINI_API_KEY", DEFAULT_API_KEY).strip()

if not API_KEY:
    logger.warning("Gemini API key is not configured!")
else:
    genai.configure(api_key=API_KEY)

# Primary model name requested
MODEL_NAME = "gemini-2.5-flash"

# Comprehensive emergency clinical red-flags
EMERGENCY_PATTERNS = [
    r"\b(chest\s+pain|chest\s+tightness|chest\s+pressure|crushing\s+chest|heart\s+attack|cardiac\s+arrest)\b",
    r"\b(can'?t\s+breathe|severe\s+shortness\s+of\s+breath|struggling\s+to\s+breathe|choking|gasping\s+for\s+air|asphyxiat)\b",
    r"\b(stroke|facial\s+droop|face\s+droop|slurred\s+speech|arm\s+weakness|sudden\s+paralysis|sudden\s+numbness\s+on\s+one\s+side)\b",
    r"\b(coughing\s+up\s+blood|vomiting\s+blood|uncontrolled\s+bleeding|severe\s+hemorrhage|arterial\s+bleed)\b",
    r"\b(anaphylaxis|throat\s+closing|throat\s+swelling|swollen\s+tongue\s+and\s+breathing\s+trouble)\b",
    r"\b(unconscious|passed\s+out|fainted\s+and\s+unresponsive|loss\s+of\s+consciousness|prolonged\s+seizure)\b",
    r"\b(suicide|kill\s+myself|end\s+my\s+life|want\s+to\s+die|self[\s-]harm)\b",
    r"\b(worst\s+headache\s+of\s+my\s+life|thunderclap\s+headache|sudden\s+severe\s+headache)\b",
    r"\b(poison|swallowed\s+bleach|overdose|ingested\s+pills|carbon\s+monoxide)\b",
]
EMERGENCY_REGEX = re.compile("|".join(EMERGENCY_PATTERNS), re.IGNORECASE)

EMERGENCY_DISCLAIMER_TEXT = (
    "🚨 **CRITICAL EMERGENCY WARNING**: The symptoms or keywords detected indicate a potential "
    "life-threatening emergency. **Immediately call your local emergency service (such as 911 in North America, "
    "999 in the UK, 112 in Europe) or go directly to the nearest emergency department.** "
    "Do NOT wait, and do not drive yourself if you are feeling faint, confused, or in severe distress."
)

CLINICAL_TRIAGE_SYSTEM_PROMPT = """You are an expert Clinical AI Medical Assistant specializing in patient triage guidance and clinical communication.
Your goal is to provide objective, clear, empathetic, and structured preliminary guidance for symptoms described by patients.

Always format your response using clean Markdown with the following clear sections:
### 1. Preliminary Summary
Briefly summarize the reported symptoms, onset, and context in clear clinical language.

### 2. Clinical Considerations
Outline 2 to 4 potential medical possibilities or categories that commonly present with these symptoms. 
CRITICAL RULE: Never state a definitive diagnosis. Frame them as possibilities for a doctor to investigate.

### 3. Triage Level & Recommended Action
Classify the triage level into one of the following:
- **Emergency Care (Immediate)**: Immediate 911 / Emergency Room evaluation.
- **Urgent Care (Within 12-24 Hours)**: Walk-in clinic or same-day medical provider.
- **Routine Medical Appointment**: Schedule an appointment with a primary care physician in the coming days.
- **Home Care & Monitoring**: Mild symptoms appropriate for supportive self-care while monitoring progression.

### 4. Red-Flag Warning Signs to Watch For
Bullet points of specific worsening symptoms that would warrant immediate escalation to emergency care.

### 5. Questions to Ask Your Doctor
3 to 5 targeted, high-value questions the user can take to their physician appointment.

### 6. Clinical Uncertainty & Limitation Notice
A short, clear reminder that AI cannot physically examine, run diagnostic tests, or replace an in-person licensed physician.
"""

REPORT_EXPLAINER_SYSTEM_PROMPT = """You are an expert Clinical Laboratory & Diagnostic Report Explainer.
Your purpose is to translate complex laboratory results, blood panels, imaging reports, or pathology summaries into patient-accessible, clear, plain language without sacrificing clinical accuracy.

Always structure your analysis with the following Markdown sections:
### 1. Report Overview
Identify the nature of the report (e.g., Complete Blood Count, Metabolic Panel, Lipid Profile, Urinalysis, Thyroid Panel, Imaging Summary) and the primary physiological systems evaluated.

### 2. Key Biomarkers & Test Findings
Break down the important markers found in the report:
- Clearly explain what each test measures in simple terms.
- Highlight values that appear within normal physiological ranges.
- Explicitly highlight any values that appear elevated, decreased, or out-of-range, explaining what that direction typically indicates.

### 3. Clinical Context & Plain-Language Explanation
Explain what these overall patterns often suggest in everyday medical terms. Avoid alarmist conclusions while ensuring the patient understands any notable anomalies.

### 4. High-Priority Questions for Your Doctor
Provide 3 to 5 specific, actionable questions for the patient's next clinical consultation with their ordering healthcare provider.

### 5. Interpretation Disclaimer
Remind the patient that laboratory reference ranges vary by testing facility, age, sex, and clinical history, and only their physician can interpret results in context.
"""

app = FastAPI(
    title="AI Medical Assistant",
    description="Autonomous Clinical Triage & Medical Report Explainer API",
    version="1.0.0",
)

# Enable CORS for universal frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Patient symptoms or health question")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Previous conversation turns")


class ChatResponse(BaseModel):
    response: str
    is_emergency: bool = False
    emergency_guidance: Optional[str] = None


class ReportRequest(BaseModel):
    report_text: str = Field(..., min_length=5, description="Raw text of medical/lab report")


class ReportResponse(BaseModel):
    summary: str
    suggested_questions: List[str]


def get_gemini_model(system_instruction: str):
    """Instantiate the Gemini generative model with fallback support."""
    # Preferred model: gemini-2.5-flash as requested
    candidate_models = [MODEL_NAME, "gemini-2.0-flash", "gemini-1.5-flash"]
    
    last_err = None
    for model_id in candidate_models:
        try:
            model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=system_instruction,
                generation_config={
                    "temperature": 0.2,  # Low temperature for clinical precision and consistency
                    "top_p": 0.95,
                    "max_output_tokens": 2048,
                },
            )
            return model
        except Exception as e:
            last_err = e
            logger.warning("Could not initialize %s: %s", model_id, e)
            
    # Fallback to standard instantiation if system_instruction is not supported in older SDK versions
    try:
        return genai.GenerativeModel(model_name="gemini-1.5-flash")
    except Exception as e:
        logger.error("Failed to initialize any Gemini model: %s", e)
        raise last_err or e


def detect_emergency(text: str) -> bool:
    """Check text for acute emergency medical red-flags."""
    return bool(EMERGENCY_REGEX.search(text))


@app.get("/")
async def serve_index():
    """Serve the single-file UI dashboard."""
    index_file = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file, media_type="text/html")
    return HTMLResponse("<h2>AI Medical Assistant is active. index.html not found.</h2>")


@app.get("/health")
async def health_check():
    """Health check endpoint for Render/container orchestration."""
    return {
        "status": "healthy",
        "service": "AI Medical Assistant",
        "model_configured": MODEL_NAME,
        "api_key_configured": bool(API_KEY),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_triage(payload: ChatRequest):
    """
    POST /api/chat:
    Accepts user symptoms, checks for acute red-flags, calls Gemini 2.5 Flash,
    and returns structured clinical guidance.
    """
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

    is_emergency = detect_emergency(user_msg)
    emergency_text = EMERGENCY_DISCLAIMER_TEXT if is_emergency else None

    # Construct conversation history
    prompt_parts = []
    
    if payload.history:
        for turn in payload.history[-6:]:  # Keep recent context
            role_label = "User" if turn.role.lower() == "user" else "Clinical Assistant"
            prompt_parts.append(f"{role_label}: {turn.content}")

    prompt_parts.append(f"User: {user_msg}")
    prompt_parts.append("Clinical Assistant:")

    full_prompt = "\n\n".join(prompt_parts)

    try:
        model = get_gemini_model(CLINICAL_TRIAGE_SYSTEM_PROMPT)
        gemini_response = model.generate_content(full_prompt)
        ai_text = gemini_response.text if hasattr(gemini_response, "text") and gemini_response.text else (
            "I was unable to formulate clinical triage for this inquiry. Please consult a qualified physician immediately."
        )

        # Prepend emergency warning banner if red-flags were detected
        if is_emergency:
            ai_text = f"{EMERGENCY_DISCLAIMER_TEXT}\n\n---\n\n{ai_text}"

        return ChatResponse(
            response=ai_text,
            is_emergency=is_emergency,
            emergency_guidance=emergency_text,
        )

    except Exception as e:
        logger.exception("Error in /api/chat: %s", e)
        # Graceful fallback so the client receives a structured clinical message
        fallback_msg = (
            f"### Preliminary Clinical Observation\n\n"
            f"You described: *\"{user_msg[:120]}...\"*\n\n"
            f"Due to temporary service connectivity with the clinical model, here is essential preliminary triage guidance:\n"
            f"- If your symptoms are sudden, severe, or worsening, seek prompt evaluation at an urgent care center or emergency room.\n"
            f"- For persistent or concerning symptoms, contact your primary care physician.\n"
            f"- Keep a detailed log of symptom frequency, intensity (1-10), and triggers to present to your doctor.\n\n"
            f"*(Service connection note: {str(e)})*"
        )
        if is_emergency:
            fallback_msg = f"{EMERGENCY_DISCLAIMER_TEXT}\n\n---\n\n{fallback_msg}"
            
        return ChatResponse(
            response=fallback_msg,
            is_emergency=is_emergency,
            emergency_guidance=emergency_text,
        )


@app.post("/api/analyze-report", response_model=ReportResponse)
async def analyze_medical_report(payload: ReportRequest):
    """
    POST /api/analyze-report:
    Accepts raw medical/lab report text, processes it via Gemini model,
    and returns a plain-language explanation and physician consultation questions.
    """
    report_text = payload.report_text.strip()
    if not report_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Report text is required.")

    prompt = (
        f"Please analyze and explain the following clinical/lab report in clear, patient-friendly language:\n\n"
        f"--- BEGIN REPORT ---\n{report_text}\n--- END REPORT ---\n\n"
        f"Ensure you provide both the detailed plain-language breakdown and a curated list of specific questions "
        f"the patient should bring to their doctor."
    )

    try:
        model = get_gemini_model(REPORT_EXPLAINER_SYSTEM_PROMPT)
        gemini_response = model.generate_content(prompt)
        analysis_text = gemini_response.text if hasattr(gemini_response, "text") and gemini_response.text else (
            "Unable to generate report explanation. Please review your report with your healthcare provider."
        )

        # Extract suggested doctor questions using regex or fallback to standard questions
        extracted_questions = []
        questions_block = re.findall(r"(?:Questions|Doctor).*", analysis_text, re.DOTALL | re.IGNORECASE)
        if questions_block:
            lines = questions_block[0].split("\n")
            for line in lines:
                cleaned = line.strip().lstrip("0123456789.-*• \t")
                if cleaned.endswith("?") and len(cleaned) > 15:
                    extracted_questions.append(cleaned)
                    if len(extracted_questions) >= 5:
                        break

        if not extracted_questions:
            extracted_questions = [
                "What do these specific test results mean for my overall health?",
                "Are any of my values significantly out of range or requiring immediate follow-up?",
                "Do you recommend repeating any of these tests in 3 to 6 months?",
                "Are there dietary, lifestyle, or medication adjustments indicated by these markers?",
                "Do these findings warrant any additional diagnostic imaging or specialist referral?",
            ]

        return ReportResponse(
            summary=analysis_text,
            suggested_questions=extracted_questions,
        )

    except Exception as e:
        logger.exception("Error in /api/analyze-report: %s", e)
        fallback_summary = (
            f"### Laboratory Report Preliminary Review\n\n"
            f"The uploaded document contains clinical/laboratory data ({len(report_text)} characters).\n\n"
            f"**Key General Guidance:**\n"
            f"1. Medical test reference intervals differ significantly across laboratory providers.\n"
            f"2. Mild elevations or decreases in single biomarkers are common and must be assessed in the context of your complete health history.\n"
            f"3. Please deliver this report directly to your ordering clinician for definitive evaluation.\n\n"
            f"*(Service notice: {str(e)})*"
        )
        return ReportResponse(
            summary=fallback_summary,
            suggested_questions=[
                "What are the most significant findings in this lab report?",
                "Do any of the out-of-range values require immediate treatment?",
                "Should we re-test any specific biomarkers to check for trends?",
            ],
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting AI Medical Assistant server on port %s...", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
