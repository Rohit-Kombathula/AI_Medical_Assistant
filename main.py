import os
import re
import logging
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from groq import Groq

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("medical_assistant")

# Securely configure Groq API key
DEFAULT_API_KEY = "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"
API_KEY = os.getenv("GROQ_API_KEY", DEFAULT_API_KEY).strip()

if not API_KEY or API_KEY.startswith("PASTE_"):
    logger.warning("GROQ_API_KEY is not set or is using a placeholder. Add GROQ_API_KEY to your .env file or environment.")

# Groq models with graceful fallback support
PRIMARY_MODEL = "qwen/qwen3.8-27b"
FALLBACK_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# Multilingual Emergency Red-Flag Detection (English, Telugu Script, and Romanized Telugu / Tanglish)
EMERGENCY_PATTERNS_EN = [
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

# Telugu Script Red-Flags (తెలుగు లిపి అత్యవసర సంకేతాలు)
EMERGENCY_PATTERNS_TE_SCRIPT = [
    r"(గుండె|ఛాతీ|గుండెల్లో).*?(నొప్పి|పోటు|భారంగా|ఆగడం|పట్టేసినట్లు)",
    r"(శ్వాస|ఊపిరి).*?(ఆడట్లేదు|ఆడటం\s*లేదు|తీసుకోలేక|ఆడక|కష్టంగా|తీసుకోలేకపోతున్న|పీల్చలేక)",
    r"(పక్షవాతం|మూతి\s*వంకర|ముఖం\s*వంకర|మాట\s*ముద్ద|మాట\s*రాలేదు|చెయ్యి\s*పడిపోయింది|కాలు\s*పడిపోయింది)",
    r"(రక్తం|రక్తపు).*?(వాంతులు|దగ్గు|కారుతోంది|ఆగడం\s*లేదు|స్రావం)",
    r"(గొంతు|నాలుక).*?(వాపు|పూడిపోవడం|వాచిపోవడం|ఊపిరాడక)",
    r"(స్పృహ\s*తప్పడం|సొమ్మసిల్లి|మూర్ఛ|ఫిట్స్|కళ్లు\s*తిరిగి\s*పడిపోవడం|తెలివి\s*తప్పడం)",
    r"(ఆత్మహత్య|చనిపోవాలని|ప్రాణం\s*తీసుకోవడం|బతకాలని\s*లేదు|చావాలని|చచ్చిపోవాలని)",
    r"(విషం|పురుగుల\s*మందు|మందులు\s*మింగే|ఓవర్\s*డోస్)",
    r"(తల\s*పగిలిపోయే\s*నొప్పి|విపరీతమైన\s*తలనొప్పి|ఆగని\s*తలనొప్పి)",
]

# Romanized Telugu / Tanglish Red-Flags (ఇంగ్లీష్ లిపిలో తెలుగు మాటలు)
EMERGENCY_PATTERNS_TE_ROMAN = [
    r"\b(gunde|gundelo|chathi|chathilo)\b.*?\b(noppi|pain|potu|bharanga|tightness)\b",
    r"\b(swasa|oopiri)\b.*?\b(aadat|aadadam\s*ledu|aadakapovadam|theesukoleka|kashtam|adaka|peelchaleka)\b",
    r"\b(pakshavatham|paralysis|moothi\s*vankara|mata\s*thiragatledu|mata\s*mudda|cheyyi\s*padipoyindi)\b",
    r"\b(raktham|rakthapu)\b.*?\b(vaanthi|vaanthulu|padutondi|vasthundi|karuthundi|cough)\b",
    r"\b(gonthu|naluka)\b.*?\b(vaachindi|poodipoyindi|swelling)\b",
    r"\b(spruha\s*thappadam|spruha\s*ledu|sommasilli|moorcha|fits\s*vachindi|kallu\s*thirigi|unconscious)\b",
    r"\b(athmahathya|chanipovalani|chavali\s*anipistundi|pranam\s*theesukovali|bathakalani\s*ledu|chachipovali)\b",
    r"\b(vishamu?|purugula\s*mandhu|overdose)\b",
    r"\b(thala\s*pagilipoye\s*noppi|theevramaina\s*thalanop)\b",
]

ALL_EMERGENCY_PATTERNS = EMERGENCY_PATTERNS_EN + EMERGENCY_PATTERNS_TE_SCRIPT + EMERGENCY_PATTERNS_TE_ROMAN
EMERGENCY_REGEX = re.compile("|".join(ALL_EMERGENCY_PATTERNS), re.IGNORECASE)

# Telugu linguistic markers for Romanized input
TELUGU_ROMAN_MARKERS = re.compile(
    r"\b(naaku|naku|gunde|gundelo|noppi|swasa|oopiri|raktham|jwaram|daggulu|vaanthi|kadupulo|undhi|undi|undha|leda|ela|chesukovali|pakshavatham|spruha|moorcha|athmahathya|chanipovalani|gonthu|theevramaina|doctor\s*garu|telugulo)\b",
    re.IGNORECASE
)

def is_telugu_query(text: str) -> bool:
    """Detect if query is in Telugu (Unicode script or Romanized Telugu)."""
    if re.search(r"[\u0C00-\u0C7F]", text):
        return True
    return bool(TELUGU_ROMAN_MARKERS.search(text))

# India-Specific Emergency Warnings (Bilingual Telugu / English)
EMERGENCY_DISCLAIMER_TEXT_EN = (
    "🚨 **CRITICAL MEDICAL EMERGENCY WARNING (INDIA - DIAL 108 / 112)**:\n\n"
    "The symptoms described indicate a potentially life-threatening medical emergency.\n"
    "- 🚑 **Ambulance Service:** Immediately dial **108** (National Medical Emergency) or **112** across India.\n"
    "- 🏥 **Immediate Hospital Evaluation:** Proceed straight to the nearest Hospital Casualty / Emergency Department (Government District/Area Hospital or Private Super-Specialty Hospital).\n"
    "- ⚠️ **Safety Caution:** Do NOT wait. Do NOT attempt to drive yourself if feeling breathless, faint, or disoriented.\n"
    "- 📞 **Mental Health Crisis:** For emotional or crisis support, call the Government of India Tele-MANAS helpline at **14416**."
)

EMERGENCY_DISCLAIMER_TEXT_TE = (
    "🚨 **అత్యవసర వైద్య హెచ్చరిక (భారతదేశ అత్యవసర సేవలు - 108 / 112)**:\n\n"
    "మీరు తెలిపిన లక్షణాలు ప్రాణాంతక అత్యవసర పరిస్థితిని సూచించవచ్చు.\n"
    "- 🚑 **అంబులెన్స్ సేవలు (Ambulance):** వెంటనే భారతదేశ అత్యవసర అంబులెన్స్ నంబర్ **108** లేదా **112** కు కాల్ చేయండి.\n"
    "- 🏥 **ఆసుపత్రి అత్యవసర విభాగం (Casualty):** ఆలస్యం చేయకుండా సమీపంలోని ఏరియా / జిల్లా ఆసుపత్రి లేదా ప్రైవేట్ ఆసుపత్రిలోని **క్యాజువాలిటీ / ఎమర్జెన్సీ** విభాగానికి వెళ్ళండి.\n"
    "- ⚠️ **ముఖ్య గమనిక:** దయచేసి ఆలస్యం చేయవద్దు. నీరసంగా లేదా కళ్ళు తిరుగుతున్నట్లు అనిపిస్తే మీరే స్వయంగా వాహనం నడపవద్దు.\n"
    "- 📞 **మానసిక ఆరోగ్య సహాయం:** తక్షణ సహాయం కొరకు భారత ప్రభుత్వ టెలి-మానస్ (Tele-MANAS) **14416** కు సంప్రదించండి.\n\n"
    "*(Call 108 / 112 immediately for Ambulance in India)*"
)

def get_emergency_disclaimer(user_msg: str) -> str:
    """Return India-specific emergency disclaimer tailored to the input language."""
    if is_telugu_query(user_msg):
        return EMERGENCY_DISCLAIMER_TEXT_TE
    return EMERGENCY_DISCLAIMER_TEXT_EN

CLINICAL_TRIAGE_SYSTEM_PROMPT = """You are an expert Clinical AI Medical Assistant specializing in patient triage guidance and clinical communication, specifically tailored for the healthcare ecosystem in INDIA while maintaining international evidence-based clinical standards.

### CRITICAL LANGUAGE DIRECTIVES:
1. Multilingual Telugu & English Adaptation:
   - If the patient writes in Telugu (తెలుగు లిపి) or Romanized Telugu (Telugu words typed in English alphabets like "gundelo noppi", "jwaram ga undi", "swasa kastam"), formulate your ENTIRE triage response in natural, empathetic, and clear TELUGU (తెలుగు). Include standard English medical terms in parentheses for precision (e.g., "డెంగ్యూ జ్వరం (Dengue Fever)", "రక్తపోటు (Hypertension)", "క్యాజువాలిటీ (Casualty/Emergency)").
   - If the patient writes in English, respond in English with clear Indian healthcare navigation context.

2. Indian Healthcare Navigation:
   - Emergency Services: Always reference 108 (National Emergency Medical Ambulance), 112 (All-in-one Emergency Helpline), 102 (Maternal & Infant), and Tele-MANAS (14416) for psychological distress.
   - Healthcare Facility Tiers in India:
     * Primary Tier: Primary Health Centre (PHC), Urban Primary Health Centre (UPHC), Ayushman Bharat Health & Wellness Centre, or local General Physician clinic (MBBS/MD).
     * Secondary Tier: Community Health Centre (CHC), Sub-Divisional / Area Hospital, or Taluk Hospital.
     * Tertiary Tier: District Hospital, Government Medical College (GMC), or private Multispecialty / Super-Specialty Hospital casualty.
   - Common Regional Epidemiological Conditions in India:
     When evaluating symptoms, consider common prevalent conditions in India (e.g., Dengue, Malaria, Typhoid, Chikungunya, seasonal viral fevers/flu, acute gastroenteritis from water/food contamination, heat-related illness in summer, post-monsoon vectors, and air-pollution induced bronchitis).

### STRUCTURED OUTPUT FORMAT (Markdown):
If responding in English:
### 1. Preliminary Summary
Brief clinical summary of reported symptoms, onset, context, and severity.

### 2. Clinical Considerations
2 to 4 potential medical possibilities to investigate. (Never state a definitive diagnosis; frame as differential possibilities for a doctor).

### 3. Triage Level & India Healthcare Action
Specify one:
- **Emergency Care (Casualty / 108)**: Immediate ambulance (108) or casualty evaluation.
- **Urgent Care (Within 12-24 Hours)**: Local area hospital casualty or private physician.
- **Routine Medical Appointment**: Primary care physician (MBBS/MD), PHC, or clinic in the next 1-3 days.
- **Home Care & Monitoring**: Safe supportive self-care (hydration, rest) with close monitoring.

### 4. Red-Flag Warning Signs to Watch For
Bullet points of specific worsening symptoms that warrant immediate hospital casualty visit.

### 5. Questions to Ask Your Doctor
3 to 5 targeted questions to ask their doctor in India.

### 6. Clinical Uncertainty & Limitation Notice
Clear reminder that AI cannot physically examine, prescribe schedule medications, or replace an in-person licensed medical practitioner in India.

If responding in Telugu (తెలుగు):
Use the corresponding Telugu sections:
### 1. ప్రాథమిక సారాంశం (Preliminary Summary)
### 2. పరిశీలించవలసిన వైద్య అంశాలు (Clinical Considerations)
### 3. తక్షణ వైద్య స్థాయి & తీసుకోవలసిన చర్య (Triage Level & Action in India)
### 4. అత్యవసర ప్రమాద సంకేతాలు (Red-Flag Warning Signs)
### 5. డాక్టర్ గారిని అడగవలసిన ముఖ్యమైన ప్రశ్నలు (Questions for Your Doctor)
### 6. ముఖ్య గమనిక (Clinical Uncertainty Notice)
"""

REPORT_EXPLAINER_SYSTEM_PROMPT = """You are an expert Clinical Laboratory & Diagnostic Report Explainer, specifically tailored for medical reports and laboratory panels in INDIA (e.g., NABL-accredited diagnostic chains such as Apollo, Dr Lal PathLabs, Vijaya, Metropolis, Thyrocare, SRL/Agilus, and Government Hospital labs).

### LANGUAGE RULES:
- If the patient query or report notes are in Telugu or Romanized Telugu, explain the report in patient-friendly TELUGU (తెలుగు) with key biomarker names also mentioned in English for easy reading.
- If in English, explain in plain English with reference to standard Indian clinical intervals.

### STRUCTURED REPORT SECTIONS (Markdown):
### 1. Report Overview / రిపోర్ట్ అవలోకనం
Identify the type of report (CBC, Lipid, LFT, KFT/RFT, HbA1c, Thyroid, Urine Routine, Dengue Serology, Typhoid Widal, etc.) and physiological systems assessed.

### 2. Key Biomarkers & Test Findings / ముఖ్యమైన పరీక్షల ఫలితాలు
- Plain-language explanation of what each biomarker measures.
- Normal values vs out-of-range (Elevated/Low) values clearly highlighted.

### 3. Clinical Context & Plain-Language Explanation / వైద్య వివరణ
What these findings typically suggest in everyday language without causing unwarranted panic.

### 4. High-Priority Questions for Your Doctor / డాక్టర్ గారిని అడగవలసిన ప్రశ్నలు
3 to 5 specific, targeted questions for their upcoming consultation with an Indian clinician.

### 5. Interpretation Disclaimer / ముఖ్య గమనిక
Reminder that reference ranges vary by laboratory and only a licensed physician can interpret results in context of physical examination.
"""

app = FastAPI(
    title="AI Medical Assistant",
    description="Autonomous Clinical Triage & Medical Report Explainer API (Powered by Groq)",
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


def get_groq_client() -> Optional[Groq]:
    """Retrieve or initialize Groq client safely."""
    key = os.getenv("GROQ_API_KEY", DEFAULT_API_KEY).strip()
    if not key or key.startswith("PASTE_"):
        return None
    try:
        return Groq(api_key=key, max_retries=0)
    except Exception as e:
        logger.error("Failed to initialize Groq client: %s", e)
        return None


def call_groq(messages: List[Dict[str, str]]) -> str:
    """
    Executes a chat completion call with Groq across candidate models with fallback.
    """
    client = get_groq_client()
    if client is None:
        return (
            "### ⚠️ Groq API Key Required\n\n"
            "The Groq API key has not been configured yet.\n\n"
            "**Quick Setup:**\n"
            "1. Obtain a free API key at [console.groq.com/keys](https://console.groq.com/keys).\n"
            "2. Add your key to the `.env` file in the project directory:\n"
            "   ```env\n"
            "   GROQ_API_KEY=gsk_your_actual_key_here\n"
            "   ```\n"
            "3. Restart the server or reload the application.\n\n"
            "*In case of an emergency in India, please contact 108 or 112 immediately.*"
        )

    last_error = None
    for model_name in FALLBACK_MODELS:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=2048,
            )
            if completion.choices and completion.choices[0].message.content:
                return completion.choices[0].message.content
        except Exception as e:
            last_error = e
            logger.warning("Groq model %s failed: %s. Trying fallback...", model_name, e)

    # If all models fail, return a structured safety fallback
    err_str = str(last_error) if last_error else "Unknown error"
    logger.error("All Groq models failed: %s", err_str)
    return (
        "### Preliminary Clinical Observation\n\n"
        "We are temporarily unable to contact the clinical inference model.\n\n"
        "**Clinical Guidance:**\n"
        "- Monitor symptom progression, severity, and any changes closely.\n"
        "- If symptoms are acute, sudden, or severe, seek evaluation at an urgent care or emergency casualty facility immediately.\n"
        "- Schedule a formal consultation with your primary physician for lingering or unexplained symptoms.\n\n"
        f"*(Notice: {err_str})*"
    )


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
    key = os.getenv("GROQ_API_KEY", DEFAULT_API_KEY).strip()
    is_key_valid = bool(key and not key.startswith("PASTE_"))
    return {
        "status": "healthy",
        "service": "AI Medical Assistant (Groq - India & Multilingual Edition)",
        "model_configured": PRIMARY_MODEL,
        "api_key_configured": is_key_valid,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_triage(payload: ChatRequest):
    """
    POST /api/chat:
    Accepts user symptoms, checks for acute red-flags (English & Telugu),
    calls Groq, and returns structured clinical guidance.
    """
    user_msg = payload.message.strip()
    if not user_msg:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty.")

    is_emergency = detect_emergency(user_msg)
    emergency_text = get_emergency_disclaimer(user_msg) if is_emergency else None

    # Construct messages array for Groq / OpenAI compatible API
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": CLINICAL_TRIAGE_SYSTEM_PROMPT}
    ]

    if payload.history:
        for turn in payload.history[-6:]:
            role = "user" if turn.role.lower() == "user" else "assistant"
            messages.append({"role": role, "content": turn.content})

    messages.append({"role": "user", "content": user_msg})

    try:
        ai_text = call_groq(messages)

        # Prepend emergency warning banner if red-flags were detected
        if is_emergency and emergency_text:
            ai_text = f"{emergency_text}\n\n---\n\n{ai_text}"

        return ChatResponse(
            response=ai_text,
            is_emergency=is_emergency,
            emergency_guidance=emergency_text,
        )

    except Exception as e:
        logger.exception("Error in /api/chat: %s", e)
        fallback_msg = (
            f"### Preliminary Clinical Observation\n\n"
            f"You described: *\"{user_msg[:120]}...\"*\n\n"
            f"Due to temporary service connectivity, please consider the following:\n"
            f"- If your symptoms are severe or worsening, seek prompt evaluation at an urgent care or emergency casualty department.\n"
            f"- For persistent or concerning symptoms, contact your primary care physician (MBBS/MD).\n"
            f"- Track symptom duration and severity to discuss with your provider.\n\n"
            f"*(Service connection note: {str(e)})*"
        )
        if is_emergency and emergency_text:
            fallback_msg = f"{emergency_text}\n\n---\n\n{fallback_msg}"
            
        return ChatResponse(
            response=fallback_msg,
            is_emergency=is_emergency,
            emergency_guidance=emergency_text,
        )


@app.post("/api/analyze-report", response_model=ReportResponse)
async def analyze_medical_report(payload: ReportRequest):
    """
    POST /api/analyze-report:
    Accepts raw medical/lab report text, processes it via Groq,
    and returns a plain-language explanation and physician consultation questions.
    """
    report_text = payload.report_text.strip()
    if not report_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Report text is required.")

    user_prompt = (
        f"Please analyze and explain the following clinical/lab report in clear, patient-friendly language:\n\n"
        f"--- BEGIN REPORT ---\n{report_text}\n--- END REPORT ---\n\n"
        f"Ensure you provide both the detailed plain-language breakdown and a curated list of specific questions "
        f"the patient should bring to their doctor in India."
    )

    messages = [
        {"role": "system", "content": REPORT_EXPLAINER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        analysis_text = call_groq(messages)

        # Extract suggested doctor questions using regex or fallback to standard questions
        extracted_questions = []
        questions_block = re.findall(r"(?:Questions|Doctor|ప్రశ్నలు).*", analysis_text, re.DOTALL | re.IGNORECASE)
        if questions_block:
            lines = questions_block[0].split("\n")
            for line in lines:
                cleaned = line.strip().lstrip("0123456789.-*• \t")
                if (cleaned.endswith("?") or cleaned.endswith("?**")) and len(cleaned) > 15:
                    extracted_questions.append(cleaned)
                    if len(extracted_questions) >= 5:
                        break

        if not extracted_questions:
            if is_telugu_query(report_text):
                extracted_questions = [
                    "ఈ ల్యాబ్ రిపోర్ట్ ఫలితాల ప్రకారం నా ఆరోగ్యం ఎలా ఉంది?",
                    "ఏవైనా విలువలు సాధారణ స్థాయి కన్నా ప్రమాదకరంగా ఎక్కువగా లేదా తక్కువగా ఉన్నాయా?",
                    "మళ్లీ ఈ పరీక్షలను ఎన్ని నెలల తర్వాత పునరావృతం చేయాలి?",
                    "నేను ఆహారం లేదా జీవనశైలిలో ఏమైనా మార్పులు చేసుకోవాలా?",
                    "ఏవైనా అదనపు స్కానింగ్ లేదా స్పెషలిస్ట్ డాక్టర్ సంప్రదింపులు అవసరమా?",
                ]
            else:
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
            f"1. Medical test reference intervals differ significantly across laboratory providers in India.\n"
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
    logger.info("Starting AI Medical Assistant server (Groq) on port %s...", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
