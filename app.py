"""Multimodal Construction Intelligence Platform (Skill Wallet Capstone Project 1)

Run:  streamlit run app.py
Deps: streamlit, google-genai, pillow
"""
import io
import json
import urllib.parse

import streamlit as st
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:  # app still works in demo mode
    genai = None
    types = None

st.set_page_config(page_title="Construction Intelligence Platform", page_icon="🏗️", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
.hero{background:linear-gradient(120deg,#0f2027,#203a43,#2c5364);padding:1.4rem 1.8rem;border-radius:14px;color:#fff;margin-bottom:1rem}
.hero h1{margin:0;font-size:1.9rem}.hero p{margin:.3rem 0 0;opacity:.85}
.card{background:#f6f8fa;border:1px solid #e1e4e8;border-radius:12px;padding:1rem;text-align:center;color:#1f2933}
.card .v{font-size:1.7rem;font-weight:700}.card .l{font-size:.8rem;color:#586069;text-transform:uppercase;letter-spacing:.05em}
.pill{display:inline-block;padding:.2rem .75rem;border-radius:999px;font-weight:600;font-size:.85rem;color:#fff}
.alert{border-left:6px solid;border-radius:8px;padding:.8rem 1rem;margin:.5rem 0;color:#1f2933}
</style>
""",
    unsafe_allow_html=True,
)

SEV_COLORS = {"Low": "#2e9e5b", "Medium": "#e0a800", "High": "#e8590c", "Critical": "#c92a2a",
              "Compliant": "#2e9e5b", "Partially Compliant": "#e0a800", "Non-Compliant": "#c92a2a"}


def pill(text: str) -> str:
    return f'<span class="pill" style="background:{SEV_COLORS.get(text, "#495057")}">{text}</span>'


def card(label: str, value: str) -> str:
    return f'<div class="card"><div class="v">{value}</div><div class="l">{label}</div></div>'


# ----------------------------------------------------------------------------
# Gemini helpers
# ----------------------------------------------------------------------------
def get_client():
    key = st.session_state.get("api_key", "").strip()
    if key and genai:
        return genai.Client(api_key=key)
    return None


def analyze_image(image: Image.Image, prompt: str) -> dict:
    """Send image + prompt to Gemini, return parsed JSON. Raises on failure."""
    client = get_client()
    resp = client.models.generate_content(
        model=st.session_state.get("model", "gemini-2.5-flash"),
        contents=[prompt, image],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2),
    )
    text = (resp.text or "").strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)


def run_analysis(uploaded, prompt: str, demo: dict):
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    image.thumbnail((1600, 1600))
    st.image(image, caption="Uploaded image", use_container_width=True)
    if not get_client():
        st.info("Demo mode: showing a sample analysis. Add a Gemini API key in the sidebar for live results.")
        return demo
    try:
        with st.spinner("Analysing image with Gemini..."):
            return analyze_image(image, prompt)
    except json.JSONDecodeError:
        st.error("The model returned malformed JSON. Please retry.")
    except Exception as e:  # network, quota, invalid key...
        st.error(f"Analysis failed: {e}")
    return None


# ----------------------------------------------------------------------------
# Prompts and demo payloads (JSON contracts)
# ----------------------------------------------------------------------------
PPE_PROMPT = """You are a certified construction safety auditor (OSHA 1926 and Indian IS 456 / BOCW norms).
Analyse the site photo for PPE compliance of every visible worker: helmets, high-visibility vests,
steel-toe boots, safety harnesses (when working at height). Also identify site hazards.
Return ONLY JSON with this exact schema:
{"safety_score": int 0-100,
 "overall_status": "Compliant" | "Partially Compliant" | "Non-Compliant",
 "workers_detected": int,
 "ppe_detection": {"helmet": bool, "high_vis_vest": bool, "steel_toe_boots": bool, "safety_harness": bool},
 "hazards": [{"description": str, "severity": "Low"|"Medium"|"High"|"Critical", "remedy": str}],
 "summary": str}
If an item cannot be verified from the image, set it false and mention this in the summary."""

CRACK_PROMPT = """You are a structural engineer. Analyse the concrete/wall surface image.
Classify the defect as one of: Hairline, Settlement, Structural Shear, Thermal Expansion, Spalling.
Return ONLY JSON with this exact schema:
{"defect_type": str, "estimated_width_mm": float, "severity_score": int 1-10,
 "urgency": "Low"|"Medium"|"High"|"Critical",
 "root_causes": [str], "recommended_repairs": [str], "monitoring_advice": str,
 "disclaimer": str}
Widths are visual estimates only; recommend a site inspection by a licensed structural engineer."""

DEMO_PPE = {
    "safety_score": 62, "overall_status": "Partially Compliant", "workers_detected": 4,
    "ppe_detection": {"helmet": True, "high_vis_vest": True, "steel_toe_boots": False, "safety_harness": False},
    "hazards": [
        {"description": "Worker at height without a safety harness", "severity": "Critical",
         "remedy": "Stop work; provide full-body harness anchored to a rated anchor point."},
        {"description": "Open footwear observed on rebar area", "severity": "High",
         "remedy": "Enforce steel-toe boots at site entry checks."},
        {"description": "Loose materials on walkway", "severity": "Medium",
         "remedy": "Introduce housekeeping shifts and designated stacking zones."}],
    "summary": "Demo output: helmets and vests present; footwear and fall protection gaps found."}

DEMO_CRACK = {
    "defect_type": "Settlement", "estimated_width_mm": 3.5, "severity_score": 7, "urgency": "High",
    "root_causes": ["Differential foundation settlement", "Poor soil compaction", "Water ingress at footing"],
    "recommended_repairs": ["Epoxy injection for cracks above 0.3 mm", "Structural re-grouting",
                            "Foundation underpinning review", "Install crack gauges"],
    "monitoring_advice": "Fit tell-tale gauges and log width weekly for 8 weeks.",
    "disclaimer": "Demo output. Visual estimates only; consult a licensed structural engineer."}

# ----------------------------------------------------------------------------
# Cost / valuation model (indicative INR figures)
# ----------------------------------------------------------------------------
COST_PER_SQFT = {"Residential (RCC Framed)": 2200, "Commercial Building": 2900,
                 "Industrial Shed (PEB)": 1800, "Luxury Villa": 3800}
TIER_COST_FACTOR = {"Tier 1 (Mumbai, Delhi, Bengaluru)": 1.25, "Tier 2 (Indore, Pune, Jaipur)": 1.0,
                    "Tier 3 (Smaller cities)": 0.82}
TIER_MARKET_RATE = {"Tier 1 (Mumbai, Delhi, Bengaluru)": 9500, "Tier 2 (Indore, Pune, Jaipur)": 5200,
                    "Tier 3 (Smaller cities)": 3000}
STEEL_KG_PER_SQFT = {"Residential (RCC Framed)": 4.0, "Commercial Building": 5.0,
                     "Industrial Shed (PEB)": 6.5, "Luxury Villa": 4.5}
CEMENT_BAGS_PER_SQFT = 0.42
MILESTONES = [(0, "Site prep"), (10, "Foundation"), (30, "Plinth"), (50, "Superstructure"),
              (70, "Masonry"), (85, "Finishing"), (100, "Handover")]


def inr(amount: float) -> str:
    if amount >= 1e7:
        return f"₹{amount / 1e7:,.2f} Cr"
    return f"₹{amount / 1e5:,.2f} L"


WEATHER = {
    "Normal": ("#2e9e5b", ["Standard operations permitted.", "Run the daily toolbox talk and PPE checks.",
                           "Maintain routine curing: water-cure concrete for at least 7 days."]),
    "High Wind": ("#e8590c", ["Suspend crane and hoisting operations above 36 km/h wind speed.",
                              "Secure scaffolding sheets, formwork and loose materials.",
                              "Halt work at height; tie down tower crane jib to weathervane."]),
    "Heavy Rain": ("#1c7ed6", ["Electrical hazard: de-energise temporary supply, check earth leakage (RCCB).",
                               "Cover fresh concrete and stop pours; protect cement from moisture.",
                               "Pump out excavations; inspect trench shoring before re-entry."]),
    "Extreme Heat": ("#c92a2a", ["Concrete curing: start curing early, use wet hessian and pour at night or early morning.",
                                 "Provide ORS, shaded rest breaks and a 15 min/hr work-rest cycle above 40°C.",
                                 "Watch for heat stress; restrict heavy work 12:00-15:00."]),
}

SYSTEM_PROMPT = ("You are a senior Indian construction site engineer assistant. Give practical, concise advice "
                 "citing IS codes (IS 456, IS 800, IS 1893, NBC 2016) and OSHA where relevant. "
                 "Use INR for costs. State assumptions and recommend professional verification for structural decisions.")


def chat_reply(history: list) -> str:
    client = get_client()
    if not client:
        return ("(Demo mode) Add a Gemini API key for live answers. Rule of thumb: for M25 concrete, "
                "use roughly 1 : 1 : 2 (cement : sand : aggregate) by volume with w/c ratio ≤ 0.5.")
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    try:
        resp = client.models.generate_content(
            model=st.session_state.get("model", "gemini-2.5-flash"),
            contents=f"{SYSTEM_PROMPT}\n\n{convo}\nASSISTANT:")
        return resp.text or "No response received."
    except Exception as e:
        return f"Assistant error: {e}"


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("🔐 Configuration")
    st.session_state["api_key"] = st.text_input("Google Gemini API key", type="password",
                                                help="Leave blank for demonstration mode.")
    st.session_state["model"] = st.text_input("Gemini model", value="gemini-2.5-flash")
    live = bool(st.session_state["api_key"].strip()) and genai is not None
    st.success("Live mode" if live else "Demonstration mode")
    st.divider()
    st.subheader("🧭 Features")
    st.markdown("- Site Safety & PPE Audit\n- Crack Diagnostics\n- Planning & Valuation\n- Weather Advisory & Chatbot")
    st.subheader("🏛️ Architecture")
    st.caption("Streamlit UI → JSON data contracts → Gemini multimodal API / local cost engine → "
               "Pollinations image API. No database; session state only.")

st.markdown('<div class="hero"><h1>🏗️ Multimodal Construction Intelligence Platform</h1>'
            '<p>PPE safety audits, crack diagnostics, cost and valuation in INR, weather-aware site advisory</p></div>',
            unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🦺 Safety & PPE Audit", "🧱 Crack Diagnostics",
                                  "📐 Planning & Valuation", "🌦️ Site Assistant"])

# ----------------------------------------------------------------------------
# Tab 1: PPE audit
# ----------------------------------------------------------------------------
with tab1:
    up = st.file_uploader("Upload a construction site photo", type=["jpg", "jpeg", "png"], key="ppe")
    if up:
        result = run_analysis(up, PPE_PROMPT, DEMO_PPE)
        if result:
            try:
                c = st.columns(3)
                c[0].markdown(card("Safety score", f"{result['safety_score']}/100"), unsafe_allow_html=True)
                c[1].markdown(card("Workers detected", str(result.get("workers_detected", "-"))), unsafe_allow_html=True)
                c[2].markdown(card("Status", pill(result["overall_status"])), unsafe_allow_html=True)
                st.progress(max(0, min(100, int(result["safety_score"]))) / 100)
                st.subheader("PPE checklist")
                labels = {"helmet": "Helmet", "high_vis_vest": "High-vis vest",
                          "steel_toe_boots": "Steel-toe boots", "safety_harness": "Safety harness"}
                cols = st.columns(4)
                for col, (k, name) in zip(cols, labels.items()):
                    ok = result["ppe_detection"].get(k, False)
                    col.markdown(card(name, "✅" if ok else "❌"), unsafe_allow_html=True)
                st.subheader("Identified hazards")
                for h in result.get("hazards", []):
                    color = SEV_COLORS.get(h["severity"], "#495057")
                    st.markdown(f'<div class="alert" style="border-color:{color};background:{color}18">'
                                f'{pill(h["severity"])} <b>{h["description"]}</b><br>🔧 {h["remedy"]}</div>',
                                unsafe_allow_html=True)
                st.write(result.get("summary", ""))
                st.download_button("⬇️ Download JSON report", json.dumps(result, indent=2),
                                   "ppe_audit_report.json", "application/json")
            except (KeyError, TypeError) as e:
                st.error(f"Unexpected response format (missing {e}). Please retry.")
    else:
        st.info("Upload a site photo to begin the audit.")

# ----------------------------------------------------------------------------
# Tab 2: Crack diagnostics
# ----------------------------------------------------------------------------
with tab2:
    up = st.file_uploader("Upload a concrete or wall surface photo", type=["jpg", "jpeg", "png"], key="crack")
    if up:
        result = run_analysis(up, CRACK_PROMPT, DEMO_CRACK)
        if result:
            try:
                c = st.columns(4)
                c[0].markdown(card("Defect type", result["defect_type"]), unsafe_allow_html=True)
                c[1].markdown(card("Est. width", f"{result['estimated_width_mm']} mm"), unsafe_allow_html=True)
                c[2].markdown(card("Severity", f"{result['severity_score']}/10"), unsafe_allow_html=True)
                c[3].markdown(card("Urgency", pill(result["urgency"])), unsafe_allow_html=True)
                l, r = st.columns(2)
                with l:
                    st.subheader("Root causes")
                    for x in result["root_causes"]:
                        st.markdown(f"- {x}")
                with r:
                    st.subheader("Recommended repairs")
                    for x in result["recommended_repairs"]:
                        st.markdown(f"- {x}")
                st.info(f"📈 Monitoring: {result.get('monitoring_advice', 'N/A')}")
                st.caption(result.get("disclaimer", "Consult a licensed structural engineer."))
                st.download_button("⬇️ Download JSON report", json.dumps(result, indent=2),
                                   "crack_report.json", "application/json")
            except (KeyError, TypeError) as e:
                st.error(f"Unexpected response format (missing {e}). Please retry.")
    else:
        st.info("Upload a surface photo to classify the defect.")

# ----------------------------------------------------------------------------
# Tab 3: Planning & valuation
# ----------------------------------------------------------------------------
with tab3:
    a, b = st.columns(2)
    stype = a.selectbox("Structure type", list(COST_PER_SQFT))
    tier = a.selectbox("Location tier", list(TIER_COST_FACTOR))
    area = b.number_input("Built-up area (sq ft)", 200, 500000, 2000, step=100)
    pct = b.slider("Completion (%)", 0, 100, 40)

    budget = area * COST_PER_SQFT[stype] * TIER_COST_FACTOR[tier]
    capex = budget * pct / 100
    market_full = area * TIER_MARKET_RATE[tier] * (1.3 if stype == "Luxury Villa" else 1.0)
    # Asset value is weighted: under-construction assets realise a share of completed value
    market_now = market_full * (0.25 + 0.75 * pct / 100) * pct / 100 if pct < 100 else market_full
    steel_t = area * STEEL_KG_PER_SQFT[stype] / 1000
    cement = area * CEMENT_BAGS_PER_SQFT

    c = st.columns(3)
    c[0].markdown(card("Total project budget", inr(budget)), unsafe_allow_html=True)
    c[1].markdown(card("Capex incurred", inr(capex)), unsafe_allow_html=True)
    c[2].markdown(card("Current asset valuation", inr(market_now)), unsafe_allow_html=True)
    st.write("")
    c = st.columns(3)
    c[0].markdown(card("Steel requirement", f"{steel_t:,.1f} tonnes"), unsafe_allow_html=True)
    c[1].markdown(card("Cement requirement", f"{cement:,.0f} bags"), unsafe_allow_html=True)
    c[2].markdown(card("Value on completion", inr(market_full)), unsafe_allow_html=True)

    st.subheader("Milestone progress")
    st.progress(pct / 100)
    current = max((m for m in MILESTONES if m[0] <= pct), key=lambda m: m[0])
    st.caption(" → ".join(f"**{n}**" if n == current[1] else n for _, n in MILESTONES))
    st.caption("Indicative estimates only; actual costs vary with specifications, material prices and market conditions.")

    st.subheader("🎨 Conceptual 3D rendering (Pollinations AI)")
    default_prompt = (f"photorealistic 3D architectural rendering of a {stype.lower()}, "
                      f"{area} sq ft, modern Indian design, golden hour, ultra detailed")
    img_prompt = st.text_area("Rendering prompt", default_prompt)
    if st.button("Generate rendering"):
        url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(img_prompt)}?width=1024&height=640&nologo=true"
        st.image(url, caption="Concept render (AI generated)", use_container_width=True)
        st.code(url, language=None)

# ----------------------------------------------------------------------------
# Tab 4: Weather advisory + chatbot
# ----------------------------------------------------------------------------
with tab4:
    st.subheader("Weather-aware safety protocol")
    w = st.radio("Today's condition", list(WEATHER), horizontal=True)
    color, alerts = WEATHER[w]
    for msg in alerts:
        st.markdown(f'<div class="alert" style="border-color:{color};background:{color}18">{msg}</div>',
                    unsafe_allow_html=True)

    st.subheader("💬 Construction engineering assistant")
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant",
                                  "content": "Hello! Ask me about concrete mix, scheduling, IS codes, safety or costing."}]
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if q := st.chat_input("Ask a construction question..."):
        st.session_state.chat.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                ans = chat_reply(st.session_state.chat[-12:] + [])
            st.markdown(ans)
        st.session_state.chat.append({"role": "assistant", "content": ans})
    if st.button("Clear chat"):
        st.session_state.pop("chat", None)
        st.rerun()
