
import os, io, json, textwrap, datetime
import streamlit as st
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors

from typing import List, Dict, Any

st.set_page_config(page_title="K–9 Worksheet Builder (Ontario)", layout="wide")
st.title("K–9 Worksheet Builder (Ontario Math)")
st.caption("Select grade → strand → skill, generate questions with an LLM, curate, and export a polished PDF worksheet.")

@st.cache_data
def load_skills():
    with open("skills.json", "r") as f:
        nested = json.load(f)
    rows = []
    for grade, strands in nested.items():
        for strand, skills in strands.items():
            for s in skills:
                rows.append({
                    "grade": s.get("grade", grade),
                    "strand": s.get("strand", strand),
                    "skill_id": s.get("skill_id", ""),
                    "expectation_code": s.get("expectation_code", ""),
                    "expectation_text": s.get("expectation_text", ""),
                    "notes": s.get("notes", ""),
                })
    df = pd.DataFrame(rows).sort_values(["grade","strand","expectation_code","skill_id"]).reset_index(drop=True)
    return nested, df

nested, skills_df = load_skills()

st.sidebar.header("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
difficulty = st.sidebar.selectbox("Difficulty", ["easy", "medium", "hard"], index=0)
n_questions = st.sidebar.slider("Number of questions", 4, 12, 6, 1)
include_types = st.sidebar.multiselect(
    "Question types (aim for a mix)",
    ["Knowledge","Application","Thinking","Communication"],
    default=["Knowledge","Application","Thinking","Communication"]
)
student_name = st.sidebar.text_input("Student name (for PDF header)", "")
footer_note = st.sidebar.text_input("Footer note (optional)", "Ontario Math • Worksheet")

left, right = st.columns([2,1])
with left:
    grades = sorted(skills_df["grade"].astype(str).unique().tolist())
    grade = st.selectbox("Grade", grades, index=grades.index("9") if "9" in grades else 0)

    strands = sorted(skills_df[skills_df["grade"].astype(str)==str(grade)]["strand"].unique().tolist())
    strand = st.selectbox("Strand", strands)

    subset = skills_df[(skills_df["grade"].astype(str)==str(grade)) & (skills_df["strand"]==strand)]
    labels = [f"{row.expectation_code} • {row.skill_id} — {row.expectation_text[:120]}" for _, row in subset.iterrows()]
    skill_choice = st.selectbox("Skill", labels)

    srow = subset.iloc[labels.index(skill_choice)]
    st.markdown(f"**Expectation ({srow['expectation_code']}):** {srow['expectation_text']}")
    if srow.get("notes"):
        st.caption(f"Notes: {srow['notes']}")

with right:
    st.subheader("How it works")
    st.write("1. Pick grade/strand/skill.\n2. Generate a balanced set of questions.\n3. Select the ones you like.\n4. Export a PDF worksheet with answers on a separate page.")
    st.info("Tip: keep questions solvable within ~5 minutes each.")

def build_prompt(grade: str, strand: str, skill_id: str, expectation: str, difficulty: str, n: int, types: List[str]) -> str:
    type_str = ", ".join(types)
    schema = {
        "questions":[
            {"type":"Knowledge|Application|Thinking|Communication",
             "prompt":"question text",
             "answer":"succinct correct answer",
             "est_time_min":"int <= 5"}
        ]
    }
    return f"""You are a world-class Ontario math teacher and assessment designer.
Grade: {grade}. Strand: {strand}. Skill: {skill_id}.
Expectation: {expectation}

Create {n} {difficulty} questions covering a mix of these categories: {type_str}.
Each question must be solvable in 5 minutes or less by a typical student, with clean numbers and unambiguous wording.

Output ONLY valid JSON matching this schema exactly:
{json.dumps(schema, indent=2)}

Rules:
- Balance the categories across the set; include 'type' for each question.
- Keep language student-friendly.
- No multi-part questions beyond (a)/(b).
- Provide concise 'answer' strings.
"""

col1, col2 = st.columns(2)
with col1:
    if st.button("Generate Questions", use_container_width=True):
        if not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
        else:
            with st.spinner("Generating..."):
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    prompt = build_prompt(str(grade), strand, srow["skill_id"], srow["expectation_text"], difficulty, int(n_questions), include_types)
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role":"user","content":prompt}],
                        temperature=0.2,
                        max_tokens=1200
                    )
                    raw = resp.choices[0].message.content
                    st.session_state["raw_output"] = raw
                    try:
                        data = json.loads(raw)
                        st.session_state["questions"] = data.get("questions", [])
                    except json.JSONDecodeError:
                        st.error("The model did not return valid JSON. Please try again.")
                        st.code(raw)
                except Exception as e:
                    st.error(f"Error: {e}")

with col2:
    if "questions" in st.session_state and st.session_state["questions"]:
        st.success(f"{len(st.session_state['questions'])} question(s) ready.")

selected = []
if "questions" in st.session_state and st.session_state["questions"]:
    st.subheader("Review & Select Questions")
    for i, q in enumerate(st.session_state["questions"], 1):
        with st.container(border=True):
            cols = st.columns([0.07, 0.63, 0.3])
            pick = cols[0].checkbox(f"{i}", value=True, key=f"pick_{i}")
            cols[1].markdown(f"**[{q.get('type','N/A')}] Q{i}.** {q.get('prompt','')}")
            cols[2].markdown(f"**Expected time:** {q.get('est_time_min', '≤5')} min  \n**Answer (peek):** {q.get('answer','')[:60]}{'...' if len(q.get('answer',''))>60 else ''}")
            if pick:
                selected.append(q)

    st.write(f"Selected: **{len(selected)}** / {len(st.session_state['questions'])}")

def build_pdf(questions: List[Dict[str,Any]], meta: Dict[str,str]) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin = 0.75*inch
    y = height - margin
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, meta.get("title", "Math Worksheet"))
    c.setFont("Helvetica", 10)
    y -= 14
    c.drawString(margin, y, f"Grade {meta.get('grade','')} • Strand: {meta.get('strand','')} • Skill: {meta.get('skill_id','')} ({meta.get('expectation_code','')})")
    y -= 12
    c.drawString(margin, y, f"Expectation: {meta.get('expectation_text','')[:1000]}")
    y -= 16
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(margin, y, f"Student Name: {meta.get('student_name','___________________________')}")
    y -= 18
    c.setStrokeColor(colors.grey)
    c.line(margin, y, width-margin, y)
    y -= 12

    c.setFont("Helvetica", 11)
    q_space = 0.6*inch
    for i, q in enumerate(questions, 1):
        prompt = f"{i}. [{q.get('type','')}] {q.get('prompt','')}"
        wrapped = textwrap.wrap(prompt, width=95)
        for line in wrapped:
            if y < margin + q_space:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 11)
            c.drawString(margin, y, line)
            y -= 14
        lines_needed = int((q_space)/14)
        c.setStrokeColor(colors.lightgrey)
        for _ in range(lines_needed):
            if y < margin + 20:
                c.showPage()
                y = height - margin
            c.line(margin, y, width - margin, y)
            y -= 14
        y -= 6

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.grey)
    c.drawString(margin, 0.5*inch, meta.get("footer", ""))
    c.drawRightString(width - margin, 0.5*inch, datetime.date.today().strftime("%b %d, %Y"))

    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.black)
    c.drawString(margin, height - margin, "Answer Key")
    c.setFont("Helvetica", 10)
    yy = height - margin - 18
    for i, q in enumerate(questions, 1):
        ans = f"{i}. {q.get('answer','')}"
        lines = textwrap.wrap(ans, width=100)
        for line in lines:
            if yy < 0.75*inch:
                c.showPage()
                c.setFont("Helvetica", 10)
                yy = height - margin
            c.drawString(margin, yy, line)
            yy -= 12

    c.save()
    buffer.seek(0)
    return buffer.read()

if selected:
    meta = {
        "title": "Ontario Math Worksheet",
        "grade": str(grade),
        "strand": strand,
        "skill_id": srow["skill_id"],
        "expectation_code": srow["expectation_code"],
        "expectation_text": srow["expectation_text"],
        "student_name": student_name,
        "footer": footer_note
    }
    pdf_bytes = build_pdf(selected, meta)
    st.download_button(
        "⬇️ Download Worksheet (PDF)",
        data=pdf_bytes,
        file_name=f"worksheet_G{grade}_{srow['skill_id']}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

st.divider()
st.subheader("All skills (filterable)")
st.dataframe(skills_df, use_container_width=True)
