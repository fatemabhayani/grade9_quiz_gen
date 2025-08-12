import os, io, json, re, textwrap, datetime
from typing import List, Dict, Any, Tuple
import streamlit as st
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors

# ------------------------- UI Setup -------------------------
st.set_page_config(page_title="Ontario Worksheet Builder — Pro", layout="wide")
st.title("Ontario Worksheet Builder — Pro")
st.caption("Curriculum-aligned worksheets with pedagogy-aware features, branding, and accessibility options.")

# ------------------------- Load Skills -------------------------
@st.cache_data
def load_skills():
    with open("skills.json","r") as f:
        nested = json.load(f)
    rows = []
    for grade, strands in nested.items():
        for strand, skills in strands.items():
            for s in skills:
                rows.append({
                    "grade": str(s.get("grade", grade)),
                    "strand": s.get("strand", strand),
                    "skill_id": s.get("skill_id",""),
                    "expectation_code": s.get("expectation_code",""),
                    "expectation_text": s.get("expectation_text",""),
                    "notes": s.get("notes",""),
                })
    df = pd.DataFrame(rows).sort_values(["grade","strand","expectation_code","skill_id"]).reset_index(drop=True)
    return nested, df

nested, skills_df = load_skills()

# ------------------------- Sidebar Settings -------------------------
st.sidebar.header("LLM & Generation")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
model = st.sidebar.selectbox("Model", ["gpt-4o-mini","gpt-4o"], index=0)
difficulty = st.sidebar.selectbox("Difficulty", ["easy","medium","hard"], index=1)
n_questions = st.sidebar.slider("Total questions", 4, 20, 8, 1)
question_formats = st.sidebar.multiselect(
    "Question format(s)",
    ["word problem","drill","multiple choice"],
    default=["word problem","drill"]
)
types_mix = st.sidebar.multiselect(
    "Purpose mix (K/A/T/C)",
    ["Knowledge","Application","Thinking","Communication"],
    default=["Knowledge","Application","Thinking","Communication"]
)

st.sidebar.header("Branding")
school_name = st.sidebar.text_input("School name", "")
logo_file = st.sidebar.file_uploader("Upload logo (PNG/JPG)", type=["png","jpg","jpeg"])

st.sidebar.header("Worksheet Theme")
theme = st.sidebar.selectbox("Theme", ["Classic (lined)","Grid (graph)","Exam (blank)"], index=0)
line_spacing = st.sidebar.slider("Workspace spacing", 10, 24, 16, 1)
remove_lines = st.sidebar.checkbox("Use whitespace only (no lines/grid)", value=False)

st.sidebar.header("Options")
include_worked = st.sidebar.checkbox("Include 1–2 worked examples", value=False)
include_hints = st.sidebar.checkbox("Include hint & check-step (teacher version)", value=True)
split_grade_mode = st.sidebar.checkbox("Split-grade worksheet (choose two grades)", value=False)

st.sidebar.header("Privacy & Safety")
pii_guard = st.sidebar.checkbox("Strip PII from prompts (recommended)", value=True)

# ------------------------- Grade/Strand/Skill Selection -------------------------
colA, colB = st.columns([2,1])

with colA:
    all_grades = sorted(skills_df["grade"].astype(str).unique().tolist(), key=lambda x: (len(x), x))
    if split_grade_mode and len(all_grades) >= 2:
        grade_pair = st.multiselect("Select two grades", all_grades, max_selections=2)
        if len(grade_pair) != 2:
            st.info("Pick exactly two grades for split-grade mode.")
    else:
        grade = st.selectbox("Grade", all_grades, index=all_grades.index("9") if "9" in all_grades else 0)

    # Strand selector
    if split_grade_mode and len(all_grades) >= 2 and len(st.session_state.get('grade_pair', grade_pair if 'grade_pair' in locals() else []))==2:
        chosen_grades = grade_pair
        subset = skills_df[skills_df["grade"].astype(str).isin(chosen_grades)]
        strands = sorted(subset["strand"].unique().tolist())
        strand = st.selectbox("Strand (optional filter)", ["All"] + strands, index=0)
        if strand != "All":
            subset = subset[subset["strand"]==strand]
        st.caption("Select multiple skills across the chosen grades.")
        skill_labels = [f"G{row.grade} • {row.expectation_code} • {row.skill_id} — {row.expectation_text[:80]}" for _, row in subset.iterrows()]
        selected_indices = st.multiselect("Skills (multi-select)", list(range(len(subset))), format_func=lambda i: skill_labels[i] if i < len(skill_labels) else "")
        chosen_rows = subset.iloc[selected_indices] if selected_indices else pd.DataFrame(columns=subset.columns)
    else:
        # Single grade path
        strands = sorted(skills_df[skills_df["grade"].astype(str)==str(grade)]["strand"].unique().tolist())
        strand = st.selectbox("Strand", strands)
        subset = skills_df[(skills_df["grade"].astype(str)==str(grade)) & (skills_df["strand"]==strand)]
        skill_labels = [f"{row.expectation_code} • {row.skill_id} — {row.expectation_text[:100]}" for _, row in subset.iterrows()]
        multi_skill = st.checkbox("Select multiple skills", value=False)
        if multi_skill:
            selected_indices = st.multiselect("Skills (multi-select)", list(range(len(subset))), format_func=lambda i: skill_labels[i] if i < len(skill_labels) else "")
            chosen_rows = subset.iloc[selected_indices] if selected_indices else pd.DataFrame(columns=subset.columns)
        else:
            skill_choice = st.selectbox("Skill", skill_labels)
            chosen_rows = subset.iloc[[skill_labels.index(skill_choice)]]

    if not chosen_rows.empty:
        st.markdown("**Selected expectations:**")
        for _, r in chosen_rows.iterrows():
            st.write(f"- **{r['expectation_code']} ({r['skill_id']})**: {r['expectation_text']}")

with colB:
    st.subheader("Quality Gate (Heuristics)")
    st.write("Generated items will be checked for:")
    st.markdown("- Ambiguity (e.g., 'choose all that apply' unless MCQ)")
    st.markdown("- Unrealistic numbers for selected grade(s)")
    st.markdown("- Estimated time > 5 minutes")
    st.markdown("- Profanity/unsafe content")

# ------------------------- Quality & Moderation -------------------------
BADWORDS = re.compile(r"\b(fuck|shit|sex|porn|kill|suicide|racist)\b", re.IGNORECASE)

def extract_numbers(text: str) -> List[float]:
    nums = re.findall(r"-?\d+\.?\d*", text)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except:
            pass
    return out

def quality_check(q: Dict[str,Any], grades: List[str], allow_mcq: bool=True) -> Tuple[bool, str]:
    # est time
    if str(q.get("est_time_min","")).strip():
        try:
            t = float(q.get("est_time_min"))
            if t > 5:
                return False, "est_time_min > 5"
        except:
            pass
    # profanity
    if BADWORDS.search(q.get("prompt","")) or BADWORDS.search(q.get("answer","")):
        return False, "profanity/unsafe language"
    # ambiguity
    prompt = q.get("prompt","").lower()
    if ("choose all that apply" in prompt or "multiple answers" in prompt) and not (allow_mcq and q.get("choices")):
        return False, "ambiguous selection (not MCQ)"
    # unrealistic numbers (simple guardrail)
    nums = extract_numbers(q.get("prompt",""))
    if nums:
        maxn = max(abs(x) for x in nums)
        try:
            max_grade = max(int(g) for g in grades if str(g).isdigit())
        except:
            max_grade = 9
        limit = 1e6 if max_grade >= 9 else 1e4 if max_grade >= 6 else 1e3
        if maxn > limit:
            return False, f"unrealistic number ({int(maxn)} > {int(limit)})"
    # MCQ structure integrity
    if q.get("type","").lower().startswith("multiple") or (q.get("format","")=="multiple choice"):
        choices = q.get("choices", [])
        if not (isinstance(choices, list) and 3 <= len(choices) <= 6):
            return False, "MCQ missing/invalid choices"
        ans = str(q.get("answer","")).strip()
        if not ans:
            return False, "missing answer"
    return True, ""

# ------------------------- Prompt Builder -------------------------
def build_prompt(chosen_rows: pd.DataFrame, difficulty: str, n: int, question_formats: List[str], types_mix: List[str], split_grade: bool) -> str:
    # Build context from selected skills
    skills_list = []
    grades = sorted(list(set(chosen_rows["grade"].astype(str).tolist())))
    for _, r in chosen_rows.iterrows():
        skills_list.append({
            "grade": str(r["grade"]),
            "strand": r["strand"],
            "skill_id": r["skill_id"],
            "expectation_code": r["expectation_code"],
            "expectation_text": r["expectation_text"]
        })
    schema = {
        "questions":[
            {
                "skill_id":"...",
                "type":"Knowledge|Application|Thinking|Communication",
                "format":"word problem|drill|multiple choice",
                "prompt":"question text",
                "choices":["A) ...","B) ...","C) ...","D) ..."],  # optional for MCQ
                "hint":"one short hint",
                "check_step":"one short self-check step",
                "answer":"succinct correct answer or correct option label",
                "est_time_min": "int <= 5"
            }
        ]
    }
    formats_str = ", ".join(question_formats)
    types_str = ", ".join(types_mix)
    mode_text = "Design items that are solvable by both grades without dumbing down content." if split_grade else "Design items appropriate for the selected grade."
    return f"""You are a world-class Ontario math teacher and assessment designer.
Selected skills:\n{json.dumps(skills_list, indent=2)}

Goal: Create {n} {difficulty} questions total, distributed across the selected skills.
Formats to include: {formats_str}.
Purpose mix to balance across the set: {types_str}.
Constraints:
- Each question must be solvable within 5 minutes.
- Clean numbers, unambiguous wording. Avoid culture-specific context unless relevant to math.
- For MCQ, provide 4–5 clear choices and ensure one correct answer.
- Include 'hint' and 'check_step' fields, concise (1 sentence each).
- {mode_text}
- DO NOT include any personally identifiable information (PII).

Output ONLY valid JSON exactly in this schema:
{json.dumps(schema, indent=2)}
"""

# ------------------------- Generation -------------------------
gen_col1, gen_col2 = st.columns([1,1])
with gen_col1:
    can_generate = api_key and not chosen_rows.empty
    if st.button("Generate Questions", use_container_width=True, disabled=not can_generate):
        if not api_key:
            st.warning("Enter API key.")
        elif chosen_rows.empty:
            st.warning("Select at least one skill.")
        else:
            with st.spinner("Generating…"):
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                prompt = build_prompt(chosen_rows, difficulty, int(n_questions), question_formats, types_mix, split_grade_mode)
                # PII guard: we never add student_name etc. to prompt.
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.2,
                    max_tokens=1600
                )
                raw = resp.choices[0].message.content
                st.session_state["raw_output"] = raw
                try:
                    data = json.loads(raw)
                    qs = data.get("questions", [])
                except json.JSONDecodeError:
                    st.error("Model did not return valid JSON.")
                    st.code(raw)
                    qs = []
                # Apply quality checks
                failed = []
                passed = []
                grades_used = sorted(list(set(chosen_rows["grade"].astype(str).tolist())))
                for q in qs:
                    ok, reason = quality_check(q, grades_used, allow_mcq=True)
                    if ok:
                        passed.append(q)
                    else:
                        q["_fail_reason"] = reason
                        failed.append(q)
                st.session_state["questions"] = passed
                st.session_state["failed"] = failed

with gen_col2:
    if "questions" in st.session_state:
        st.success(f"Ready: {len(st.session_state['questions'])}  •  Filtered out: {len(st.session_state.get('failed',[]))}")
        if st.session_state.get("failed"):
            df_failed = pd.DataFrame(st.session_state["failed"])
            st.download_button("Download failed items log (CSV)", data=df_failed.to_csv(index=False), file_name="failed_items.csv", mime="text/csv", use_container_width=True)

# ------------------------- Review & Selection -------------------------
selected = []
if "questions" in st.session_state and st.session_state["questions"]:
    st.subheader("Review & Select")
    for i, q in enumerate(st.session_state["questions"], 1):
        with st.container(border=True):
            cols = st.columns([0.06, 0.6, 0.34])
            pick = cols[0].checkbox(f"{i}", value=True, key=f"pick_{i}")
            head = f"**[{q.get('type','')}/{q.get('format','')}] Q{i}.** {q.get('prompt','')}"
            cols[1].markdown(head)
            meta = f"**Time:** {q.get('est_time_min','≤5')} min"
            cols[2].markdown(meta)
            if q.get("choices"):
                cols[1].markdown("\n".join([f"- {c}" for c in q.get("choices",[])]))
            if pick:
                selected.append(q)

    st.write(f"Selected: **{len(selected)}** / {len(st.session_state['questions'])}")

# ------------------------- PDF Rendering -------------------------
def draw_logo(c, logo_bytes, x, y, max_w=1.2*inch, max_h=0.8*inch):
    try:
        from reportlab.lib.utils import ImageReader
        img = ImageReader(logo_bytes)
        iw, ih = img.getSize()
        scale = min(max_w/iw, max_h/ih)
        c.drawImage(img, x, y-ih*scale, width=iw*scale, height=ih*scale, mask='auto')
    except Exception:
        pass

def draw_grid(c, width, height, margin, spacing=14):
    c.setStrokeColor(colors.lightgrey)
    y = height - margin - 20
    while y > margin:
        c.line(margin, y, width-margin, y)
        y -= spacing

def draw_graph_grid(c, width, height, margin, cell=12):
    c.setStrokeColor(colors.lightgrey)
    y = height - margin - 20
    while y > margin:
        c.line(margin, y, width-margin, y); y -= cell
    x = margin
    while x < width - margin:
        c.line(x, height - margin - 20, x, margin)
        x += cell

def build_pdf(questions: List[Dict[str,Any]], meta: Dict[str,str], logo_file, theme: str, line_spacing: int, remove_lines: bool, include_hints: bool, include_worked: bool) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 0.75*inch

    # Header
    y = height - margin
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 14)
    title = meta.get("title","Ontario Math Worksheet")
    c.drawString(margin, y, title)
    if logo_file:
        draw_logo(c, logo_file, width - margin - 1.2*inch, y+10, max_w=1.2*inch, max_h=0.6*inch)
    c.setFont("Helvetica", 10)
    y -= 14
    header_line = f"{meta.get('school','')}  •  Grade {meta.get('grade','')}  •  Strand: {meta.get('strand','')}"
    if meta.get("split_grades"):
        header_line = f"{meta.get('school','')}  •  Split Grade {meta.get('split_grades')}  •  {meta.get('strand','')}"
    c.drawString(margin, y, header_line)
    y -= 12
    c.drawString(margin, y, f"Skill(s): {meta.get('skill_line','')}")
    y -= 12
    c.drawString(margin, y, f"Expectation: {meta.get('expectation','')[:1000]}")
    y -= 14
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(margin, y, "Student Name: ___________________________    Date: ____________")
    y -= 16
    c.setStrokeColor(colors.grey)
    c.line(margin, y, width-margin, y)
    y -= 10

    # Worked examples (optional)
    if include_worked and meta.get("worked"):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin, y, "Worked Example(s)")
        y -= 14
        c.setFont("Helvetica", 10)
        for we in meta["worked"]:
            for line in textwrap.wrap(we, width=100):
                if y < margin + 80:
                    c.showPage(); y = height - margin
                    c.setFont("Helvetica", 10)
                c.drawString(margin, y, line); y -= 12
            y -= 6
        c.setStrokeColor(colors.grey); c.line(margin, y, width-margin, y); y -= 8

    # Questions
    c.setFont("Helvetica", 11)
    space = line_spacing  # workspace density control
    for i, q in enumerate(questions, 1):
        prompt = f"{i}. [{q.get('type','')}/{q.get('format','')}] {q.get('prompt','')}"
        lines = textwrap.wrap(prompt, width=95)
        for line in lines:
            if y < margin + 80:
                c.showPage(); y = height - margin; c.setFont("Helvetica", 11)
            c.drawString(margin, y, line); y -= 14
        # MCQ choices
        choices = q.get("choices", [])
        if choices:
            c.setFont("Helvetica", 10)
            for choice in choices:
                for line in textwrap.wrap(str(choice), width=95):
                    if y < margin + 80:
                        c.showPage(); y = height - margin; c.setFont("Helvetica", 10)
                    c.drawString(margin + 16, y, line); y -= 12
            c.setFont("Helvetica", 11)
        # Workspace
        if not remove_lines:
            if theme.startswith("Classic"):
                draw_grid(c, width, height, margin, spacing=space)
            elif theme.startswith("Grid"):
                draw_graph_grid(c, width, height, margin, cell=max(8, space-2))
            elif theme.startswith("Exam"):
                # No extra lines; just whitespace
                pass
        y -= max(40, space*2)

    # Answers / teacher page
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - margin, "Teacher Version — Answers" + (" + Hints" if include_hints else ""))
    yy = height - margin - 16
    c.setFont("Helvetica", 9)
    for i, q in enumerate(questions, 1):
        ans = f"{i}. {q.get('answer','')}"
        if include_hints:
            hint = q.get("hint",""); chk = q.get("check_step","")
            ans += f"  |  Hint: {hint}  |  Check: {chk}"
        for line in textwrap.wrap(ans, width=110):
            if yy < 0.75*inch:
                c.showPage(); c.setFont("Helvetica", 9); yy = height - margin
            c.drawString(margin, yy, line); yy -= 12

    c.save()
    buffer.seek(0)
    return buffer.read()

# Generate worked examples (optional) via LLM, after questions are chosen
def build_worked_examples(api_key: str, model: str, chosen_rows: pd.DataFrame, count: int = 2) -> List[str]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        skills = [
            {
                "grade": str(r["grade"]),
                "strand": r["strand"],
                "skill_id": r["skill_id"],
                "expectation_code": r["expectation_code"],
                "expectation_text": r["expectation_text"]
            } for _, r in chosen_rows.iterrows()
        ]
        prompt = f"""Create {count} concise worked examples (step-by-step) for these Ontario math skills. 
Keep each example within 5 steps, with clean numbers and a short concluding sentence.
Skills:\n{json.dumps(skills, indent=2)}"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=600
        )
        raw = resp.choices[0].message.content
        # Simple parsing: split by blank lines or numbering
        examples = [s.strip("- ").strip() for s in re.split(r"\n\s*\n|^\d+\.\s*", raw, flags=re.MULTILINE) if s.strip()]
        return examples[:count]
    except Exception:
        return []

# ------------------------- Export -------------------------
if selected:
    # Build meta
    if split_grade_mode and 'grade_pair' in locals() and len(grade_pair)==2:
        skill_line = ", ".join([f"{r['skill_id']} ({r['expectation_code']})" for _, r in chosen_rows.iterrows()])
        meta = {
            "title": "Ontario Math Worksheet",
            "school": school_name,
            "grade": "",
            "split_grades": f"{grade_pair[0]}–{grade_pair[1]}",
            "strand": strand if 'strand' in locals() else "Mixed",
            "skill_line": skill_line,
            "expectation": "; ".join(chosen_rows["expectation_text"].astype(str).tolist()[:2]) + (" ..." if len(chosen_rows)>2 else "")
        }
    else:
        skill_line = ", ".join([f"{r['skill_id']} ({r['expectation_code']})" for _, r in chosen_rows.iterrows()])
        meta = {
            "title": "Ontario Math Worksheet",
            "school": school_name,
            "grade": str(grade) if 'grade' in locals() else "",
            "strand": strand if 'strand' in locals() else "Mixed",
            "skill_line": skill_line,
            "expectation": "; ".join(chosen_rows["expectation_text"].astype(str).tolist()[:2]) + (" ..." if len(chosen_rows)>2 else "")
        }
    # Worked examples (optional)
    worked_examples = []
    if include_worked and api_key:
        worked_examples = build_worked_examples(api_key, model, chosen_rows, count=2)
        if worked_examples:
            meta["worked"] = worked_examples

    logo_bytes = None
    if logo_file is not None:
        logo_bytes = io.BytesIO(logo_file.read())

    pdf_bytes = build_pdf(selected, meta, logo_bytes, theme, int(line_spacing), remove_lines, include_hints, include_worked and bool(worked_examples))
    st.download_button("⬇️ Download Worksheet (PDF)", data=pdf_bytes, file_name="worksheet.pdf", mime="application/pdf", use_container_width=True)

st.divider()
st.subheader("All Skills (filterable)")
st.dataframe(skills_df, use_container_width=True)



'''import os, io, json, textwrap, datetime
import streamlit as st
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors

from typing import List, Dict, Any

st.set_page_config(page_title="Grade 1–9 Math Question Generator (Ontario)", layout="wide")
st.title("Grade 1–9 Question Generator (Ontario Math)")
st.caption("Select grade → strand → skill, generate questions, curate, and export a polished PDF worksheet.")

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
'''