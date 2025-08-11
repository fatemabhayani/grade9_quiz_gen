
import os, json, time
import streamlit as st
import pandas as pd

# ---------- Settings ----------
st.set_page_config(page_title="Grade 9 Quiz Generator (Ontario)", layout="wide")
st.title("Ontario Grade 9 Math — Quiz Generator")
st.caption("Pick a skill → generate questions (easy/medium/hard) → copy or export")

# Load skills
@st.cache_data
def load_skills():
    skills = json.load(open("skills.json"))
    # Build a DataFrame for filtering by strand
    rows = []
    for k, v in skills.items():
        rows.append({"skill_id": k, **v})
    df = pd.DataFrame(rows)
    # Sort by strand + expectation_code
    df = df.sort_values(["strand", "expectation_code", "skill_id"]).reset_index(drop=True)
    return skills, df

skills, skills_df = load_skills()

# Sidebar: API key & model
st.sidebar.header("LLM Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Create at platform.openai.com → API Keys")
model = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
max_tokens = st.sidebar.slider("Max tokens", min_value=256, max_value=2000, value=600, step=64)

# Strand & Skill pickers
strand = st.selectbox("Strand", sorted(skills_df["strand"].unique().tolist()))
subset = skills_df[skills_df["strand"] == strand]
label_options = [f"{row.expectation_code} • {row.skill_id} — {row.expectation_text}" for _, row in subset.iterrows()]
choice = st.selectbox("Skill", label_options, index=0)
choice_row = subset.iloc[label_options.index(choice)]
skill_id = choice_row["skill_id"]
st.write("**Expectation:**", choice_row["expectation_text"])
st.write("**Notes:**", choice_row["notes"])

# Difficulty and count
colA, colB = st.columns(2)
with colA:
    difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=0)
with colB:
    n_q = st.number_input("How many questions?", min_value=1, max_value=10, value=3, step=1)

# Prompt builder
def build_prompt(skill_id, expectation_text, difficulty, n_q):
    return f"""You are a helpful Ontario Grade 9 math teacher.

Generate {n_q} {difficulty} questions for the skill {skill_id}.
Expectation: {expectation_text}

Output ONLY valid JSON exactly in this schema:
{{"questions":[{{"prompt":"...","answer":"..."}}, ...]}}. Keep numbers clean and avoid ambiguous wording.
"""

# Generate questions
if st.button("Generate"):
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    else:
        with st.spinner("Calling model..."):
            try:
                # Lazy import to avoid dependency if not used
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                prompt = build_prompt(skill_id, choice_row["expectation_text"], difficulty, n_q)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content":prompt}],
                    temperature=0.3,
                    max_tokens=max_tokens
                )
                raw = resp.choices[0].message.content
                try:
                    data = json.loads(raw)
                    qs = data.get("questions", [])
                except json.JSONDecodeError:
                    st.error("Model did not return valid JSON. You can try again.")
                    st.code(raw)
                    qs = []
                if qs:
                    st.success(f"Generated {len(qs)} question(s).")
                    for i, q in enumerate(qs, 1):
                        st.markdown(f"**Q{i}.** {q.get('prompt','')}")
                        with st.expander("Show answer"):
                            st.write(q.get("answer",""))
                    # Download JSON
                    st.download_button(
                        "Download as JSON",
                        data=json.dumps(data, indent=2),
                        file_name=f"{skill_id}_{difficulty}_questions.json",
                        mime="application/json"
                    )
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

st.divider()
st.subheader("Browse Skills")
st.dataframe(skills_df, use_container_width=True)
