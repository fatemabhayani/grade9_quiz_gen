
Ontario Math Worksheet Generator

An AI-powered worksheet generation tool for Grades 1–9 Ontario Mathematics Curriculum, covering strands B–F.
Teachers can select a grade, strand, and skill, choose difficulty, and instantly generate a pedagogically sound worksheet with a clean, print-ready PDF.
✨ Features

    Curriculum-Aligned
    Fully mapped to Ontario’s Mathematics Curriculum (B–F strands, Grades 1–9).

    Smart Question Generation
    Uses OpenAI API to produce knowledge, application, thinking, and communication questions — each designed to take ≤5 minutes.

    Difficulty Levels
    Easy / Medium / Hard modes.

    PDF Export

        Professional header (student name space)

        Space for working out answers

        Expectation description for each skill

        Answer key in small font on a separate page

    Interactive Selection Flow

        Pick a grade

        Pick a strand

        Pick a skill

        Choose difficulty & generate

    Clean, Sleek Design
    Worksheets styled as if crafted by a world-class educator.

🚀 Planned Enhancements

    Teacher Branding: Upload logo + school name → auto-styled header/footer

    Theme Presets: Classic, Grid, Exam

    Item Quality Checks: Reject ambiguous/multi-answer items or >5-min solutions; log failed items

    Worked Examples: Optional inclusion (cognitive load theory)

    UDL & Accessibility: Dyslexia-friendly fonts, larger line spacing, alt-text, printable grids

    Hints & Scaffolds: Toggle hints and check steps, embed in margins or “teacher version”

    Multi-Skill Worksheets: Target multiple expectations at once

    Split-Grade Worksheets: Create one worksheet usable by two consecutive grades

    Whitespace Control: Option to remove lines and keep blank space only

    Question Type Selector: Word problems, drills, multiple choice

    Content Moderation: PII/profanity filter, block unsafe topics

    Math Validation: (Planned) SymPy integration to verify numeric/algebraic correctness — even for some word problems

🛠 Tech Stack

    Frontend/UI: Streamlit

    Backend: Python + OpenAI API

    PDF Generation: ReportLab / FPDF

    Data Source: CSV of Ontario Curriculum expectations (B–F strands, Grades 1–9)

    Validation: (Planned) SymPy

⚡ Installation

    Clone the repo

git clone https://github.com/yourusername/ontario-math-worksheet-gen.git
cd ontario-math-worksheet-gen

Create a virtual environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

Install dependencies

pip install -r requirements.txt

Add your OpenAI API key
Create a .env file:

    OPENAI_API_KEY=your_api_key_here

▶ Usage

Run the app locally:

streamlit run app.py

    Select Grade → Strand → Skill → Difficulty

    Generate and preview questions

    Export to PDF with answer key

🤝 Contributing

We welcome contributions!
Planned contributions include:

    Adding SymPy validation logic

    Expanding curriculum mapping

    Improving PDF layouts and themes

    Enhancing accessibility features

📜 License

MIT License — feel free to use and adapt with attribution.
