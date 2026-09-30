
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"

ARIAL_REGULAR = r"C:\Windows\Fonts\arial.ttf"
ARIAL_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
ARIAL_ITALIC = r"C:\Windows\Fonts\ariali.ttf"


class MediCheckReport(FPDF):
    def __init__(self):
        super().__init__()

        # Register Unicode-capable Arial fonts.
        self.add_font("Arial", "", ARIAL_REGULAR)
        self.add_font("Arial", "B", ARIAL_BOLD)
        self.add_font("Arial", "I", ARIAL_ITALIC)

    def header(self):
        self.set_font("Arial", "B", 16)
        self.set_text_color(30, 158, 117)
        self.cell(
            0,
            10,
            "MediCheck - Symptom Report",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )

        self.set_font("Arial", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(
            0,
            6,
            f"Generated on {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            align="C",
            new_x="LMARGIN",
            new_y="NEXT",
        )

        self.ln(2)
        self.set_draw_color(30, 158, 117)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(
            0,
            10,
            "Disclaimer: This report is AI-generated and not a substitute for professional medical advice.",
            align="C",
        )


def generate_pdf(symptoms, predictions, age_group, gender, duration):
    pdf = MediCheckReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(
        0,
        8,
        "Patient Context",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(
        0,
        7,
        f"Age Group: {age_group}    Gender: {gender}    Symptom Duration: {duration}",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(3)

    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(
        0,
        8,
        "Symptoms Detected",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(80, 80, 80)
    symptom_text = ", ".join(
        [s.replace("_", " ").title() for s in symptoms]
    )
    pdf.multi_cell(0, 7, symptom_text)
    pdf.ln(3)

    severity = predictions[0]["severity"]
    risk_flag = predictions[0].get("risk_flag", False)

    severity_colors = {
        "Urgent": (220, 53, 69),
        "Moderate": (255, 165, 0),
        "Mild": (30, 158, 117),
    }

    r, g, b = severity_colors.get(
        severity,
        (80, 80, 80),
    )

    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(40, 8, "Severity Level:")

    pdf.set_text_color(r, g, b)
    pdf.cell(
        0,
        8,
        f"  {severity}",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    if risk_flag:
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(220, 53, 69)
        pdf.multi_cell(
            0,
            6,
            "! One or more reported symptoms are commonly associated with medical "
            "emergencies. Please seek immediate medical attention.",
        )

    pdf.ln(3)

    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(
        0,
        8,
        "Top Possible Conditions",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(2)

    for i, pred in enumerate(predictions):
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(30, 158, 117)
        pdf.cell(
            0,
            7,
            f"{i + 1}. {pred['disease']}  "
            f"({pred['confidence']}% symptom match)",
            new_x="LMARGIN",
            new_y="NEXT",
        )

        pdf.set_font("Arial", "", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 6, pred["description"])

        if pred["precautions"]:
            pdf.set_font("Arial", "B", 9)
            pdf.set_text_color(60, 60, 60)
            pdf.cell(
                0,
                6,
                "Precautions:",
                new_x="LMARGIN",
                new_y="NEXT",
            )

            pdf.set_font("Arial", "", 9)
            pdf.set_text_color(80, 80, 80)

            for p in pred["precautions"]:
                pdf.cell(
                    0,
                    6,
                    f"  - {p}",
                    new_x="LMARGIN",
                    new_y="NEXT",
                )

        pdf.set_font("Arial", "B", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(
            40,
            6,
            "Recommended specialist:",
        )

        pdf.set_font("Arial", "", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(
            0,
            6,
            f"  {pred['specialist']}",
            new_x="LMARGIN",
            new_y="NEXT",
        )

        pdf.ln(3)

    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(
        0,
        5,
        "This report is generated by MediCheck, an AI-powered symptom checker. "
        "It is not a medical diagnosis. Please consult a qualified healthcare "
        "professional for proper evaluation and treatment.",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / (
        f"medicheck_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )

    pdf.output(str(path))

    return str(path)
