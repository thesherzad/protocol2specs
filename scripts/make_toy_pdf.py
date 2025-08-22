from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# -----------------------------------------
# Content for sections
# -----------------------------------------
content = """
Clinical Study Protocol – Example

Section 3.1: Laboratory Assessments
AST (Aspartate Aminotransferase, also called SGOT) is a liver enzyme. 
Definition: AST measures the concentration of aspartate aminotransferase in serum.
Units: U/L (Units per Liter).

ALT (Alanine Aminotransferase, also called SGPT) is another liver enzyme.
Definition: ALT measures the concentration of alanine aminotransferase in serum.
Units: U/L.

BILI (Bilirubin, also called TBIL).
Definition: BILI measures the concentration of bilirubin in serum.
Units: U/L.

Section 4.2: Vital Signs
Systolic Blood Pressure (SYSBP) is defined as the maximum arterial pressure during contraction of the left ventricle.
Units: mmHg.

Diastolic Blood Pressure (DIABP) is defined as the minimum arterial pressure during relaxation of the heart.
Units: mmHg.
"""

# -----------------------------------------
# Define toy tables (visit schedule, dose escalation)
# -----------------------------------------
visit_schedule = [
    ["Visit", "Day", "Window", "Assessments"],
    ["Screening", "-28 to -1", "±2", "Informed Consent, Labs, Vitals"],
    ["Visit 1", "0", "±1", "Dosing, Labs, Vitals"],
    ["Visit 2", "7", "±2", "Labs, Vitals"],
    ["Visit 3", "14", "±2", "Labs, Vitals, ECG"],
]

dose_escalation = [
    ["Cohort", "Dose Level (mg)", "Number of Subjects", "Escalation Criteria"],
    ["1", "10", "6", "If ≤1/6 subjects experience DLT"],
    ["2", "20", "6", "If ≤1/6 subjects experience DLT"],
    ["3", "40", "6", "If ≤1/6 subjects experience DLT"],
]

# -----------------------------------------
# PDF build
# -----------------------------------------
out = Path("data/Protocol.pdf")
out.parent.mkdir(exist_ok=True)

doc = SimpleDocTemplate(str(out))
styles = getSampleStyleSheet()
story = []

# Add text paragraphs
for para in content.strip().split("\n\n"):
    story.append(Paragraph(para, styles["Normal"]))
    story.append(Spacer(1, 12))

# Add Visit Schedule Table
story.append(Paragraph("Section 5.1: Visit Schedule", styles["Heading2"]))
table1 = Table(visit_schedule, hAlign="LEFT")
table1.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
]))
story.append(table1)
story.append(Spacer(1, 20))

# Add Dose Escalation Table
story.append(Paragraph("Section 6.2: Dose Escalation", styles["Heading2"]))
table2 = Table(dose_escalation, hAlign="LEFT")
table2.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
    ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
]))
story.append(table2)

doc.build(story)
print(f"Realistic toy protocol with tables created at {out}")
