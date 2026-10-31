"""Test PDFs for the reader (tools/device_test.sh): a two-column paper with running headers and page numbers, a resume
with a date column, a book chapter with dialogue, a one-page report with a table. Needs reportlab.
    python3 android/tools/make_test_pdfs.py OUT_DIR"""
import os, sys
os.makedirs(sys.argv[1] if len(sys.argv) > 1 else ".", exist_ok=True)
os.chdir(sys.argv[1] if len(sys.argv) > 1 else ".")
from reportlab.lib.pagesizes import letter
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle, FrameBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=10, leading=13, alignment=4)
W, H = letter
LOREM = ("Speech synthesis on small devices has improved quickly in recent years. Models that once needed a server "
         "now run in real time on a phone, which changes how assistants can behave. In this section we describe the "
         "architecture, the training data and the evaluation protocol, and we compare the results with earlier work. ")

def header_footer(title):
    def f(c, d):
        c.saveState(); c.setFont("Helvetica", 8)
        c.drawString(inch, H - 0.6 * inch, title); c.drawRightString(W - inch, H - 0.6 * inch, "Preprint, October 2026")
        c.drawCentredString(W / 2, 0.5 * inch, str(d.page)); c.restoreState()
    return f

# 1. two-column paper
doc = BaseDocTemplate("paper.pdf", pagesize=letter)
gap = 0.3 * inch; cw = (W - 2 * inch - gap) / 2
top = Frame(inch, H - 2.6 * inch, W - 2 * inch, 1.6 * inch, id="top")
l1 = Frame(inch, inch, cw, H - 3.7 * inch, id="l"); r1 = Frame(inch + cw + gap, inch, cw, H - 3.7 * inch, id="r")
l2 = Frame(inch, inch, cw, H - 2 * inch, id="l2"); r2 = Frame(inch + cw + gap, inch, cw, H - 2 * inch, id="r2")
doc.addPageTemplates([PageTemplate("first", [top, l1, r1], onPage=header_footer("Expressive Speech on the Edge")),
                      PageTemplate("rest", [l2, r2], onPage=header_footer("Expressive Speech on the Edge"))])
from reportlab.platypus import NextPageTemplate
s = [Paragraph("Expressive Speech on the Edge", ParagraphStyle("t", parent=ss["Title"], fontSize=20)),
     Paragraph("Ava Chen and Leo Park, Whissle Research", ParagraphStyle("a", parent=ss["Normal"], alignment=1)),
     Spacer(1, 8),
     Paragraph("<b>Abstract.</b> We present a compact text-to-speech model that follows prosody controls on a phone. "
               "It reaches a first sound in under sixty milliseconds.", body), FrameBreak(), NextPageTemplate("rest"),
     Paragraph("1 Introduction", ss["Heading2"])]
for i in range(3): s.append(Paragraph(LOREM + "The approach is simple and the gains are large. ", body))
s.append(Paragraph("2 Method", ss["Heading2"]))
for i in range(7): s.append(Paragraph(LOREM, body))
s.append(Paragraph("3 Results", ss["Heading2"]))
for i in range(5): s.append(Paragraph(LOREM + "Table 1 lists the measurements for every device we tried. ", body))
doc.build(s)

# 2. resume with a date column
doc = BaseDocTemplate("resume.pdf", pagesize=letter)
doc.addPageTemplates([PageTemplate("p", [Frame(inch * 0.8, inch * 0.8, W - 1.6 * inch, H - 1.6 * inch)])])
st = [Paragraph("Jordan Rivera", ParagraphStyle("n", parent=ss["Title"], fontSize=24, alignment=0)),
      Paragraph("Product Designer · jordan@example.com · (555) 010-2030", ss["Normal"]), Spacer(1, 10),
      Paragraph("Experience", ss["Heading2"])]
for role, place, dates, pts in [("Senior Designer", "Acme Corp", "2022 – Present", ["Led the redesign of the mobile app, raising retention by 18%.", "Built a design system used by six teams."]),
                                 ("Designer", "Northwind", "2019 – 2022", ["Shipped the onboarding flow for two million users.", "Ran weekly research sessions with customers."])]:
    t = Table([[Paragraph(f"<b>{role}</b>, {place}", ss["Normal"]), Paragraph(dates, ParagraphStyle("d", parent=ss["Normal"], alignment=2))]], colWidths=[4.6 * inch, 2.2 * inch])
    st.append(t)
    for p in pts: st.append(Paragraph("• " + p, ss["Normal"]))
    st.append(Spacer(1, 6))
st.append(Paragraph("Skills", ss["Heading2"])); st.append(Paragraph("Figma, prototyping, user research, accessibility.", ss["Normal"]))
doc.build(st)

# 3. a book chapter over three pages
doc = BaseDocTemplate("book.pdf", pagesize=(5.5 * inch, 8.5 * inch))
doc.addPageTemplates([PageTemplate("p", [Frame(0.7 * inch, 0.8 * inch, 4.1 * inch, 6.9 * inch)], onPage=lambda c, d: (c.setFont("Times-Italic", 8), c.drawCentredString(2.75 * inch, 8.0 * inch, "The Harbour"), c.drawCentredString(2.75 * inch, 0.45 * inch, str(d.page))))])
bk = ParagraphStyle("bk", fontName="Times-Roman", fontSize=11, leading=15, firstLineIndent=14, alignment=4)
s = [Paragraph("Chapter One", ParagraphStyle("c", fontName="Times-Bold", fontSize=18, leading=24, spaceAfter=14))]
paras = ["It was a quiet morning when Mrs. Hall opened the shop. The bell above the door rang once, and the street outside was still wet from the night's rain.",
         "“Good morning,” she said to nobody in particular. “Another slow day, I suppose.”",
         "At 7:45 the first boat came in, carrying 1,200 kg of fish and three tired men who had been at sea since Tuesday.",
         "“Run! They're coming!” shouted a boy from the pier, and everyone turned to look at the water.",
         "She did not run. She had seen enough storms to know that this one would pass, and that the harbour would be <i>exactly</i> as it was before."]
for k in range(4):
    for p in paras: s.append(Paragraph(p, bk))
doc.build(s)

# 4. a report with a table
doc = BaseDocTemplate("report.pdf", pagesize=letter)
doc.addPageTemplates([PageTemplate("p", [Frame(inch, inch, W - 2 * inch, H - 2 * inch)], onPage=header_footer("Quarterly Report"))])
s = [Paragraph("Quarterly Report", ss["Title"]), Paragraph("Summary", ss["Heading2"]),
     Paragraph("Revenue grew 12% to $4.2M, driven by the new subscription plan. Costs were flat.", body),
     Table([["Region", "Q2", "Q3", "Change"], ["North", "$1.1M", "$1.3M", "+18%"], ["South", "$0.9M", "$1.0M", "+11%"]], style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)])),
     Spacer(1, 12), Paragraph("Outlook", ss["Heading2"]),
     Paragraph("We expect growth to continue next quarter as the plan reaches more customers.", body)]
doc.build(s)
print("ok")
