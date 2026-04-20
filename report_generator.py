from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import os


def generate_report(label, dys_prob, non_dys_prob, gradcam_path):

    # 🔒 Force float conversion (absolute safety)
    dys_prob = float(dys_prob)
    non_dys_prob = float(non_dys_prob)

    report_path = "dyslexia_report.pdf"

    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "Dyslexia Detection Report")

    # Prediction
    c.setFont("Helvetica", 14)
    c.drawString(50, height - 100, f"Prediction: {label}")

    # Probabilities
    c.drawString(
        50,
        height - 150,
        f"Dyslexic Probability: {round(dys_prob * 100, 2)}%"
    )

    c.drawString(
        50,
        height - 180,
        f"Non-Dyslexic Probability: {round(non_dys_prob * 100, 2)}%"
    )

    # Add GradCAM Image
    if os.path.exists(gradcam_path):
        img = ImageReader(gradcam_path)
        c.drawImage(img, 50, height - 500, width=300, height=300)



    c.save()

    return report_path
    
'''
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import os
import datetime


def generate_report(label, dys_prob, non_dys_prob, gradcam_path):
    dys_prob = float(dys_prob)
    non_dys_prob = float(non_dys_prob)

    report_path = "dyslexia_report.pdf"
    c = canvas.Canvas(report_path, pagesize=letter)
    width, height = letter

    # ── Header bar ──
    c.setFillColorRGB(0.176, 0.416, 0.310)   # #2D6A4F green
    c.rect(0, height - 72, width, 72, fill=1, stroke=0)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(44, height - 36, "DysleXpert")
    c.setFont("Helvetica", 11)
    c.drawString(44, height - 54, "AI-Powered Dyslexia Detection Report")

    # Date top-right
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    c.setFont("Helvetica", 10)
    c.drawRightString(width - 44, height - 44, now)

    # ── Section: Prediction ──
    y = height - 110
    c.setFillColorRGB(0.102, 0.086, 0.055)  # #1A160E
    c.setFont("Helvetica-Bold", 14)
    c.drawString(44, y, "Prediction Result")
    y -= 6
    c.setStrokeColorRGB(0.176, 0.416, 0.310)
    c.setLineWidth(1.5)
    c.line(44, y, width - 44, y)
    y -= 20

    is_dyslexic = label.lower() == "dyslexic"
    badge_r, badge_g, badge_b = (0.784, 0.294, 0.192) if is_dyslexic else (0.176, 0.416, 0.310)

    # Prediction badge
    c.setFillColorRGB(badge_r, badge_g, badge_b)
    c.roundRect(44, y - 14, 160, 26, 5, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(124, y - 3, label)
    y -= 36

    # Probabilities
    c.setFillColorRGB(0.102, 0.086, 0.055)
    c.setFont("Helvetica", 11)

    def draw_prob_bar(cx, bar_y, label_text, prob, bar_color_rgb):
        c.setFillColorRGB(0.102, 0.086, 0.055)
        c.setFont("Helvetica", 11)
        c.drawString(cx, bar_y, f"{label_text}:")
        c.setFont("Helvetica-Bold", 11)
        c.drawString(cx + 130, bar_y, f"{round(prob * 100, 1)}%")

        # Track
        c.setFillColorRGB(0.941, 0.929, 0.902)   # light bg
        c.rect(cx, bar_y - 14, 300, 10, fill=1, stroke=0)
        # Fill
        c.setFillColorRGB(*bar_color_rgb)
        c.rect(cx, bar_y - 14, 300 * prob, 10, fill=1, stroke=0)

    draw_prob_bar(44, y, "Dyslexic", dys_prob, (0.784, 0.294, 0.192))
    y -= 30
    draw_prob_bar(44, y, "Non-Dyslexic", non_dys_prob, (0.176, 0.416, 0.310))
    y -= 40

    # ── Section: Grad-CAM ──
    c.setFillColorRGB(0.102, 0.086, 0.055)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(44, y, "Grad-CAM Heatmap")
    y -= 6
    c.setStrokeColorRGB(0.176, 0.416, 0.310)
    c.setLineWidth(1.5)
    c.line(44, y, width - 44, y)
    y -= 12

    if gradcam_path and os.path.exists(gradcam_path):
        img = ImageReader(gradcam_path)
        img_h = 220
        img_w = 280
        c.drawImage(img, 44, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True)

        # Legend label
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.360, 0.318, 0.251)
        c.drawString(44, y - img_h - 14, "Red = high diagnostic relevance · Blue = low relevance")
        y -= img_h + 30
    else:
        c.setFillColorRGB(0.784, 0.294, 0.192)
        c.setFont("Helvetica", 10)
        c.drawString(44, y - 16, "Grad-CAM image not available.")
        y -= 36

    # ── Section: Interpretation ──
    c.setFillColorRGB(0.102, 0.086, 0.055)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(44, y, "Clinical Interpretation")
    y -= 6
    c.setStrokeColorRGB(0.176, 0.416, 0.310)
    c.setLineWidth(1.5)
    c.line(44, y, width - 44, y)
    y -= 18

    interp = (
        "The model detected handwriting patterns associated with dyslexia. "
        "Irregular letter spacing, possible letter reversals, and baseline instability "
        "may be present. Early educational intervention is recommended. Please consult "
        "a qualified educational psychologist for a formal assessment."
        if is_dyslexic else
        "The handwriting sample displays consistent letter formation, regular spacing, "
        "and stable baseline alignment. No significant dyslexia markers were detected by "
        "the model. Continue regular literacy monitoring as part of routine educational practice."
    )

    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.360, 0.318, 0.251)
    # Word-wrap manually
    words = interp.split()
    line_words, line_width_sum = [], 0
    char_w = 5.5
    max_line = int((width - 88) / char_w)
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_line:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)

    for line in lines:
        c.drawString(44, y, line)
        y -= 14

    # ── Footer ──
    c.setFillColorRGB(0.941, 0.929, 0.902)
    c.rect(0, 0, width, 36, fill=1, stroke=0)
    c.setFillColorRGB(0.618, 0.568, 0.502)
    c.setFont("Helvetica", 8)
    c.drawString(44, 13, "DysleXpert · Maanakula Vinayagar Institute of Technology · Pondicherry University")
    c.drawRightString(width - 44, 13, "Research prototype — not a clinical diagnosis")

    c.save()
    return report_path'''
