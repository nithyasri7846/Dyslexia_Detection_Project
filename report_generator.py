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
