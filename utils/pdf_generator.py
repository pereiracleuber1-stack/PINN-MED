import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime

def generate_clinical_report(patient_id, operator, model_name, metrics, chart_path, output_filename):
    doc = SimpleDocTemplate(output_filename, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    
    # Cabeçalho
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor("#1e3d59"), alignment=1)
    story.append(Paragraph("SGP-PINN ENTERPRISE | LAUDO DE FISIOLOGIA COMPUTACIONAL", title_style))
    story.append(Spacer(1, 10))
    
    meta_info = [
        [Paragraph(f"<b>ID do Paciente:</b> {patient_id}", styles['Normal']), Paragraph(f"<b>Data/Hora (UTC):</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])],
        [Paragraph(f"<b>Operador/CRM:</b> {operator}", styles['Normal']), Paragraph(f"<b>Modelo Utilizado:</b> {model_name}", styles['Normal'])],
    ]
    t_meta = Table(meta_info, colWidths=[260, 260])
    t_meta.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f5f5f5")), ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#cccccc")), ('PADDING', (0,0), (-1,-1), 6)]))
    story.append(t_meta)
    story.append(Spacer(1, 15))
    
    # Tabela de Parâmetros Calibrados
    story.append(Paragraph("<b>Métricas e Parâmetros Calibrados pela Rede:</b>", styles['Heading3']))
    table_data = [["Parâmetro / Biomarcador", "Valor Estimado", "Status Clínico"]]
    for k, v in metrics.items():
        table_data.append([k, str(v.get("valor", "-")), str(v.get("status", "Normal"))])
    
    t_metrics = Table(table_data, colWidths=[200, 160, 160])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e3d59")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ('PADDING', (0, 0), (-1, -1), 5)
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 15))
    
    # Gráfico de Projeção
    if os.path.exists(chart_path):
        story.append(Paragraph("<b>Projeção Dinâmica Fisiológica Integrada:</b>", styles['Heading3']))
        story.append(Image(chart_path, width=520, height=230))
        story.append(Spacer(1, 15))
        
    # Rodapé / Declaração
    footer_text = "<i>Aviso: Este laudo foi gerado por Inteligência Artificial Informada por Leis Fisiológicas como ferramenta de apoio à decisão clínica. A intervenção final é de responsabilidade médica exclusiva.</i>"
    story.append(Paragraph(footer_text, styles['Italic']))
    
    doc.build(story)
    return output_filename
