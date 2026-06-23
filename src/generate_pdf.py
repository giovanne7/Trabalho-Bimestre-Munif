"""Gera o relatorio PDF a partir do README.md.

Uso:
    python src/generate_pdf.py

Importante:
- Este script le o README.md atualizado.
- O PDF gerado fica na raiz do projeto com o nome relatorio.pdf.
- As imagens referenciadas no README, como results/comparacao_modelos.png,
  tambem sao inseridas no PDF se existirem.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Como este arquivo fica em src/, parents[1] volta para a raiz do projeto.
ROOT_DIR = Path(__file__).resolve().parents[1]

README_PATH = ROOT_DIR / "README.md"
PDF_PATH = ROOT_DIR / "relatorio.pdf"


def clean_text(text: str) -> str:
    """Remove espacos extras do texto."""
    return text.strip()


def format_inline_markdown(text: str) -> str:
    """Converte uma parte simples do Markdown para o formato aceito pelo ReportLab."""
    text = escape(text)

    # Negrito: **texto**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    # Italico simples: *texto*
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)

    # Codigo inline: `texto`
    text = re.sub(
        r"`([^`]+)`",
        r'<font name="Courier" backColor="#F3F4F6">\1</font>',
        text,
    )

    return text


def get_styles():
    """Cria os estilos usados no PDF."""
    base = getSampleStyleSheet()

    return {
        "Title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=10,
        ),
        "H1": ParagraphStyle(
            "Heading1Custom",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceBefore=8,
            spaceAfter=8,
            textColor=colors.HexColor("#111827"),
        ),
        "H2": ParagraphStyle(
            "Heading2Custom",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            spaceBefore=8,
            spaceAfter=6,
            textColor=colors.HexColor("#111827"),
        ),
        "H3": ParagraphStyle(
            "Heading3Custom",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            spaceBefore=6,
            spaceAfter=4,
            textColor=colors.HexColor("#111827"),
        ),
        "Body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "Bullet": ParagraphStyle(
            "BulletCustom",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            leftIndent=0.45 * cm,
            firstLineIndent=-0.25 * cm,
            spaceAfter=3,
        ),
        "Quote": ParagraphStyle(
            "QuoteCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9.5,
            leading=13,
            leftIndent=0.5 * cm,
            textColor=colors.HexColor("#374151"),
            borderColor=colors.HexColor("#D1D5DB"),
            borderWidth=1,
            borderPadding=5,
            spaceAfter=6,
        ),
        "Caption": ParagraphStyle(
            "CaptionCustom",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=7,
        ),
        "Code": ParagraphStyle(
            "CodeCustom",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            backColor=colors.HexColor("#F3F4F6"),
            borderColor=colors.HexColor("#E5E7EB"),
            borderWidth=0.5,
            borderPadding=5,
            leftIndent=0,
            rightIndent=0,
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


def add_paragraph(story, text: str, style) -> None:
    """Adiciona um paragrafo ao PDF."""
    text = clean_text(text)
    if text:
        story.append(Paragraph(format_inline_markdown(text), style))


def add_image(story, image_relative_path: str, caption: str, styles) -> None:
    """Adiciona imagem ao PDF respeitando a proporcao original."""
    image_path = ROOT_DIR / image_relative_path

    if not image_path.exists():
        add_paragraph(
            story,
            f"[Imagem não encontrada: {image_relative_path}]",
            styles["Body"],
        )
        return

    try:
        reader = ImageReader(str(image_path))
        original_width, original_height = reader.getSize()

        max_width = 16.0 * cm
        max_height = 10.5 * cm

        ratio = min(max_width / original_width, max_height / original_height)
        width = original_width * ratio
        height = original_height * ratio

        story.append(Spacer(1, 0.15 * cm))
        story.append(Image(str(image_path), width=width, height=height))
        story.append(Paragraph(format_inline_markdown(caption), styles["Caption"]))
    except Exception as error:
        add_paragraph(
            story,
            f"[Erro ao carregar imagem {image_relative_path}: {error}]",
            styles["Body"],
        )


def is_table_separator(line: str) -> bool:
    """Identifica a linha separadora de tabela Markdown."""
    cleaned = line.strip().replace("|", "").replace(":", "").replace("-", "").strip()
    return cleaned == ""


def parse_table(lines: List[str]) -> List[List[str]]:
    """Converte linhas de tabela Markdown em matriz para ReportLab."""
    table_data = []

    for line in lines:
        if is_table_separator(line):
            continue

        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        table_data.append([format_inline_markdown(cell) for cell in cells])

    return table_data


def add_table(story, table_lines: List[str], styles) -> None:
    """Adiciona uma tabela Markdown ao PDF."""
    data = parse_table(table_lines)

    if not data:
        return

    number_of_columns = len(data[0])
    available_width = 17.0 * cm
    col_widths = [available_width / number_of_columns] * number_of_columns

    table = Table(data, hAlign="LEFT", colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9CA3AF")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    story.append(Spacer(1, 0.12 * cm))
    story.append(table)
    story.append(Spacer(1, 0.25 * cm))


def markdown_to_story(markdown_text: str, styles):
    """Transforma o conteudo do README.md em elementos do ReportLab."""
    story = []
    lines = markdown_text.splitlines()

    i = 0
    in_code_block = False
    code_lines: List[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Bloco de codigo Markdown.
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lines = []
            else:
                in_code_block = False
                code_text = "\n".join(code_lines)
                story.append(Preformatted(code_text, styles["Code"]))
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # Linha horizontal do Markdown.
        if stripped in {"---", "***", "___"}:
            story.append(Spacer(1, 0.2 * cm))
            i += 1
            continue

        # Linha em branco.
        if not stripped:
            story.append(Spacer(1, 0.08 * cm))
            i += 1
            continue

        # Imagem Markdown: ![texto](caminho)
        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if image_match:
            caption = image_match.group(1) or "Imagem"
            image_path = image_match.group(2)
            add_image(story, image_path, caption, styles)
            i += 1
            continue

        # Tabela Markdown.
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_table(story, table_lines, styles)
            continue

        # Titulos.
        if stripped.startswith("# "):
            add_paragraph(story, stripped[2:], styles["Title"])
            i += 1
            continue

        if stripped.startswith("## "):
            add_paragraph(story, stripped[3:], styles["H2"])
            i += 1
            continue

        if stripped.startswith("### "):
            add_paragraph(story, stripped[4:], styles["H3"])
            i += 1
            continue

        # Citacao.
        if stripped.startswith("> "):
            add_paragraph(story, stripped[2:], styles["Quote"])
            i += 1
            continue

        # Lista com marcador.
        if stripped.startswith("- "):
            add_paragraph(story, "• " + stripped[2:], styles["Bullet"])
            i += 1
            continue

        # Lista numerada.
        if re.match(r"^\d+\.\s+", stripped):
            add_paragraph(story, stripped, styles["Bullet"])
            i += 1
            continue

        # Paragrafo comum.
        add_paragraph(story, stripped, styles["Body"])
        i += 1

    return story


def build_pdf() -> None:
    """Le o README.md e gera o arquivo relatorio.pdf."""
    if not README_PATH.exists():
        raise FileNotFoundError(f"README.md nao encontrado em: {README_PATH}")

    markdown_text = README_PATH.read_text(encoding="utf-8")
    styles = get_styles()
    story = markdown_to_story(markdown_text, styles)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Trabalho Final - Inteligencia Artificial",
        author="Unicesumar",
    )

    doc.build(story)
    print(f"PDF gerado com sucesso em: {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
