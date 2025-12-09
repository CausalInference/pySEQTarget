import datetime


def _build_md(self, img_path: str = None) -> str:
    """
    Builds markdown content for SEQuential analysis results.

    :param self: SEQoutput instance
    :param img_path: Path to saved KM graph image (if any)
    :return: Markdown string
    """

    lines = []

    lines.append(f"# SEQuential Analysis: {datetime.date.today()}: {self.method}")
    lines.append("")

    if self.options.weighted:
        lines.append("## Weighting")
        lines.append("")

        lines.append("### Numerator Model")
        lines.append("")
        lines.append("```")
        lines.append(str(self.numerator_models[0].summary()))
        lines.append("```")
        lines.append("")

        lines.append("### Denominator Model")
        lines.append("")
        lines.append("```")
        lines.append(str(self.denominator_models[0].summary()))
        lines.append("```")
        lines.append("")

        if self.options.compevent_colname is not None and self.compevent_models:
            lines.append("### Competing Event Model")
            lines.append("")
            lines.append("```")
            lines.append(str(self.compevent_models[0].summary()))
            lines.append("```")
            lines.append("")

        lines.append("### Weighting Statistics")
        lines.append("")
        lines.append(self.weight_statistics.to_pandas().to_markdown(index=False))
        lines.append("")

    lines.append("## Outcome")
    lines.append("")

    lines.append("### Outcome Model")
    lines.append("")
    lines.append("```")
    lines.append(str(self.outcome_models[0].summary()))
    lines.append("```")
    lines.append("")

    if self.options.hazard_estimate and self.hazard is not None:
        lines.append("### Hazard")
        lines.append("")
        lines.append(self.hazard.to_pandas().to_markdown(index=False))
        lines.append("")

    if self.options.km_curves:
        lines.append("### Survival")
        lines.append("")

        if self.risk_difference is not None:
            lines.append("#### Risk Differences")
            lines.append("")
            lines.append(self.risk_difference.to_pandas().to_markdown(index=False))
            lines.append("")

        if self.risk_ratio is not None:
            lines.append("#### Risk Ratios")
            lines.append("")
            lines.append(self.risk_ratio.to_pandas().to_markdown(index=False))
            lines.append("")

        if self.km_graph is not None and img_path is not None:
            lines.append("#### Survival Curves")
            lines.append("")
            lines.append(f"![Kaplan-Meier Survival Curves]({img_path})")
            lines.append("")

    if self.diagnostic_tables:
        lines.append("## Diagnostic Tables")
        lines.append("")
        for name, table in self.diagnostic_tables.items():
            lines.append(f"### {name.replace('_', ' ').title()}")
            lines.append("")
            lines.append(table.to_pandas().to_markdown(index=False))
            lines.append("")

    return "\n".join(lines)


def _build_pdf(md_content: str, filename: str, img_path: str = None) -> None:
    """
    Converts markdown content to PDF.

    :param md_content: Markdown string
    :param filename: Output PDF path
    :param img_path: Absolute path to image file (if any)
    """
    try:
        import markdown
        from weasyprint import CSS, HTML
    except ImportError:
        raise ImportError(
            "PDF generation requires 'markdown' and 'weasyprint'. "
            "Install with: pip install markdown weasyprint"
        )

    html_content = markdown.markdown(md_content, extensions=["tables", "fenced_code"])

    if img_path:
        img_name = img_path.split("/")[-1]
        html_content = html_content.replace(
            f'src="{img_name}"', f'src="file://{img_path}"'
        )

    css = CSS(
        string="""
        body {
            font-family: Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.4;
            margin: 2cm;
        }
        h1 { color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 0.3em; }
        h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 0.2em; }
        h3 { color: #7f8c8d; }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
        }
        th, td {
            border: 1px solid #bdc3c7;
            padding: 8px;
            text-align: left;
        }
        th { background-color: #ecf0f1; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        pre {
            background-color: #f4f4f4;
            padding: 1em;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 9pt;
        }
        code { font-family: 'Courier New', monospace; }
        img { max-width: 100%; height: auto; }
    """
    )

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body>{html_content}</body>
    </html>
    """

    HTML(string=full_html).write_pdf(filename, stylesheets=[css])
