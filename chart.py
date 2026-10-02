"""Dependency-free SVG chart for the prefix-length sweep."""

from pathlib import Path


def save_sweep_chart(statistics: dict, prefix_lengths: tuple[int, ...], path: str) -> None:
    width, height = 900, 620
    colors = {"different_prefix": "#d97706", "shared_prefix": "#2563eb"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Median TTFT and prompt evaluation time by prefix length">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font:14px Arial,sans-serif;fill:#1f2937}.title{font-size:22px;font-weight:bold}'
        '.panel{font-size:17px;font-weight:bold}.axis{stroke:#9ca3af;stroke-width:1}'
        '.grid{stroke:#e5e7eb;stroke-width:1}</style>',
        '<text x="60" y="35" class="title">Shared-prefix sweep: median latency</text>',
    ]
    for scenario, label in (("different_prefix", "Different prefix"), ("shared_prefix", "Shared prefix")):
        x = 570 if scenario == "different_prefix" else 740
        parts.append(f'<line x1="{x}" y1="52" x2="{x + 28}" y2="52" stroke="{colors[scenario]}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 35}" y="57">{label}</text>')

    for panel_index, (field, title) in enumerate((("ttft_s", "Time to first token"), ("prefill_s", "Prompt evaluation time"))):
        top = 105 + panel_index * 270
        left, right, bottom = 75, 850, top + 195
        values = [
            statistics[str(length)][scenario][field]["median"]
            for length in prefix_lengths
            for scenario in colors
        ]
        ymax = max((value for value in values if value is not None), default=0) * 1.15 or 1
        parts.append(f'<text x="75" y="{top - 20}" class="panel">{title} (seconds)</text>')
        for tick in range(5):
            y = bottom - tick * (bottom - top) / 4
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="65" y="{y + 5:.1f}" text-anchor="end">{ymax * tick / 4:.1f}</text>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
        low, high = min(prefix_lengths), max(prefix_lengths)
        def x_position(length):
            return left + (length - low) / (high - low) * (right - left) if high > low else (left + right) / 2
        for length in prefix_lengths:
            x = x_position(length)
            parts.append(f'<text x="{x:.1f}" y="{bottom + 21}" text-anchor="middle">{length}</text>')
        for scenario, color in colors.items():
            points = []
            for length in prefix_lengths:
                value = statistics[str(length)][scenario][field]["median"]
                if value is not None:
                    x = x_position(length)
                    y = bottom - value / ymax * (bottom - top)
                    points.append((x, y))
            if len(points) > 1:
                coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    parts.append('<text x="450" y="608" text-anchor="middle">Configured prefix length (lines)</text></svg>')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts), encoding="utf-8")
