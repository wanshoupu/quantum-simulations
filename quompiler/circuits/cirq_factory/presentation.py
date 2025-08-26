import re
import cirq
from cirq.contrib.svg import circuit_to_svg


def cirq2svg(circuit: cirq.Circuit, moment_size=15, spacing=50):
    num_chunks = (len(circuit) + moment_size - 1) // moment_size

    chunk_sizes = []
    chunks = []

    for i in range(num_chunks):
        subcircuit = circuit[i * moment_size: (i + 1) * moment_size]
        svg = circuit_to_svg(subcircuit)
        chunks.append(svg)

        # Try width/height first
        match_wh = re.search(r'width="([0-9.]+)[a-z]*"\s+height="([0-9.]+)[a-z]*"', svg)
        if match_wh:
            w, h = map(float, match_wh.groups())
            chunk_sizes.append((w, h))
            continue

        # Fallback: parse viewBox="0 0 W H"
        match_vb = re.search(r'viewBox="0 0 ([0-9.]+) ([0-9.]+)"', svg)
        if match_vb:
            w, h = map(float, match_vb.groups())
            chunk_sizes.append((w, h))
            continue

        raise ValueError("Could not parse width/height or viewBox from SVG")

    total_width = max(w for w, _ in chunk_sizes)
    total_height = sum(h for _, h in chunk_sizes) + spacing * (num_chunks - 1)

    # Root SVG with explicit width/height
    svg_output = [f'<svg xmlns="http://www.w3.org/2000/svg" '
                  f'width="{total_width}" height="{total_height}" '
                  f'viewBox="0 0 {total_width} {total_height}">']

    y_offset = 0
    for svg, (_, h) in zip(chunks, chunk_sizes):
        inner = svg.split("<svg", 1)[1].split(">", 1)[1].rsplit("</svg>", 1)[0]
        svg_output.append(f'<g transform="translate(0,{y_offset})">')
        svg_output.append(inner)
        svg_output.append('</g>')
        y_offset += h + spacing

    svg_output.append("</svg>")
    return "\n".join(svg_output)
