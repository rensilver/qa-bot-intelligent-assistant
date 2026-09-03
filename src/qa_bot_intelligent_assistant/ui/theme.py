import gradio as gr

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill="#10161B",
    body_background_fill_dark="#10161B",
    body_text_color="#E9EDEE",
    body_text_color_dark="#E9EDEE",
    body_text_color_subdued="#8CA0AC",
    body_text_color_subdued_dark="#8CA0AC",
    background_fill_primary="#1A2229",
    background_fill_primary_dark="#1A2229",
    background_fill_secondary="#161D22",
    background_fill_secondary_dark="#161D22",
    block_background_fill="#1A2229",
    block_background_fill_dark="#1A2229",
    block_border_color="#2A343B",
    block_border_color_dark="#2A343B",
    block_label_background_fill="#1A2229",
    block_label_background_fill_dark="#1A2229",
    block_label_text_color="#8CA0AC",
    block_label_text_color_dark="#8CA0AC",
    block_title_text_color="#E9EDEE",
    block_title_text_color_dark="#E9EDEE",
    border_color_primary="#2A343B",
    border_color_primary_dark="#2A343B",
    input_background_fill="#161D22",
    input_background_fill_dark="#161D22",
    input_border_color="#2A343B",
    input_border_color_dark="#2A343B",
    color_accent_soft="#2A2419",
    color_accent_soft_dark="#2A2419",
    button_primary_background_fill="#E8A33D",
    button_primary_background_fill_dark="#E8A33D",
    button_primary_background_fill_hover="#F0B25A",
    button_primary_background_fill_hover_dark="#F0B25A",
    button_primary_text_color="#1A1206",
    button_primary_text_color_dark="#1A1206",
    button_secondary_background_fill="#232D34",
    button_secondary_background_fill_dark="#232D34",
    button_secondary_text_color="#E9EDEE",
    button_secondary_text_color_dark="#E9EDEE",
)

CSS = """
#app-shell { max-width: 1080px; margin: 0 auto !important; padding: 32px 20px 48px; }

.header-block { text-align: center; margin-bottom: 12px; }
.header-block .eyebrow {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase;
    color: #E8A33D; margin: 0 0 6px;
}
.header-block .title-row {
    display: flex; align-items: center; justify-content: center; gap: 10px;
}
.header-block h1 { font-size: 28px; font-weight: 600; margin: 0; color: #E9EDEE; }
.header-block .subtitle {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 12px; color: #8CA0AC; margin: 6px 0 0;
}
.status-dot {
    width: 8px; height: 8px; border-radius: 50%; background: #E8A33D; flex-shrink: 0;
    box-shadow: 0 0 0 0 rgba(232, 163, 61, 0.45);
    animation: status-pulse 2.4s infinite;
}
@keyframes status-pulse {
    0% { box-shadow: 0 0 0 0 rgba(232, 163, 61, 0.45); }
    70% { box-shadow: 0 0 0 8px rgba(232, 163, 61, 0); }
    100% { box-shadow: 0 0 0 0 rgba(232, 163, 61, 0); }
}
@media (prefers-reduced-motion: reduce) {
    .status-dot { animation: none; }
}

.panel { border: 1px solid #2A343B; border-radius: 12px; padding: 18px !important; }
.panel-title.prose h3, .panel-title h3 {
    font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
    color: #8CA0AC; border-left: 3px solid #E8A33D; padding-left: 8px; margin: 4px 0 14px;
}
"""
