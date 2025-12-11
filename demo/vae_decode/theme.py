"""
Tangled Cables Theme - Sage Green DAW Aesthetic

Inspired by the logo: two retro computers connected via tangled wires,
sharing a heart. Muted army/sage green palette with matrix grid pattern.
"""

# =============================================================================
# TANGLED CABLES THEME
# =============================================================================

BRUTALIST_CSS = """
/* =================================================================
   DESIGN TOKENS - Sage/Army Green Palette
   ================================================================= */
:root {
    /* Base - Deep forest black-greens */
    --tc-black: #080a08;
    --tc-dark: #0c0e0c;
    --tc-surface: #121612;
    --tc-surface-raised: #181c18;
    --tc-border: #1e261e;
    --tc-border-bright: #2a362a;

    /* Accent - Muted sage/army green (from logo) */
    --tc-accent: #6b8b5a;
    --tc-accent-bright: #8aaa78;
    --tc-accent-dim: #4a6a3a;
    --tc-accent-glow: rgba(107, 139, 90, 0.3);

    /* Wire colors (for decorative elements) */
    --tc-wire: #5a6b52;
    --tc-wire-light: #7a8b72;

    /* Text */
    --tc-text: #b8c4b0;
    --tc-text-dim: #6a7a62;
    --tc-text-muted: #3a4a32;

    /* Status */
    --tc-ok: #8aaa78;
    --tc-warn: #aa8a58;
    --tc-error: #aa5858;

    /* Typography */
    --tc-mono: 'IBM Plex Mono', 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;

    /* Spacing - tighter */
    --tc-gap-xs: 2px;
    --tc-gap-sm: 4px;
    --tc-gap-md: 8px;
    --tc-gap-lg: 12px;
}

/* =================================================================
   GLOBAL + MATRIX BACKGROUND
   ================================================================= */
.gradio-container {
    font-family: var(--tc-mono) !important;
    background: var(--tc-black) !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
    /* Matrix grid pattern */
    background-image:
        linear-gradient(var(--tc-border) 1px, transparent 1px),
        linear-gradient(90deg, var(--tc-border) 1px, transparent 1px) !important;
    background-size: 20px 20px !important;
}

* {
    border-radius: 0 !important;
}

/* Hide default Gradio footer */
footer { display: none !important; }

/* =================================================================
   TYPOGRAPHY - Compact & Technical
   ================================================================= */
h1, h2, h3, .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
    font-family: var(--tc-mono) !important;
    color: var(--tc-text) !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    margin: 0 0 var(--tc-gap-sm) 0 !important;
}

.gr-markdown h1 { font-size: 14px !important; color: var(--tc-accent) !important; }
.gr-markdown h2 { font-size: 11px !important; }
.gr-markdown h3 { font-size: 10px !important; color: var(--tc-wire-light) !important; }

label, .gr-label, .label-wrap, .label-wrap span {
    font-family: var(--tc-mono) !important;
    font-size: 9px !important;
    color: var(--tc-text-dim) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}

p, .gr-markdown p, .prose {
    color: var(--tc-text-dim) !important;
    font-size: 10px !important;
    line-height: 1.4 !important;
}

/* =================================================================
   BLOCKS & PANELS - Tight Spacing
   ================================================================= */
.gr-block, .gr-box, .gr-group, .gr-form {
    background: var(--tc-surface) !important;
    border: 1px solid var(--tc-border) !important;
    padding: var(--tc-gap-md) !important;
    margin-bottom: var(--tc-gap-sm) !important;
}

.gr-panel {
    background: var(--tc-dark) !important;
}

.gr-padded {
    padding: var(--tc-gap-md) !important;
}

/* Compact row spacing */
.gr-row {
    gap: var(--tc-gap-sm) !important;
}

.gr-column {
    gap: var(--tc-gap-sm) !important;
}

/* =================================================================
   INPUTS - DAW Style
   ================================================================= */
input, select, textarea, .gr-input {
    background: var(--tc-black) !important;
    border: 1px solid var(--tc-border) !important;
    color: var(--tc-text) !important;
    font-family: var(--tc-mono) !important;
    font-size: 11px !important;
    padding: 4px 8px !important;
}

input:focus, select:focus {
    border-color: var(--tc-accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 1px var(--tc-accent-glow) !important;
}

/* =================================================================
   SLIDERS - Fader Style
   ================================================================= */
input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    background: transparent !important;
    height: 24px !important;
    margin: 4px 0 !important;
}

input[type="range"]::-webkit-slider-track {
    background: linear-gradient(
        to right,
        var(--tc-accent-dim) 0%,
        var(--tc-border) 50%,
        var(--tc-accent-dim) 100%
    ) !important;
    height: 3px !important;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    background: var(--tc-accent) !important;
    height: 16px !important;
    width: 6px !important;
    margin-top: -7px !important;
    cursor: grab !important;
    border: 1px solid var(--tc-accent-bright) !important;
}

input[type="range"]::-webkit-slider-thumb:hover {
    background: var(--tc-accent-bright) !important;
}

input[type="range"]::-webkit-slider-thumb:active {
    cursor: grabbing !important;
    background: var(--tc-accent-bright) !important;
}

/* Number display next to slider */
.gr-number input {
    width: 50px !important;
    text-align: right !important;
}

/* =================================================================
   BUTTONS - Patch Bay Style
   ================================================================= */
.gr-button, button {
    font-family: var(--tc-mono) !important;
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 6px 12px !important;
    border: 1px solid var(--tc-border) !important;
    background: var(--tc-surface) !important;
    color: var(--tc-text) !important;
    transition: all 0.1s !important;
    min-height: 28px !important;
}

.gr-button:hover, button:hover {
    background: var(--tc-surface-raised) !important;
    border-color: var(--tc-wire) !important;
}

.gr-button-primary, button.primary, button[variant="primary"] {
    background: var(--tc-accent-dim) !important;
    color: var(--tc-text) !important;
    border-color: var(--tc-accent) !important;
    font-weight: 600 !important;
}

.gr-button-primary:hover, button.primary:hover {
    background: var(--tc-accent) !important;
    color: var(--tc-black) !important;
}

/* Transport buttons - bigger touch targets */
.transport-btn {
    min-width: 80px !important;
    padding: 8px 12px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}

.transport-active {
    background: var(--tc-accent) !important;
    color: var(--tc-black) !important;
    border-color: var(--tc-accent-bright) !important;
}

/* =================================================================
   RADIO BUTTONS - Channel Selector Style (HIGH VISIBILITY)
   ================================================================= */
.gr-radio {
    gap: var(--tc-gap-xs) !important;
}

.gr-radio label,
.gr-radio .wrap label,
label.svelte-1qxcj04 {
    background: var(--tc-surface) !important;
    border: 1px solid var(--tc-border) !important;
    padding: 6px 12px !important;
    margin: 0 !important;
    cursor: pointer !important;
    font-size: 10px !important;
    transition: all 0.15s !important;
    color: var(--tc-text-dim) !important;
}

.gr-radio label:hover,
label.svelte-1qxcj04:hover {
    border-color: var(--tc-wire-light) !important;
    color: var(--tc-text) !important;
}

/* SELECTED STATE - Very visible (multiple selectors for Gradio) */
.gr-radio input:checked + label,
.gr-radio label.selected,
input:checked + label.svelte-1qxcj04,
label.svelte-1qxcj04.selected,
.gr-radio .wrap input:checked ~ span,
.gr-radio input[type="radio"]:checked + span,
[data-testid="radio"] input:checked + label,
.radio-group input:checked + label,
span.svelte-1qxcj04:has(input:checked),
label:has(input[type="radio"]:checked) {
    background: var(--tc-accent) !important;
    border-color: var(--tc-accent-bright) !important;
    color: var(--tc-black) !important;
    font-weight: 700 !important;
    box-shadow: 0 0 16px var(--tc-accent-glow), inset 0 0 20px rgba(0,0,0,0.3) !important;
}

/* Also style the wrapper when checked */
.gr-radio .wrap:has(input:checked),
.gr-radio > div:has(input:checked) {
    background: var(--tc-accent) !important;
    color: var(--tc-black) !important;
}

/* =================================================================
   DROPDOWN - Patch Select
   ================================================================= */
.gr-dropdown {
    background: var(--tc-black) !important;
}

.gr-dropdown .wrap {
    background: var(--tc-black) !important;
    border-color: var(--tc-border) !important;
    min-height: 28px !important;
}

.gr-dropdown ul {
    background: var(--tc-surface) !important;
    border-color: var(--tc-border) !important;
}

.gr-dropdown li {
    color: var(--tc-text) !important;
    font-size: 10px !important;
    padding: 4px 8px !important;
}

.gr-dropdown li:hover {
    background: var(--tc-accent-glow) !important;
}

/* =================================================================
   AUDIO PLAYER - Minimal
   ================================================================= */
.gr-audio {
    background: var(--tc-black) !important;
    border: 1px solid var(--tc-border) !important;
    padding: 4px !important;
}

.gr-audio audio {
    height: 36px !important;
}

/* =================================================================
   TABS - Channel Strip Style
   ================================================================= */
.gr-tabs .tab-nav {
    background: var(--tc-dark) !important;
    border-bottom: 1px solid var(--tc-border) !important;
    gap: 0 !important;
    padding: 0 !important;
}

.gr-tabs .tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 6px 12px !important;
    color: var(--tc-text-dim) !important;
    font-size: 9px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

.gr-tabs .tab-nav button:hover {
    color: var(--tc-text) !important;
}

.gr-tabs .tab-nav button.selected {
    color: var(--tc-accent) !important;
    border-bottom-color: var(--tc-accent) !important;
}

/* =================================================================
   DATA READOUT - LED Display Style
   ================================================================= */
.data-value {
    font-family: var(--tc-mono) !important;
    font-size: 12px !important;
    color: var(--tc-accent) !important;
    background: var(--tc-black) !important;
    padding: 3px 6px !important;
    border: 1px solid var(--tc-border) !important;
    text-shadow: 0 0 4px var(--tc-accent-glow) !important;
}

.data-label {
    font-size: 8px !important;
    color: var(--tc-text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
}

/* =================================================================
   STATUS INDICATORS - LED Style
   ================================================================= */
.status-ok { color: var(--tc-ok) !important; text-shadow: 0 0 4px var(--tc-accent-glow) !important; }
.status-warn { color: var(--tc-warn) !important; }
.status-dim { color: var(--tc-text-dim) !important; }

/* =================================================================
   SECTION HEADERS - Wire Labels
   ================================================================= */
.section-header {
    font-size: 9px !important;
    color: var(--tc-wire-light) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.15em !important;
    padding: 4px 0 2px 0 !important;
    border-bottom: 1px solid var(--tc-border) !important;
    margin-bottom: 6px !important;
}

/* =================================================================
   STEP INDICATORS - Flow Guide
   ================================================================= */
.step-number {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 16px !important;
    height: 16px !important;
    background: var(--tc-accent-dim) !important;
    color: var(--tc-text) !important;
    font-size: 9px !important;
    font-weight: 600 !important;
    margin-right: 6px !important;
}

.step-active .step-number {
    background: var(--tc-accent) !important;
    color: var(--tc-black) !important;
    box-shadow: 0 0 8px var(--tc-accent-glow) !important;
}

/* =================================================================
   PLOTS - Match Theme
   ================================================================= */
.gr-plot {
    background: var(--tc-black) !important;
    border: 1px solid var(--tc-border) !important;
}

/* =================================================================
   COMPACT OVERRIDES
   ================================================================= */
/* Remove excess padding from Gradio defaults */
.contain {
    gap: var(--tc-gap-sm) !important;
}

.gr-form > div {
    margin-bottom: var(--tc-gap-xs) !important;
}

/* Tighter markdown spacing */
.gr-markdown {
    margin: 0 !important;
    padding: 0 !important;
}

.gr-markdown * {
    margin-top: 0 !important;
    margin-bottom: var(--tc-gap-xs) !important;
}

/* Hide excess labels on some components */
.hide-label .label-wrap {
    display: none !important;
}
"""
