# MobileLLM Model Lab — UI/UX Design System & Architectural Specification

**Document Version:** 1.0.0  
**Target Platform:** Android 14+ (API 34/35), Jetpack Compose, Material 3  
**Design Persona:** Principal AI Systems Designer & Senior Android Product Engineer  
**Aesthetic Theme:** *Obsidian Cyber-Lime Studio* (Inspired by dark-mode developer telemetry dashboards and mobile AI playgrounds)  
**Hardware Reference:** OnePlus 12R (SM8650, Adreno 740/750 GPU, 120Hz Fluid AMOLED Display)  

---

## 1. Visual Direction & Design Philosophy

The design of **MobileLLM Model Lab** directly translates the sleek, technical, dark-first visual language of the reference design into a production-grade inference laboratory:

1. **Obsidian Matte Surfaces:** Deep obsidian and charcoal background (`#0D1110` / `#121615`) rather than harsh pure black, providing depth, visual comfort during prolonged benchmarking, and OLED battery optimization.
2. **Electric Cyber-Lime Accents (`#A3E635` / `#84CC16`):** Vibrant, high-visibility green highlights active engine states, live telemetry indicators, selected backend pills, and primary action triggers.
3. **Restrained Technical Hierarchy:** Information is structured into distinct, rounded cards (16dp–20dp corner radius) with 1dp subtle hairline borders (`#232B28`). No clutter or decorative gradients.
4. **Data-Dense Telemetry Components:** Borrowing the circular arc gauges, glowing vertical progress meters, and segmented pill toggles from modern telemetry interfaces, presenting live inference metrics (`tok/s`, `TTFT`, `RSS MB`) with zero UI latency.

---

## 2. Design System Tokens & Color Palette

### 2.1 Color Tokens

```kotlin
// Compose Color Palette
object ModelLabColors {
    // Backgrounds & Surfaces
    val Background = Color(0xFF0D1110)       // Deep Obsidian Canvas
    val SurfaceDark = Color(0xFF131816)       // Primary Elevated Surface
    val SurfaceCard = Color(0xFF1A211E)       // Interactive Card Surface
    val SurfaceCardHighlight = Color(0xFF222C28) // Hover / Pressed / Focused Surface
    val CardBorder = Color(0xFF28342F)        // 1dp Hairline Outline

    // Primary Accents (Electric Cyber-Lime)
    val PrimaryLime = Color(0xFFA3E635)       // Electric Lime (Active buttons, gauges)
    val PrimaryLimeDark = Color(0xFF65A30D)   // Muted Lime for inactive tracks
    val LimeGlow = Color(0x33A3E635)          // 20% Alpha Glow for active states
    val LimeTextOnPrimary = Color(0xFF0D1110) // High-contrast text on Lime

    // Secondary & Semantic Accents
    val AccentBlue = Color(0xFF38BDF8)        // Cold Blue (Prefill / TTFT indicators)
    val AccentPurple = Color(0xFFA855F7)      // Purple (Vulkan GPU specific badges)
    val AccentAmber = Color(0xFFFBBF24)       // Warm Amber (Warmup / Warnings / Throttling)
    val AccentRed = Color(0xFFF87171)         // Crimson Red (OOM / Divergence / Errors)
    val AccentGreenSuccess = Color(0xFF4ADE80)// Mint Green (Bit-exact parity pass)

    // Typography & Content
    val TextPrimary = Color(0xFFF1F5F3)       // Clean High-Contrast White
    val TextSecondary = Color(0xFF94A39D)     // Muted Slate-Green for labels
    val TextTertiary = Color(0xFF64746E)      // Monospace stats / units
    val Divider = Color(0xFF1F2825)
}
```

### 2.2 Typography Hierarchy

- **Hero / Metrics Headings:** JetBrains Mono or Roboto Flex Tabular (`24sp` - `32sp`, Bold). Numbers never jump or jitter during live streaming inference.
- **Section Headers:** Sans-serif (`14sp` - `16sp`, SemiBold, Letter-spacing `0.5sp`, Uppercase subtitle styling).
- **Body / Chat Prompts:** Inter or Roboto (`14sp` - `15sp`, Normal, line-height `22sp`).
- **Telemetry Readouts:** Strict Tabular Monospace (`11sp` - `12sp`, Medium, `TextSecondary`).

---

## 3. Core App Navigation Architecture

The app uses a 4-tab bottom navigation bar with a floating active telemetry pill and top status bar:

```
+-------------------------------------------------------------+
|  MobileLLM Lab  ● READY                OnePlus 12R · Vulkan  ⚙  |
+-------------------------------------------------------------+
|                                                             |
|                    ACTIVE SCREEN CONTENT                     |
|                                                             |
+-------------------------------------------------------------+
|    [ ⚡ Lab ]    [ 📁 Models ]    [ 📊 Bench ]    [ 📜 History ]   |
+-------------------------------------------------------------+
```

---

## 4. Screen-by-Screen UI Specifications

### 4.1 Screen 1 — The Lab (Main Inference Playground)

```
+-------------------------------------------------------------+
|  ACTIVE MODEL                                              |
|  Qwen 2.5 1.5B Instruct               [ GGUF · Q4_K_M ]     |
|  28 Layers · 1.12 GB · 2048 Ctx       Status: LOADED ●      |
|  [ Switch Model ]                       [ Unload Engine ]   |
+-------------------------------------------------------------+
|  INFERENCE BACKEND                                          |
|  (●) CPU (ARM NEON)            ( ) GPU (Qualcomm Vulkan)     |
+-------------------------------------------------------------+
|  LIVE TELEMETRY                                      LIVE ● |
|  +--------------------+  +--------------------+             |
|  |   152.6 tok/s      |  |    104.2 ms        |             |
|  |   Decode Speed     |  |    Prefill TTFT    |             |
|  +--------------------+  +--------------------+             |
|  |   32 tok Gen       |  |    489 MB RSS      |             |
|  |   Completed        |  |    Native Memory   |             |
|  +--------------------+  +--------------------+             |
|  [▼ Expand Full Telemetry: KV-Cache, Thermals, Graph Split] |
+-------------------------------------------------------------+
|  CONVERSATION CONSOLE                                       |
|  User: Explain quantum computing in simple terms.           |
|                                                             |
|  Assistant: Quantum computing uses qubits to process        |
|  superposition states simultaneously...                     |
|  [ 32 tokens · 209 ms · Bit-Exact ]                         |
+-------------------------------------------------------------+
|  [ Type prompt or select benchmark vector...      ] [ STOP/RUN ] |
+-------------------------------------------------------------+
```

#### Key UI Elements:
- **Active Model Hero Card:** Features a dark matte container, green status pill (`LOADED ●`), parameter counts, and quick-action buttons.
- **Segmented Backend Switcher:** Smooth animated pill switch between `CPU (ARM NEON)` and `GPU (Vulkan)`. Unsupported backends are disabled with an explicit tooltip.
- **Compact Live Telemetry Card:** High-contrast 2x2 grid displaying pure decode tokens/second, decomposed TTFT, generated count, and native process RSS.
- **Expandable Telemetry Drawer:** Tapping expands detailed metrics: Prompt eval rate, memory delta, KV-cache occupancy bar, Qualcomm Adreno wavefront telemetry, and thermal status.
- **Streaming Response Bubble:** Streaming text output with auto-scrolling, elapsed generation timer, and copy/share buttons.

---

### 4.2 Screen 2 — Models Manager

```
+-------------------------------------------------------------+
|  LOCAL GGUF MODELS (2)                    [ + Import GGUF ]  |
+-------------------------------------------------------------+
|  [ All ]  [ Favorites ★ ]  [ Q4_K_M ]  [ Small (<1GB) ]      |
+-------------------------------------------------------------+
|  +-------------------------------------------------------+  |
|  | ★ Qwen 2.5 1.5B Instruct                [ ACTIVE ]    |  |
|  |   1.12 GB · Q4_K_M · 28 Layers · 2048 Ctx · GGUF v3   |  |
|  |   Path: /sdcard/Download/qwen2.5-1.5b-instruct-...    |  |
|  |   [ Unload ]   [ Benchmark ]   [ Details ]   [ 🗑 Delete ] |  |
|  +-------------------------------------------------------+  |
|  +-------------------------------------------------------+  |
|  |   SmolLM2 135M Instruct                 [ READY ]     |  |
|  |   105 MB · Q4_K_M · 30 Layers · 2048 Ctx · GGUF v3    |  |
|  |   Path: /sdcard/Android/data/.../SmolLM2-135M-...     |  |
|  |   [ Load to Engine ]      [ Benchmark ]      [ Details ]|  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
```

#### Key UI Elements:
- **Storage Access Framework (SAF) Integration:** Single-tap `+ Import GGUF` triggers the native system document picker.
- **GGUF Auto-Metadata Inspector:** Instant parsing of architecture name, quantization type, tensor count, and context limits directly from file headers.
- **Lifecycle Transition Indicator:** Progressive loading progress bar with granular states: `Reading Header -> Allocating Buffers -> Loading Tensors (XX%) -> Initializing Backend -> Ready`.

---

### 4.3 Screen 3 — Benchmark Mode (A/B & Performance Matrix)

```
+-------------------------------------------------------------+
|  AUTOMATED BENCHMARK RUNNER                                 |
+-------------------------------------------------------------+
|  Selected Model:  [ SmolLM2 135M Instruct ▼ ]                |
|  Target Backend:  [ All (CPU vs Vulkan Comparison) ▼ ]       |
|  Preset Vector:   [ Short Prompt (16 tok / 32 gen) ▼ ]       |
|  Iterations:      [ 5 Runs (1 Warmup + 4 Measured) ▼ ]       |
+-------------------------------------------------------------+
|  [ ▶ START BENCHMARK SUITE ]                                |
+-------------------------------------------------------------+
|  RESULTS MATRIX                                             |
|  ---------------------------------------------------------  |
|  Backend    TTFT (Median)    Decode (Median)    Peak RAM    |
|  ---------------------------------------------------------  |
|  CPU NEON     136.1 ms         151.7 tok/s       497 MB     |
|  Vulkan GPU   104.2 ms         154.2 tok/s       512 MB     |
|  Speedup:      +23.4%            +1.6%           +3.0%      |
|  ---------------------------------------------------------  |
|  [ 📈 View Speedup Bar Charts ]       [ 📥 Export CSV/JSON ] |
+-------------------------------------------------------------+
```

#### Key UI Elements:
- **Warmup Isolation:** Visually separates warmup iteration metrics from measured statistical aggregates (median, mean, min, max, standard deviation $\sigma$).
- **Thermal Throttling Protection:** Displays hardware temperature and thermal throttling alerts if SoC thermal state rises during sustained runs.

---

### 4.4 Screen 4 — CPU vs. GPU Correctness & Parity Lab

```
+-------------------------------------------------------------+
|  NUMERICAL & TOKEN PARITY VALIDATOR                         |
|  Validating Qwen 2.5 1.5B (L=0 CPU vs L=28 Vulkan GPU)      |
+-------------------------------------------------------------+
|  PARITY STATUS:  [ ✔ 100% BIT-EXACT MATCH ]                 |
+-------------------------------------------------------------+
|  CPU Reference:                                             |
|  "Quantum computing is a type of computing that uses..."    |
|                                                             |
|  Vulkan GPU:                                                |
|  "Quantum computing is a type of computing that uses..."    |
+-------------------------------------------------------------+
|  Divergence Telemetry:                                      |
|  - Compared Tokens: 32 / 32                                 |
|  - First Divergent Token: None                              |
|  - Logit L2 Relative Error: < 0.001%                        |
|  - Subgroup Size: 64 (Adreno Wavefront Verified)            |
+-------------------------------------------------------------+
```

---

## 5. Implementation Artifacts Created

To allow immediate visual and functional exploration of this design, an interactive Generative UI prototype has been constructed as an artifact in:
`model_lab_design_preview.html`

This interactive design artifact features:
1. Complete Obsidian & Cyber-Lime styling with live theme synchronization.
2. Functional tab navigation switching between **Lab Playground**, **Model Management**, **Benchmarks**, and **Correctness Analysis**.
3. Live interactive controls: Backend toggle, prompt execution simulation, live telemetry cards, and comparative bar graphs.
