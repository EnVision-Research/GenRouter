# Skill: Text Rendering

## Description
This skill should be triggered when the user's request contains **specific quotes, slogans, names, or any intended text** that needs to appear accurately and aesthetically within the generated image.
- **Trigger when**: The user uses quotation marks (e.g., "Welcome"), or phrases like "a sign saying...", "a logo with the word...", "text on the shirt," or "the title is...".
- **Do NOT trigger when**: The request is purely for abstract imagery, textures, or scenes where no specific legible characters are required. Do NOT trigger for object positioning (→ spatial), lighting/camera (→ aesthetic), body correctness (→ anatomy), or conceptual style (→ creative).

---

## Instructions
When this skill is active, rewrite the user's prompt by applying the following six principles to ensure high-fidelity text generation and professional typographic layout:

1. **Spatial Layout & Planning (Positional Clarity)**
- Explicitly define the position of the text using natural language (e.g., "centered on the top third," "aligned to the bottom right," "wrapped around the center").
- Create a clear hierarchy if there are multiple text elements.

2. **Line-Level Decomposition (Structural Breakdown)**
- If the text is long, break it into distinct lines. Do not let the model "guess" the line breaks.
- Describe the structure: "A two-line inscription where the first line reads '...' and the second line reads '...'".

3. **Contextual Carrier & Integration (The Surface)**
- Clearly define what the text is written on (e.g., a neon sign, a weathered wooden plank, a sleek digital screen, or embroidered on a silk fabric).
- Ensure the text interacts with the carrier's texture (e.g., "the letters are deeply carved into the stone," "the neon light casts a pink glow on the surroundings").

4. **Typographic Style & Materiality (Visual Identity)**
- Specify font characteristics: (e.g., "bold sans-serif," "elegant cursive calligraphy," "distressed typewriter font," "3D chrome letters").
- Define colors, lighting, and materials (e.g., "glowing gold letters," "matte black ink," "translucent frosted glass characters").

5. **Explicit Signaling (Syntactic Reinforcement)**
- Use strong, directive verbs such as "rendered," "inscribed," "embossed," "spelled out," or "stenciled."
- Always wrap the target text in double quotation marks to signal its importance to the diffusion model's encoder.

## Output Format
Return ONLY the final enhanced prompt text. Do not include any conversational filler, introductory remarks, or prefixes like "Enhanced prompt:".
