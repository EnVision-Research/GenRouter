# Skill: Attribute Binding & Multi-Object Property Assignment

## Description
This skill addresses **attribute leakage/bleeding** — when multiple objects each have different properties (color, size, texture, material), models frequently swap or mix them. "A red car next to a blue house" may produce a blue car and a red house.
- **Trigger when**: The prompt has 2+ objects that must each have distinct, specific attributes (different colors, different sizes, different materials, different styles).
- **Do NOT trigger when**: All objects share the same attributes, or there is only one object with attributes, or attribute precision is not important.

---

## Instructions
When this skill is active, rewrite the user's prompt by applying the following principles to ensure correct attribute-to-object binding:

1. **Self-Contained Subject Clauses**
- Describe each subject in its own complete clause with all attributes attached.
- WRONG: "a red and blue car and house"
- RIGHT: "a glossy red sports car parked beside a two-story blue house with white trim"
- Each object's description should be a complete, unambiguous phrase.

2. **Spatial Separation Reinforces Attribute Separation**
- Place differently-attributed objects in clearly different locations.
- "On the LEFT side of the frame, a golden retriever with fluffy cream fur. On the RIGHT side, a black poodle with tight curly fur."
- Spatial distance makes the model less likely to blend attributes.

3. **Avoid Floating Adjective Lists**
- WRONG: "big red small blue spheres and cubes"
- RIGHT: "a large red sphere on the left, a small blue cube on the right"
- Every adjective must be structurally bound to exactly one noun.

4. **Material & Texture Binding**
- When objects have different materials, describe each material with its subject.
- "A wooden chair with visible grain texture beside a sleek chrome metal table with reflective surface."
- Do not say "wooden and chrome furniture" — this invites mixing.

5. **Size & Scale Binding**
- When objects differ in size, use relative or absolute measurements tied to each.
- "A miniature toy soldier (3 inches tall) standing next to a full-size basketball (regulation size)."
- Avoid vague comparisons like "the big one and the small one."

6. **Color Conflict Resolution**
- If two objects are similar but different colors, add extra distinguishing details beyond just color.
- Instead of "a red cup and a blue cup," write "a tall, narrow red ceramic mug with a thin handle, beside a short, wide blue porcelain teacup with a saucer."

## Output Format
Return ONLY the final enhanced prompt text. Do not include any conversational filler, introductory remarks, or prefixes like "Enhanced prompt:".
