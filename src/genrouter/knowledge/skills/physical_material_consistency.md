# Skill: Physical Plausibility & Material Interaction

## Description
This skill addresses **physically impossible scenes** — objects floating without support, incorrect reflections/shadows, wrong relative scales, impossible material interactions (fire underwater, dry objects in rain), and structural errors in real-world objects (wrong number of wheels, backwards clock hands).
- **Trigger when**: The prompt involves physical interactions between objects (gravity, reflections, shadows, liquid, fire, glass refraction), specific material properties that affect appearance (transparency, reflectivity, roughness), or real-world objects where structural accuracy matters.
- **Do NOT trigger when**: The scene is abstract art where physics doesn't apply, or the prompt is intentionally surreal/impossible (use creative skill instead).

---

## Instructions
When this skill is active, rewrite the user's prompt by applying the following principles to ensure physical plausibility:

1. **Gravity, Support & Contact**
- Every object must be grounded. Specify what supports it.
- WRONG: "a book and a shelf" (book may float)
- RIGHT: "a book resting flat on the middle shelf, its spine visible and pages fanning slightly under gravity"
- For hanging/suspended objects, specify the attachment: "a lantern hanging from a wrought-iron hook bolted to the beam."

2. **Shadow & Light Consistency**
- Shadows must match the light source direction and intensity.
- "A single warm spotlight from the upper left casts a long, sharp shadow extending to the lower right on the concrete floor."
- If multiple light sources, describe each shadow: "the main shadow from the overhead light falls directly below, while a softer secondary shadow from the window light extends to the right."

3. **Reflection & Refraction**
- For reflective surfaces (mirrors, water, glass, chrome), describe what is reflected.
- "The polished marble floor reflects a slightly darkened, inverted image of the chandelier above."
- For glass/water refraction: "the straw appears bent at the water surface due to refraction."

4. **Material Interaction Physics**
- Describe how materials interact with each other and with light.
- "Rain droplets bead on the waxed surface of the leather jacket, not soaking in."
- "The hot coffee mug produces a ring of condensation on the cold glass table."
- "The candle flame creates a warm orange glow that illuminates the surrounding wax and casts flickering light on the wall."

5. **Scale & Proportion Anchoring**
- Use known real-world objects as scale references.
- "A housecat sitting beside a standard dining chair, its head reaching about knee height."
- For unfamiliar objects, give explicit dimensions: "a 6-foot-tall grandfather clock" or "a coin-sized ladybug."

6. **Real-World Object Structural Accuracy**
- When depicting specific real objects, name their correct structural features.
- "A standard acoustic guitar with six strings, a round sound hole, and wooden frets along the neck."
- "A bicycle with two spoked wheels, drop handlebars, a chain connecting the pedals to the rear wheel sprocket."
- "An analog clock showing 3:15, with the short hour hand pointing to 3 and the long minute hand pointing to 3 (the 15-minute mark)."

## Output Format
Return ONLY the final enhanced prompt text. Do not include any conversational filler, introductory remarks, or prefixes like "Enhanced prompt:".
