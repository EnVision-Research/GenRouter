# Skill: Spatial Layout & Positioning

## Description
This skill focuses exclusively on **WHERE things are** in the scene — the arrangement, positioning, depth layering, and directional relationships between multiple elements. It does NOT cover what properties objects have (→ attribute_binding), how the scene looks visually (→ aesthetic), physical plausibility (→ physics_material), or how many objects (→ quantity).

- **Trigger when**: The prompt involves **multi-element positioning** — terms like "on the left/right," "above/below," "behind/in front of," "foreground/midground/background," perspective requirements, or when multiple distinct objects need to be arranged in a specific spatial configuration.
- **Do NOT trigger when**: The request is for a single, centered subject with no spatial constraints. Do NOT trigger for attribute assignment (→ attribute_binding), counting (→ quantity), physical interactions (→ physics_material), or visual style (→ aesthetic/creative).

---

## Instructions
When this skill is active, rewrite the user's prompt explicitly covering the following spatial dimensions:

1.  **Object Listing (S1):** Enumerate all distinct objects/subjects that need positioning.

2.  **Absolute Position (S2):** Use frame-relative coordinates for each object.
    - Examples: "in the top-left corner," "centered in the frame," "along the bottom edge."
    - Prefer absolute terms like "LEFT SIDE OF THE FRAME" over relative terms like "to the left of X."

3.  **Relative Position (S3):** When objects relate to each other, specify the relationship explicitly.
    - "Object A is directly behind Object B, partially occluded."
    - "Object A is 2 feet to the right of Object B, at the same height."

4.  **Orientation & Facing (S4):** Specify which direction each subject faces.
    - "facing the camera," "in profile view facing left," "turned 45 degrees away from the viewer."

5.  **Depth Layering (S5):** Explicitly define foreground, midground, and background.
    - "In the foreground (closest to camera): [X]. In the midground: [Y]. In the background: [Z]."
    - Depth creates the illusion of 3D space — always specify it for complex scenes.

6.  **Occlusion & Overlap (S6):** When objects overlap, state which is in front.
    - "Object A partially obscures the lower half of Object B."
    - "The tree trunk blocks the left portion of the building behind it."

7.  **Scale Relationships (S7):** Relative sizes between objects.
    - "The statue towers over the human figure, approximately three times taller."
    - "The miniature model is palm-sized compared to the full-scale building behind it."

8.  **Group Arrangements (S8):** For multiple similar objects, define the formation.
    - "Three figures standing in a row from left to right."
    - "Arranged in a semicircle facing the central object."

Writing Principles:
- **Logical Soundness:** Ensure no contradictions (e.g., avoid "A is left of B, B is left of A").
- **Explicitness:** Replace vague words like "nearby" with "standing 2 feet to the right."
- **Format:** Output the result as a single, fluent, and descriptive paragraph.

## Output Format
Return ONLY the final enhanced prompt text. Do not include any conversational filler, introductory remarks, or prefixes like "Enhanced prompt:".
