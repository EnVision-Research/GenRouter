# Skill: Anatomy & Body Coherence

## Description
This skill addresses the most common failure in image generation: **malformed human/animal bodies** — extra fingers, fused hands, impossible joints, distorted faces, wrong limb counts, and implausible poses.
- **Trigger when**: The prompt involves human figures, animals, or creatures where body correctness matters — portraits, full-body shots, action poses, group scenes with people, close-ups of hands/faces.
- **Do NOT trigger when**: The scene contains no living subjects, or subjects are too small/distant for anatomy to matter (e.g., tiny silhouettes in a landscape).

---

## Instructions
When this skill is active, rewrite the user's prompt by applying the following principles to maximize anatomical correctness:

1. **Pose Specification (Avoid Ambiguity)**
- Never leave the pose undefined. Specify exactly what each limb is doing.
- Use known, stable poses: "arms crossed over chest," "hands clasped behind back," "hands resting on a table," "one hand on hip."
- For complex poses, describe joint angles: "right arm extended forward at shoulder height, elbow slightly bent, palm open facing viewer."

2. **Hand & Finger Control**
- Hands are the #1 failure point. Always specify hand state explicitly.
- Safe hand poses: "hands in pockets," "holding [object]," "fists clenched," "fingers interlaced," "palms flat on surface."
- If hands must be visible and open, specify: "five distinct fingers on each hand, naturally spread."
- Avoid prompts that force complex finger arrangements unless necessary.

3. **Face & Expression Coherence**
- Specify eye direction: "looking directly at camera," "gazing to the left," "eyes downcast."
- Specify expression clearly: "slight smile with closed lips," not just "happy."
- For profiles, specify which side: "left profile view showing left ear."
- Avoid ambiguous multi-face scenarios without clear spatial separation.

4. **Body Proportion Anchoring**
- Reference real proportions: "adult human proportions," "the figure's head is approximately 1/7 of total height."
- For stylized figures, explicitly state the style: "chibi proportions with oversized head" or "heroic proportions with elongated limbs."
- Specify age/build when relevant: "athletic build," "elderly posture with slight stoop."

5. **Strategic Framing to Manage Difficulty**
- If anatomical perfection isn't critical, use compositions that minimize risk: medium shots (waist-up), subjects partially behind objects, hands occupied with props.
- For full-body shots requiring hand detail, increase emphasis: "anatomically correct hands with clearly defined fingers."

6. **Animal & Creature Anatomy**
- Specify limb count: "a horse with four legs, each hoof clearly touching the ground."
- For fantasy creatures, ground in real anatomy: "a dragon with bat-like wing membrane structure" rather than leaving wing anatomy undefined.

## Output Format
Return ONLY the final enhanced prompt text. Do not include any conversational filler, introductory remarks, or prefixes like "Enhanced prompt:".
