# Skill: Quantity & Counting Control

## Description
This skill addresses the well-documented failure of image generation models to produce the **exact specified number of objects**. Models frequently generate wrong counts — ask for "three apples" and get 2 or 5.
- **Trigger when**: The prompt specifies an exact count of objects (e.g., "three cats," "a pair of shoes," "five candles"), or requires multiple instances of the same type that must be individually distinguishable.
- **Do NOT trigger when**: The quantity is vague ("some," "several," "a crowd") and exactness doesn't matter, or there is only a single instance of each object.

---

## Instructions
When this skill is active, rewrite the user's prompt by applying the following principles to ensure correct object counts:

1. **Enumerate Individually Instead of Using Numbers**
- WRONG: "three red apples on a table"
- RIGHT: "a red apple on the left side of the table, a second red apple in the center, and a third red apple on the right side"
- Each instance should have its own spatial anchor to make the count unambiguous.

2. **Bind Each Instance to a Unique Position**
- Every counted object must have a distinct location: "the first bird perched on the left branch, the second bird perched on the right branch."
- Use spatial grid language: "top-left," "center," "bottom-right" to pin each instance.

3. **Use Geometric Arrangements for Groups**
- For 3 objects: "arranged in a triangle" or "in a row from left to right."
- For 4 objects: "arranged in a 2×2 grid" or "at the four corners."
- For 5+: "in a straight horizontal line" or "in a semicircle."
- The arrangement gives the model a structural scaffold that enforces the count.

4. **Differentiate Instances When Possible**
- Even if objects are the same type, add slight variations: "the leftmost candle is slightly taller, the middle one is medium height, the rightmost is shortest."
- Variations help the model treat each as a distinct entity rather than a repeated texture.

5. **Keep Counts Low and Manageable**
- Models are most reliable with 1-5 objects. For 6+, group them: "two rows of three candles each."
- For large quantities ("a dozen eggs"), describe the container/arrangement rather than individual items: "a full egg carton with all 12 slots occupied."

6. **State the Count Redundantly**
- Mention the number both at the start and end: "Exactly three birds — one on each branch of the tree, for a total of three birds."
- Redundancy reinforces the count signal.

## Output Format
Return ONLY the final enhanced prompt text. Do not include any conversational filler, introductory remarks, or prefixes like "Enhanced prompt:".
