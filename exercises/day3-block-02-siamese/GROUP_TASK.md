# Build a similarity decision

30 minutes in pairs; swap who edits.

Implement rowwise distance, contrastive loss, error counts and one embedding-gradient step (TODO 1–4). Compare cutoffs .4 → .7 → 1.0 on the same pairs, first noisy, then clean, and inspect loss before/after the bounded step. Stretch: choose the cutoff with the fewest errors (TODO 5).

Keep your code and computed results. The trainer presents the completed solution afterward. No written prediction or observation fields; trial download is optional. [Inputs and APIs](TASK_API.md).
