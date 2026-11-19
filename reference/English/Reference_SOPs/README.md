# Reference SOPs (answer key for KPIs A3, A4, A5)

These are the "correct SOPs" the spec asks for in KPI A1. They are the answer key that the
generated SOPs are scored against:

- A3 Steps found: share of reference steps that appear in the generated SOP (target 80%+)
- A4 Step order: share of found steps that are in the same order as here (target 90%+)
- A5 Tools and safety: share of reference tools and safety points that appear (target 70%+)

## How they were made

Written from the ground-truth transcripts in `reference/English/Ground_Truth/`, without looking at
the generated SOPs, so the key stays independent of the system it scores. Then checked against each
video. Each video's scores are in `<video>_score.md` next to its reference.

## What counts as a step

- An action a worker does, or a rule they must follow while doing the task.
- Not counted: introductions, sponsor reads (Jackery, Titebond adverts), jokes, product reviews,
  "subscribe" endings, and the presenter's opinions.
- Optional tips are listed under "Tips" and are not scored as steps.
- eng_05 has no worker, so its key lists process stages instead of worker actions.

Times in `[mm:ss]` are the transcript section the step comes from (eng_02 only, the other ground
truths have no times).
