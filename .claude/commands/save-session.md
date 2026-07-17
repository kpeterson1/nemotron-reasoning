# Prompt & output auto-save
If the user's prompt begins with a session tag in brackets, automatically
save session artifacts when the task completes.

Tag format: [{phase}/{category}/{topic}/{version}]
Example: [phase3/investigate/bit-manip-learning-gap/v01]

Categories: eval, train, run, investigate, test

At task completion, after writing the summary (after ---):
1. Get today's date as YYYY-MM-DD.
2. Save the user's original prompt (everything after the tag) to:
   prompts/{phase}/{date}_{category}_{topic}_{version}_prompt.md
3. Save the summary (everything after the final --- delimiter) to:
   logs/interactive/{date}_{category}_{topic}_{version}_cli.log
4. Create directories as needed. Preserve text verbatim.
5. Briefly confirm both saves at the end.

Example for tag [phase3/investigate/bit-manip-learning-gap/v01] on 2026-06-09:
  prompts/phase3/2026-06-09_investigate_bit-manip-learning-gap_v01_prompt.md
  logs/interactive/2026-06-09_investigate_bit-manip-learning-gap_v01_cli.log

If no session tag is present, do not auto-save. The user can invoke
/save-session {phase}/{category}/{topic}/{version} manually instead.
