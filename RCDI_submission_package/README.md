# RCDI submission package

Paper: "Beyond Prediction: Risk-Calibrated AI Decision Intelligence for Reliable and Cost-Aware Data-Driven Decision Making" (target: Knowledge-Based Systems).

Start with COMPLIANCE_AND_AUDIT_NOTES.md.

- RCDI_full_paper.docx / .pdf: full paper (20 pages)
- RCDI_short_communication.docx / .pdf: short version (8 pages)
- RCDI_supplementary_material.docx / .pdf: Tables S1-S4
- highlights.txt, graphical_abstract/, figures/, tables/
- code/: pipeline (collect.py needs a GPU; rescore.py and analyze.py do not)
- data/repaired/: records used for the main text; data/strict/: first-pass scoring (with the model-output cache, 20 MB) kept as a robustness check
- logs/: collection and analysis logs of the real run

Regenerate everything from the records (no GPU):
    cd code && python analyze.py --res ../data/repaired --reps 200
Re-score from the cache:
    python rescore.py --res ../data/strict --out ../data/repaired --gsm8k-jsonl gsm8k_test.jsonl

Upload to GitHub (from the folder code/):
    git init && git add . && git commit -m "RCDI code" && git branch -M main
    git remote add origin https://github.com/akhilareddycirumani/<repo> && git push -u origin main
(or use "Add file > Upload files" on the GitHub website).
