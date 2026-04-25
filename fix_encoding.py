import pathlib

# CP1252 mojibake -> correct UTF-8 characters
# Each bad sequence is what you get when UTF-8 bytes are decoded as CP1252
# and then re-encoded as UTF-8. Fix by replacing with the intended Unicode char.
fixes = [
    ('\u00e2\u0153\u201c', '\u2713'),  # âœ"  -> ✓  (U+2713 CHECK MARK)
    ('\u00e2\u0153\u2014', '\u2717'),  # âœ—  -> ✗  (U+2717 BALLOT X)
    ('\u00e2\u20ac\u201d', '\u2014'),  # â€"  -> —  (U+2014 EM DASH)
    ('\u00e2\u201d\u20ac', '\u2500'),  # â"€  -> ─  (U+2500 BOX DRAWINGS LIGHT HORIZONTAL)
    ('\u00e2\u2020\u2019', '\u2192'),  # â†'  -> →  (U+2192 RIGHTWARDS ARROW)
    ('\u00e2\u2020\u0090', '\u2190'),  # â†\x90 -> ← (U+2190 LEFTWARDS ARROW)
]

targets = [
    pathlib.Path('notebooks/01_run_pipeline.ipynb'),
    pathlib.Path('src/evaluation.py'),
    pathlib.Path('src/preprocessing.py'),
    pathlib.Path('src/visualization.py'),
    pathlib.Path('src/models/bnn_model.py'),
    pathlib.Path('src/models/meta_learner.py'),
    pathlib.Path('src/models/nhits_model.py'),
    pathlib.Path('src/models/xgboost_model.py'),
    pathlib.Path('config.py'),
    pathlib.Path('README.md'),
]

for path in targets:
    if not path.exists():
        continue
    text = path.read_text(encoding='utf-8')
    changed = False
    for bad, good in fixes:
        count = text.count(bad)
        if count:
            print(f'{path}: {repr(bad)} -> {repr(good)} ({count}x)')
            text = text.replace(bad, good)
            changed = True
    if changed:
        path.write_text(text, encoding='utf-8')

print('Done.')
