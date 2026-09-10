import json
import subprocess
import sys

import pytest

from source.gui_progress import (
    PROGRESS_PREFIX,
    TqdmProgressParser,
    emit,
    run_with_progress,
)


def _bar(current: int, total: int) -> str:
    return f"\r{int(100 * current / total):3d}%|#         | {current}/{total} [00:01<00:02,  9.99it/s]"


def _nnunet_like_output(n_passes: int = 4, total: int = 5) -> str:
    parts = ["Predicting case_0000:\n"]
    for _ in range(n_passes):
        parts += [_bar(i, total) for i in range(total + 1)]
        parts.append(_bar(total, total) + "\n")  # tqdm re-prints the final state on close
    parts.append("\rCollecting results: 100%|##########| 1/1 [00:00<00:00]\n")
    parts.append("done with case_0000\n")
    return "".join(parts)


def test_parser_tracks_folds_across_arbitrary_chunk_boundaries():
    parser = TqdmProgressParser(n_passes=4)
    text = _nnunet_like_output()
    events, lines = [], []
    for start in range(0, len(text), 7):  # pipe reads split lines anywhere
        e, lines_chunk = parser.feed(text[start:start + 7])
        events += e
        lines += lines_chunk
    e, lines_chunk = parser.flush()
    events += e
    lines += lines_chunk

    predicting = [e for e in events if e.stage == "predicting"]
    fractions = [e.fraction for e in predicting]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    assert parser.pass_index == 3
    assert predicting[-1].message == "Fold 4/4 | patch 5/5"
    assert [e.stage for e in events if e.stage != "predicting"] == ["exporting"]
    assert lines == ["Predicting case_0000:", "done with case_0000"]


def test_emit_writes_one_json_progress_line(capsys):
    emit("predicting", 3, 4, "hello")
    out = capsys.readouterr().out.strip()
    assert out.startswith(PROGRESS_PREFIX)
    assert json.loads(out[len(PROGRESS_PREFIX):]) == {
        "stage": "predicting", "current": 3, "total": 4, "fraction": 0.75, "message": "hello",
    }


def test_run_with_progress_forwards_lines_and_progress(capsys):
    script = (
        "import sys\n"
        "print('starting')\n"
        "for i in range(3): sys.stderr.write(f'\\r {i}/2 [00:00<00:00]')\n"
        "sys.stderr.write('\\n')\n"
    )
    run_with_progress([sys.executable, "-c", script], TqdmProgressParser(n_passes=1))
    out = capsys.readouterr().out.splitlines()
    assert "starting" in out
    progress = [json.loads(line[len(PROGRESS_PREFIX):]) for line in out if line.startswith(PROGRESS_PREFIX)]
    assert progress and progress[-1]["fraction"] == 1.0


def test_run_with_progress_raises_on_failure():
    with pytest.raises(subprocess.CalledProcessError):
        run_with_progress([sys.executable, "-c", "raise SystemExit(3)"], TqdmProgressParser(n_passes=1))
