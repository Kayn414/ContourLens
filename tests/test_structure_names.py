import json
from pathlib import Path

from source.structure_names import load_aliases, match_key, normalize


def test_normalize_ignores_case_and_punctuation():
    assert normalize("Optic Nerve-Lt") == normalize("optic_nerve_lt") == "opticnervelt"


def test_aliases_pair_differently_spelled_names(tmp_path: Path):
    # Its own alias file: configs/structure_aliases.json is user-editable (the GUI's
    # "Pair with..." / remove buttons), so the test mustn't depend on its contents.
    path = tmp_path / "structure_aliases.json"
    path.write_text(json.dumps({"aliases": {"OpticNrv_L": "optic_nerve_l", "Eye_PL": "eyeball_l", "BRSTEM": "brainstem"}}))
    aliases = load_aliases(path)

    for source, model_name in [
        ("OpticNrv_L", "optic_nerve_l"),
        ("Eye_PL", "eyeball_l"),
        ("BRSTEM", "brainstem"),
        ("Brainstem", "brainstem"),  # no alias needed: normalizes the same
    ]:
        assert match_key(source, aliases) == match_key(model_name, aliases)
    assert match_key("Eye_AL", aliases) != match_key("lens_l", aliases)  # no entry, no match


def test_missing_alias_file_means_no_aliases(tmp_path: Path):
    assert load_aliases(tmp_path / "missing.json") == {}
    assert match_key("OpticNrv_L", {}) == "opticnrvl"
