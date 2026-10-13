"""Model catalog: parsing, custom sources, local files and quantization choice (spec §13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lantern_talk.catalog import (
    CustomModels, choose_quant, classify, custom_entry, load_catalog, local_only_entries,
    parse_custom_source, pick_repo_file, quant_of,
)

ROOT = Path(__file__).parents[1]
CATALOG = load_catalog(ROOT / "config" / "model_catalog.yaml")
BY_ID = {e.id: e for e in CATALOG}


# ---------------------------------------------------------------- shipped catalog

def test_shipped_catalog_is_valid() -> None:
    assert len(CATALOG) == 9
    for e in CATALOG:
        assert e.variant in ("censored", "uncensored")
        assert e.hf_repo.count("/") == 1
        assert {"Q4_K_M", "Q5_K_M", "Q6_K"} <= set(e.files)
        for qf in e.files.values():
            assert qf.file.endswith(".gguf") and qf.size_bytes > 1e9
            assert qf.min_vram_gb > qf.size_gb  # KV cache and buffers come on top


def test_catalog_has_both_variants_and_german_focus() -> None:
    assert {e.variant for e in CATALOG} == {"censored", "uncensored"}
    assert sum("German" in e.languages_note for e in CATALOG) == len(CATALOG)
    assert any(e.id == "gemma4-12b" for e in CATALOG)  # starter model


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    entry = ("  - id: x\n    variant: censored\n    hf_repo: a/b\n    files:\n"
             "      Q4_K_M: {file: x.gguf, size_bytes: 5000000000}\n")
    f.write_text("models:\n" + entry + entry)
    with pytest.raises(ValueError, match="Duplicate"):
        load_catalog(f)


def test_invalid_variant_rejected(tmp_path: Path) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("models:\n  - id: x\n    variant: spicy\n    files:\n      Q4_K_M: {file: x.gguf, size_bytes: 1}\n")
    with pytest.raises(ValueError, match="variant"):
        load_catalog(f)


# ---------------------------------------------------------------- quantization choice

@pytest.mark.parametrize(
    "model, available, expected_quant, expected_status",
    [
        ("gemma4-12b", 15.0, "Q5_K_M", "fits"),        # 24 GB laptop with voice: Q5 fits
        ("gemma4-12b", 25.0, "Q5_K_M", "fits"),        # never above the profile's preference
        ("gemma4-12b", 8.5, "Q4_K_M", "fits"),         # small GPU: steps down
        ("cydonia-24b-v4.3", 15.0, "Q4_K_M", "partial"),  # 16.5 GB needed: slightly on the CPU
        ("cydonia-24b-v4.3", 26.0, "Q5_K_M", "fits"),  # 32 GB desktop
        ("skyfall-31b-v4.2", 10.0, "Q4_K_M", "too_big"),
    ],
)
def test_choose_quant(model: str, available: float, expected_quant: str, expected_status: str) -> None:
    fit = choose_quant(BY_ID[model], "Q5_K_M", available)
    assert (fit.quant, fit.status) == (expected_quant, expected_status)


def test_downloaded_quant_is_preferred() -> None:
    e = BY_ID["gemma4-12b"]
    q4 = e.files["Q4_K_M"].local_name
    assert choose_quant(e, "Q5_K_M", 20.0, downloaded={q4}).quant == "Q4_K_M"


def test_classify() -> None:
    assert classify(10, 12) == "fits"
    assert classify(15, 12) == "partial"
    assert classify(20, 12) == "too_big"


# ---------------------------------------------------------------- custom models

@pytest.mark.parametrize(
    "text, kind, repo, file",
    [
        ("TheDrummer/Cydonia-24B-v4.3-GGUF", "repo", "TheDrummer/Cydonia-24B-v4.3-GGUF", ""),
        ("https://huggingface.co/TheDrummer/Cydonia-24B-v4.3-GGUF", "repo", "TheDrummer/Cydonia-24B-v4.3-GGUF", ""),
        ("https://huggingface.co/a/b/resolve/main/x-Q4_K_M.gguf?download=true", "repo", "a/b", "x-Q4_K_M.gguf"),
        ("https://huggingface.co/a/b/blob/main/sub/x-Q5_K_M.gguf", "repo", "a/b", "sub/x-Q5_K_M.gguf"),
        ("https://example.org/models/my%20model.Q4_K_M.gguf", "url", "", "my model.Q4_K_M.gguf"),
    ],
)
def test_parse_custom_source(text: str, kind: str, repo: str, file: str) -> None:
    src = parse_custom_source(text)
    assert (src.kind, src.repo, src.file) == (kind, repo, file)


@pytest.mark.parametrize("text", ["", "hello", "ftp://x/y.gguf", "https://example.org/model.bin",
                                  "https://huggingface.co/a/b/resolve/main/readme.md"])
def test_parse_custom_source_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        parse_custom_source(text)


REPO_FILES = {
    "model-Q4_K_M.gguf": 7_000_000_000, "model-Q5_K_M.gguf": 8_000_000_000, "model-Q8_0.gguf": 12_000_000_000,
    "mmproj-model-f16.gguf": 800_000_000, "big-Q6_K-00001-of-00002.gguf": 5_000_000_000, "README.md": 10,
}


def test_pick_repo_file() -> None:
    assert pick_repo_file(REPO_FILES, None, "Q5_K_M")[0] == "model-Q5_K_M.gguf"
    assert pick_repo_file(REPO_FILES, "q8_0", "Q5_K_M")[0] == "model-Q8_0.gguf"
    assert pick_repo_file(REPO_FILES, None, "Q6_K")[0] == "model-Q5_K_M.gguf"  # next smaller
    with pytest.raises(ValueError, match="No Q6_K"):
        pick_repo_file(REPO_FILES, "Q6_K", "Q5_K_M")  # only exists split
    with pytest.raises(ValueError, match="No single-file GGUF"):
        pick_repo_file({"a-00001-of-00003.gguf": 1}, None, "Q4_K_M")


def test_quant_of() -> None:
    assert quant_of("gemma-4-12b-it-Q5_K_M.gguf") == "Q5_K_M"
    assert quant_of("x.i1-Q4_K_M.gguf") == "Q4_K_M"
    assert quant_of("model.IQ4_XS.gguf") == "IQ4_XS"
    assert quant_of("plain-model.gguf") == "?"


def test_custom_models_store(tmp_path: Path) -> None:
    store = CustomModels(tmp_path / "custom_models.json")
    src = parse_custom_source("someone/Cool-RP-12B-GGUF")
    entry = custom_entry(src, "Cool-RP-12B-Q4_K_M.gguf", 7_100_000_000, "uncensored")
    store.add(entry)
    reloaded = CustomModels(tmp_path / "custom_models.json").entries()
    assert len(reloaded) == 1 and reloaded[0].variant == "uncensored"
    assert reloaded[0].display_name == "Cool-RP-12B"
    assert reloaded[0].files["Q4_K_M"].min_vram_gb >= 7.1
    store.set_label("Cool-RP-12B-Q4_K_M.gguf", "censored")
    assert CustomModels(tmp_path / "custom_models.json").entries()[0].variant == "censored"
    store.remove(entry["id"])
    assert CustomModels(tmp_path / "custom_models.json").entries() == []


def test_local_files_are_detected(tmp_path: Path) -> None:
    llm = tmp_path / "llm"
    llm.mkdir()
    (llm / "my-own-model.Q4_K_M.gguf").write_bytes(b"x" * 10)
    (llm / CATALOG[0].files["Q4_K_M"].local_name).write_bytes(b"x")  # belongs to the catalog
    (llm / "partial.gguf.part").write_bytes(b"x")
    local = local_only_entries(llm, CATALOG, {"my-own-model.Q4_K_M.gguf": "uncensored"})
    assert [e.display_name for e in local] == ["my-own-model.Q4_K_M"]
    assert local[0].variant == "uncensored" and local[0].source == "local"


def test_broken_custom_file_is_ignored(tmp_path: Path) -> None:
    f = tmp_path / "custom_models.json"
    f.write_text("{broken")
    assert CustomModels(f).entries() == []
