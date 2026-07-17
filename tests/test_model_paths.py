from pathlib import Path

from provise.models.generative import PROJECT_ROOT, create_model


def test_project_root_discovers_repo_root():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "provise").is_dir()


def test_joyai_model_path_uses_local_models_root_override(monkeypatch):
    monkeypatch.setenv("PROVISE_LOCAL_MODELS_DIR", "/tmp/provise-models")
    model = create_model("joyai-image")

    assert model.model_path == "/tmp/provise-models/JoyAI-Image-Edit"


def test_joyai_specific_overrides_take_priority(monkeypatch):
    monkeypatch.setenv("PROVISE_LOCAL_MODELS_DIR", "/tmp/provise-models")
    monkeypatch.setenv("PROVISE_JOYAI_MODEL_PATH", "/tmp/custom/JoyAI-Image-Edit")
    monkeypatch.setenv("PROVISE_JOYAI_REPO_PATH", "/tmp/repos/JoyAI-Image")
    model = create_model("joyai-image")

    assert model.model_path == "/tmp/custom/JoyAI-Image-Edit"
    assert model.repo_path == "/tmp/repos/JoyAI-Image"


def test_janus_falls_back_to_repo_checkpoint_when_present(monkeypatch, tmp_path):
    repo_dir = tmp_path / "Janus"
    checkpoint_dir = repo_dir / "deepseek-ai" / "Janus-Pro-7B"
    checkpoint_dir.mkdir(parents=True)
    monkeypatch.setenv("PROVISE_LOCAL_MODELS_DIR", str(tmp_path / "local_models"))
    monkeypatch.setenv("PROVISE_JANUS_REPO_PATH", str(repo_dir))
    model = create_model("janus-pro-7b")

    assert Path(model.model_path) == checkpoint_dir
    assert model.repo_path == str(repo_dir)
