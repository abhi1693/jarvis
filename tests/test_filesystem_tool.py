from app.tools.filesystem import FilesystemTool


def test_search_text_skips_virtualenv_like_directories(tmp_path):
    tool = FilesystemTool(tmp_path)
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "ignored.py").write_text("TODO: do not report me", encoding="utf-8")
    (tmp_path / "app.py").write_text("TODO: report me", encoding="utf-8")

    result = tool.search_text("TODO:", ".")

    assert result["ok"] is True
    assert len(result["matches"]) == 1
    assert result["matches"][0]["path"] == "app.py"


def test_search_text_allows_agent_brain_under_data(tmp_path):
    brain_root = tmp_path / "data" / "agent_brain"
    brain_root.mkdir(parents=True)
    (brain_root / "persona.md").write_text("operator: Asha", encoding="utf-8")
    (tmp_path / "data" / "snapshots").mkdir(parents=True)
    (tmp_path / "data" / "snapshots" / "ignore.txt").write_text("operator: hidden", encoding="utf-8")
    tool = FilesystemTool(tmp_path, brain_root)

    result = tool.search_text("operator", "data")

    assert result["ok"] is True
    assert len(result["matches"]) == 1
    assert result["matches"][0]["path"] == "data/agent_brain/persona.md"


def test_move_path_and_make_directory_support_brain_workspace(tmp_path):
    brain_root = tmp_path / "data" / "agent_brain"
    tool = FilesystemTool(tmp_path, brain_root)

    mkdir_result = tool.make_directory("data/agent_brain/workspace/plans")
    assert mkdir_result["ok"] is True

    source = tmp_path / "data" / "agent_brain" / "workspace" / "draft.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("draft", encoding="utf-8")

    move_result = tool.move_path(
        "data/agent_brain/workspace/draft.md",
        "data/agent_brain/workspace/plans/draft.md",
    )

    assert move_result["ok"] is True
    assert (tmp_path / "data" / "agent_brain" / "workspace" / "plans" / "draft.md").exists()
