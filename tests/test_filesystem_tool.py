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
