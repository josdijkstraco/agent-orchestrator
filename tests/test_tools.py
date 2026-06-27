"""Unit tests for the built-in tools, including the ROOT_DIR sandbox.

ROOT_DIR is monkeypatched to a tmp dir so writes/reads happen in isolation while
still exercising the real path-validation logic.
"""

import pytest

import tools


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ROOT_DIR", tmp_path)
    return tmp_path


def test_read_file_returns_content(_sandbox):
    (_sandbox / "hello.txt").write_text("hi there")
    assert tools.handle_read_file({"path": "hello.txt"}) == "hi there"


def test_read_file_missing_reports_error(_sandbox):
    assert "File not found" in tools.handle_read_file({"path": "nope.txt"})


def test_read_file_outside_root_is_denied(_sandbox):
    result = tools.handle_read_file({"path": "../escape.txt"})
    assert "Access denied" in result


def test_write_file_creates_file_and_parents(_sandbox):
    msg = tools.handle_write_file({"path": "sub/dir/out.txt", "content": "data"})
    assert "Successfully wrote" in msg
    assert (_sandbox / "sub/dir/out.txt").read_text() == "data"


def test_write_file_outside_root_is_denied(_sandbox):
    assert "Access denied" in tools.handle_write_file({"path": "../evil.txt", "content": "x"})


def test_find_files_matches_glob(_sandbox):
    (_sandbox / "a.txt").write_text("")
    (_sandbox / "b.txt").write_text("")
    (_sandbox / "c.md").write_text("")
    result = tools.handle_find_files({"pattern": "*.txt"})
    assert "a.txt" in result and "b.txt" in result and "c.md" not in result


def test_find_files_no_match(_sandbox):
    assert tools.handle_find_files({"pattern": "*.none"}) == "No files found."


def test_bash_runs_in_root(_sandbox):
    (_sandbox / "marker.txt").write_text("")
    assert "marker.txt" in tools.handle_bash({"command": "ls"})


def test_bash_blocks_absolute_path_outside_root(_sandbox):
    assert "Access denied" in tools.handle_bash({"command": "cat /etc/passwd"})
