import os
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
from pathlib import Path
from main import FileAnalyzer, organize_files

class DummyLLM:
    pass

def test_scan_directory_excludes_output(tmp_path):
    input_dir = tmp_path / "in"
    output_dir = input_dir / "organized"
    output_dir.mkdir(parents=True)
    (input_dir / "a.txt").write_text("hi")
    (output_dir / "b.txt").write_text("skip")

    files = FileAnalyzer.scan_directory(str(input_dir), exclude_dirs={str(output_dir)})
    assert str(input_dir / "a.txt") in files
    assert str(output_dir / "b.txt") not in files

def test_organize_files_collision_and_sanitize(tmp_path):
    src1 = tmp_path / "a" / "test.txt"
    src1.parent.mkdir()
    src1.write_text("x")
    src2 = tmp_path / "b" / "test.txt"
    src2.parent.mkdir()
    src2.write_text("y")
    out_dir = tmp_path / "out"
    structure = {"My:Folder": [str(src1), str(src2)]}
    organize_files(structure, str(out_dir))
    target = out_dir / "My_Folder"
    names = sorted(p.name for p in target.iterdir())
    assert names == ["test.txt", "test_1.txt"]
