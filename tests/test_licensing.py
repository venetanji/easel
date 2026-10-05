"""Keep source, package metadata, and container license materials aligned."""
from pathlib import Path
import hashlib
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_package_declares_scoped_gpl_and_includes_license_materials():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    # Removing the own-code grant must fail this contract.
    assert project.get("license") == "GPL-3.0-or-later"
    assert project.get("readme") == "README.md"
    patterns = project.get("license-files", [])
    included = {path.relative_to(ROOT).as_posix() for pattern in patterns for path in ROOT.glob(pattern)}
    assert {
        "LICENSE", "NOTICE", "licenses/creative-skills-MIT.txt",
        "licenses/comfy-workflow-templates-MIT.txt", "licenses/ComfyUI-GPL-3.0.txt",
    } <= included


def test_full_gpl_text_and_scoped_notice_are_present():
    assert (ROOT / "LICENSE").is_file(), "Ship the full GPLv3 text"
    license_text = (ROOT / "LICENSE").read_text()
    assert "Version 3, 29 June 2007" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
    assert len(license_text) > 35000
    assert (ROOT / "NOTICE").is_file(), "Explain the grant and upstream exceptions"
    notice = (ROOT / "NOTICE").read_text()
    for required in (
        "Copyright (c) 2026 Venetanji", "GPL-3.0-or-later",
        "either version 3 of the License, or (at your option) any later version",
        "WITHOUT ANY WARRANTY", "creative-skills-MIT.txt",
        "comfy-workflow-templates-MIT.txt", "ComfyUI-GPL-3.0.txt",
        "Model weights", "runtime_nodes.json",
    ):
        assert required in notice


@pytest.mark.parametrize("name,copyright", [
    ("creative-skills-MIT.txt", "Copyright (c) 2026 Venetanji"),
    ("comfy-workflow-templates-MIT.txt", "Copyright (c) 2023-present Comfy Org"),
])
def test_retained_mit_notices_have_complete_grant(name, copyright):
    path = ROOT / "licenses" / name
    assert path.is_file(), f"Retain upstream notice: {name}"
    content = path.read_text()
    assert copyright in content
    assert "Permission is hereby granted, free of charge" in content
    assert "The above copyright notice and this permission notice shall be included" in content
    assert "SOFTWARE." in content


def test_comfyui_license_is_preserved_verbatim():
    path = ROOT / "licenses/ComfyUI-GPL-3.0.txt"
    assert path.is_file(), "Retain the upstream ComfyUI GPLv3 text"
    # Git blob digest at b5cc8830279eae909a59de030af1e50761c36751.
    content = path.read_bytes()
    digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    assert digest == "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7"


def test_container_copies_license_and_readme_materials_before_build():
    dockerfile = (ROOT / "Dockerfile").read_text()
    before_build = dockerfile.split("RUN uv sync", maxsplit=1)[0]
    copy_lines = [line.split() for line in before_build.splitlines() if line.startswith("COPY ")]
    copied = {source.rstrip("/") for line in copy_lines for source in line[1:-1]}
    assert {"LICENSE", "NOTICE", "README.md", "licenses"} <= copied


def test_graph_modules_point_to_retained_notices():
    for name in ("flux_graph.py", "workflow_graph.py", "h3_graph.py", "video_graph.py"):
        source = (ROOT / "easel" / name).read_text()
        assert "NOTICE" in source, f"Keep license provenance visible in {name}"
