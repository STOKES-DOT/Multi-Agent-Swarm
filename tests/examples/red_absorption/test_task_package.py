from __future__ import annotations

from pathlib import Path
import time

import yaml

from multi_agent_pso.configuration import load_task_package
import multi_agent_pso.configuration.loader as loader
from multi_agent_pso.configuration.models import SnapshotConfig


TASK=Path("examples/red_absorption/task.yaml")


def test_task_yaml_has_five_by_five_readonly_production_contract() -> None:
    raw=yaml.safe_load(TASK.read_text(encoding="utf-8")); assert raw["pso"]["population_size"]==raw["pso"]["iterations"]==5
    assert raw["pso"]["topology"]["type"]=="ring"; assert raw["wiki"]["read_only"] is True; assert raw["agent"]["model"]=="gpt-5.6-terra"
    text=TASK.read_text(); assert "parent" not in raw and "spectrum_argv" not in text and "backend:" not in text


def test_task_package_loads_real_maintained_wiki_without_raw_snapshot() -> None:
    started=time.monotonic(); package=load_task_package(TASK); elapsed=time.monotonic()-started
    assert package.spec.pso.population_size==5; assert elapsed<20
    assert package.plugins.task_adapter is not None; assert len(package.schema_bytes)==3
    wiki=[entry for entry in package.manifest.entries if entry.role=="wiki"]
    assert wiki and sum(entry.size_bytes for entry in wiki)<10*1024*1024
    assert not any(entry.path.startswith(("raw/","derived/",".obsidian/","structures/")) for entry in wiki)


def test_maintained_wiki_snapshot_tracks_markdown_not_raw(tmp_path:Path) -> None:
    (tmp_path/"AGENTS.md").write_text("rules"); (tmp_path/"index.md").write_text("index")
    (tmp_path/"sources").mkdir(); source=tmp_path/"sources"/"a.md"; source.write_text("one")
    (tmp_path/"raw").mkdir(); raw=tmp_path/"raw"/"x.txt"; raw.write_text("raw-one")
    def entries():
        builder=loader._ManifestBuilder(SnapshotConfig())
        loader._add_maintained_wiki_entries(builder,tmp_path)
        return tuple((entry.path,entry.sha256) for entry in builder.entries)
    first=entries(); raw.write_text("raw-two"); assert entries()==first
    source.write_text("two"); assert entries()!=first
    bad=tmp_path/"sources"/"bad.md"; bad.symlink_to(raw)
    import pytest
    with pytest.raises(ValueError): entries()
