"""Tests for pie_pytorch.data.downloader.

All tests run against a local threaded HTTP server and a tarball built
in a tmp dir. No real network calls.
"""

from __future__ import annotations

import http.server
import io
import os
import socketserver
import tarfile
import threading
from pathlib import Path
from unittest import mock

import pytest

from pie_pytorch.data import downloader as dl


# ---------------------------------------------------------------------------
# Local HTTP server fixture (serves a tmp dir, supports Range)
# ---------------------------------------------------------------------------
class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with Range support for resume testing."""

    def log_message(self, *a, **kw):  # silence
        return

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        if not rng or not rng.startswith("bytes="):
            return super().do_GET()
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            self.send_error(404)
            return
        try:
            start_s, _end_s = rng[len("bytes=") :].split("-", 1)
            start = int(start_s)
        except ValueError:
            self.send_error(416)
            return
        size = os.path.getsize(path)
        if start >= size:
            self.send_error(416)
            return
        self.send_response(206)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(size - start))
        self.send_header("Content-Range", f"bytes {start}-{size - 1}/{size}")
        self.end_headers()
        with open(path, "rb") as fh:
            fh.seek(start)
            self.wfile.write(fh.read())


@pytest.fixture
def http_root(tmp_path_factory):
    """A directory served over HTTP. Yields (root_path, base_url)."""
    root = tmp_path_factory.mktemp("httproot")
    os.chdir(root)  # SimpleHTTPRequestHandler serves CWD

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _RangeHandler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield Path(root), f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# Inventory / selection
# ---------------------------------------------------------------------------
def test_inventory_totals():
    """Sanity-check the hard-coded inventory against the published counts."""
    assert sum(len(v) for v in dl.INVENTORY.values()) == 53
    assert list(dl.INVENTORY) == [
        "set01", "set02", "set03", "set04", "set05", "set06",
    ]


def test_select_by_sets():
    refs = dl.select(sets=["set01", "set05"])
    assert [(r.set_id, r.filename) for r in refs] == [
        ("set01", "video_0001.mp4"),
        ("set01", "video_0002.mp4"),
        ("set01", "video_0003.mp4"),
        ("set01", "video_0004.mp4"),
        ("set05", "video_0001.mp4"),
        ("set05", "video_0002.mp4"),
    ]


def test_select_by_videos_and_union_dedup():
    refs = dl.select(
        sets=["set05"],
        videos=["set05:video_0001", "set05:video_0002.mp4"],
    )
    assert len(refs) == 2  # set05 already gave 2, videos dup them


def test_select_errors():
    with pytest.raises(ValueError):
        dl.select(sets=["set99"])
    with pytest.raises(ValueError):
        dl.select(videos=["set01/video_0001"])  # missing ':'
    with pytest.raises(ValueError):
        dl.select(videos=["set01:video_9999"])


# ---------------------------------------------------------------------------
# download_one round-trip with a local server
# ---------------------------------------------------------------------------
def _make_fake_set(root: Path, set_id: str, videos: dict[str, bytes]) -> None:
    d = root / "PIE_dataset" / "PIE_clips" / set_id
    d.mkdir(parents=True, exist_ok=True)
    for name, payload in videos.items():
        (d / name).write_bytes(payload)


def test_download_one_fetches_bytes(http_root, tmp_path, monkeypatch):
    root, base = http_root
    payload = b"\x00\x01\x02" * 1000
    _make_fake_set(root, "set01", {"video_0001.mp4": payload})

    monkeypatch.setattr(
        dl, "BASE_URL", f"{base}/PIE_dataset/PIE_clips", raising=True
    )
    ref = dl.VideoRef("set01", "video_0001.mp4")
    out = dl.download_one(ref, tmp_path, show_progress=False)
    assert out.read_bytes() == payload


def test_download_one_skips_when_complete(http_root, tmp_path, monkeypatch):
    root, base = http_root
    payload = b"ABCDEFGHIJ" * 200
    _make_fake_set(root, "set02", {"video_0001.mp4": payload})
    monkeypatch.setattr(dl, "BASE_URL", f"{base}/PIE_dataset/PIE_clips")

    ref = dl.VideoRef("set02", "video_0001.mp4")
    # Pre-seed the local file
    final = ref.local_path(tmp_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(payload)
    mtime_before = final.stat().st_mtime_ns

    dl.download_one(ref, tmp_path, show_progress=False)
    # Should be untouched (fast-path skip)
    assert final.stat().st_mtime_ns == mtime_before


def test_download_one_resumes_partial(http_root, tmp_path, monkeypatch):
    root, base = http_root
    payload = b"0123456789" * 512
    _make_fake_set(root, "set01", {"video_0003.mp4": payload})
    monkeypatch.setattr(dl, "BASE_URL", f"{base}/PIE_dataset/PIE_clips")

    ref = dl.VideoRef("set01", "video_0003.mp4")
    final = ref.local_path(tmp_path)
    partial = final.with_suffix(final.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    # Seed with the first 1000 bytes
    partial.write_bytes(payload[:1000])

    dl.download_one(ref, tmp_path, resume=True, show_progress=False)
    assert final.read_bytes() == payload
    assert not partial.exists()


def test_download_videos_batch(http_root, tmp_path, monkeypatch):
    root, base = http_root
    _make_fake_set(
        root,
        "set05",
        {
            "video_0001.mp4": b"aaa" * 100,
            "video_0002.mp4": b"bbb" * 100,
        },
    )
    monkeypatch.setattr(dl, "BASE_URL", f"{base}/PIE_dataset/PIE_clips")

    paths = dl.download_videos(
        dest=tmp_path, sets=["set05"], workers=1, show_progress=False
    )
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_download_videos_requires_selection(tmp_path):
    with pytest.raises(ValueError):
        dl.download_videos(dest=tmp_path, show_progress=False)


# ---------------------------------------------------------------------------
# Annotations downloader
# ---------------------------------------------------------------------------
def _make_fake_pie_tarball() -> bytes:
    """Build an in-memory tar.gz shaped like aras62/PIE."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Some ignored files + the three annotation dirs.
        contents = {
            "PIE-master/README.md": b"# PIE",
            "PIE-master/annotations/set01/video_0001_annt.xml": b"<root>A</root>",
            "PIE-master/annotations/set02/video_0001_annt.xml": b"<root>B</root>",
            "PIE-master/annotations_attributes/set01/video_0001_attributes.xml": b"<r>C</r>",
            "PIE-master/annotations_vehicle/set01/video_0001_obd.xml": b"<r>D</r>",
            "PIE-master/utilities/pie_data.py": b"# code",
        }
        for name, data in contents.items():
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tar.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_download_annotations_extracts_only_annotation_dirs(tmp_path, http_root):
    root, base = http_root
    tarball = _make_fake_pie_tarball()
    (root / "PIE.tar.gz").write_bytes(tarball)

    extracted = dl.download_annotations(
        dest=tmp_path,
        url=f"{base}/PIE.tar.gz",
        show_progress=False,
    )
    assert len(extracted) == 4
    assert (tmp_path / "annotations" / "set01" / "video_0001_annt.xml").exists()
    assert (tmp_path / "annotations" / "set02" / "video_0001_annt.xml").exists()
    assert (tmp_path / "annotations_attributes" / "set01" / "video_0001_attributes.xml").exists()
    assert (tmp_path / "annotations_vehicle" / "set01" / "video_0001_obd.xml").exists()
    # Non-annotation files must NOT be extracted
    assert not (tmp_path / "README.md").exists()
    assert not (tmp_path / "utilities").exists()


def test_download_annotations_skips_if_present(tmp_path, capsys):
    (tmp_path / "annotations").mkdir()
    # URL is unreachable, but the skip happens before any network call.
    out = dl.download_annotations(
        dest=tmp_path, url="http://127.0.0.1:1/never", show_progress=True
    )
    assert out == []
    assert "[skip]" in capsys.readouterr().out


def test_download_annotations_raises_on_empty_tarball(tmp_path, http_root):
    root, base = http_root
    # Tarball with no annotation dirs
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        ti = tarfile.TarInfo(name="PIE-master/other.txt")
        ti.size = 3
        tar.addfile(ti, io.BytesIO(b"hi\n"))
    (root / "empty.tar.gz").write_bytes(buf.getvalue())

    with pytest.raises(RuntimeError, match="no annotation files"):
        dl.download_annotations(
            dest=tmp_path,
            url=f"{base}/empty.tar.gz",
            show_progress=False,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_cli_videos_requires_selection(tmp_path, capsys):
    # argparse raises SystemExit(2) when a required mutex-group is missing.
    with pytest.raises(SystemExit) as exc:
        dl.main(["videos", "--dest", str(tmp_path)])
    assert exc.value.code == 2


def test_cli_videos_dry_run(http_root, tmp_path, monkeypatch, capsys):
    root, base = http_root
    _make_fake_set(root, "set05", {"video_0001.mp4": b"x" * 10, "video_0002.mp4": b"y" * 20})
    monkeypatch.setattr(dl, "BASE_URL", f"{base}/PIE_dataset/PIE_clips")
    rc = dl.main(["videos", "--dest", str(tmp_path), "--sets", "set05", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "set05/video_0001.mp4" in out
    assert "set05/video_0002.mp4" in out


def test_cli_annotations(http_root, tmp_path, monkeypatch):
    root, base = http_root
    (root / "PIE.tar.gz").write_bytes(_make_fake_pie_tarball())
    rc = dl.main(
        [
            "annotations",
            "--dest",
            str(tmp_path),
            "--url",
            f"{base}/PIE.tar.gz",
            "--quiet",
        ]
    )
    assert rc == 0
    assert (tmp_path / "annotations" / "set01").exists()
