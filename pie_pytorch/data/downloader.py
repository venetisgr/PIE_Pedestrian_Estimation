"""PIE downloader(s): videos + annotations.

Videos come from the York University mirror:
    https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/

Annotations come from the official github mirror (no auth needed):
    https://github.com/aras62/PIE/archive/refs/heads/master.tar.gz
    -> extract only annotations/, annotations_attributes/, annotations_vehicle/

The full video inventory (6 sets, 53 videos, ~74 GB) is hard-coded
below. Hard-coding avoids re-scraping the HTML index on every run and
gives us a stable manifest to validate against.

Usage (Python):
    from pie_pytorch.data.downloader import download_videos, download_annotations
    download_videos(dest='/data/pie', sets=['set03'])
    download_annotations(dest='/data/pie')

Usage (CLI):
    # videos
    python -m pie_pytorch.data.downloader videos --dest /data/pie --sets set03
    python -m pie_pytorch.data.downloader videos --dest /data/pie --videos set03:video_0001
    python -m pie_pytorch.data.downloader videos --dest /data/pie --all --workers 4
    python -m pie_pytorch.data.downloader videos --dry-run --sets set03 set05

    # annotations
    python -m pie_pytorch.data.downloader annotations --dest /data/pie

Downloads resume by default (Range requests for videos, temp-file rename
for the annotation tarball). A local file matching server-reported
Content-Length is skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures as _cf
import io
import os
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips"
ANNOTATIONS_TARBALL_URL = (
    "https://github.com/aras62/PIE/archive/refs/heads/master.tar.gz"
)
ANNOTATION_DIRS = ("annotations", "annotations_attributes", "annotations_vehicle")

# Full video inventory (verified 2026-04-16 via HTML index of each set dir).
INVENTORY: dict[str, list[str]] = {
    "set01": [f"video_{i:04d}.mp4" for i in range(1, 5)],  # 4
    "set02": [f"video_{i:04d}.mp4" for i in range(1, 4)],  # 3
    "set03": [f"video_{i:04d}.mp4" for i in range(1, 20)],  # 19
    "set04": [f"video_{i:04d}.mp4" for i in range(1, 17)],  # 16
    "set05": [f"video_{i:04d}.mp4" for i in range(1, 3)],  # 2
    "set06": [f"video_{i:04d}.mp4" for i in range(1, 10)],  # 9
}


@dataclass(frozen=True)
class VideoRef:
    set_id: str
    filename: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.set_id}/{self.filename}"

    def local_path(self, dest_root: Path) -> Path:
        return dest_root / "PIE_clips" / self.set_id / self.filename


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def all_videos() -> list[VideoRef]:
    return [VideoRef(s, v) for s, vs in INVENTORY.items() for v in vs]


def select(
    sets: list[str] | None = None,
    videos: list[str] | None = None,
) -> list[VideoRef]:
    """Build a VideoRef list from high-level selectors.

    - ``sets``    : list of set_ids (e.g. ['set03']) -> all videos in those sets.
    - ``videos``  : list of 'setXX:video_YYYY' entries for single-file grabs.
      You can pass either or both; the result is the union deduplicated.
    """
    refs: list[VideoRef] = []
    if sets:
        for s in sets:
            if s not in INVENTORY:
                raise ValueError(f"unknown set {s!r}; known: {sorted(INVENTORY)}")
            for v in INVENTORY[s]:
                refs.append(VideoRef(s, v))
    if videos:
        for entry in videos:
            if ":" not in entry:
                raise ValueError(
                    f"--videos entries must be 'setXX:video_YYYY', got {entry!r}"
                )
            s, v = entry.split(":", 1)
            if not v.endswith(".mp4"):
                v = v + ".mp4"
            if s not in INVENTORY or v not in INVENTORY[s]:
                raise ValueError(f"unknown video {s}:{v}")
            refs.append(VideoRef(s, v))

    seen: set[tuple[str, str]] = set()
    dedup: list[VideoRef] = []
    for r in refs:
        key = (r.set_id, r.filename)
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    return dedup


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------
def _remote_size(url: str, timeout: int = 30) -> int:
    """Return the server-reported Content-Length (bytes). -1 if unknown."""
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        cl = r.headers.get("Content-Length")
        return int(cl) if cl is not None else -1


def _open_range(url: str, start: int, timeout: int = 60):
    """Open a streaming GET with a Range header starting at ``start`` bytes."""
    req = urllib.request.Request(url)
    if start > 0:
        req.add_header("Range", f"bytes={start}-")
    return urllib.request.urlopen(req, timeout=timeout)


# ---------------------------------------------------------------------------
# Per-video download with resume
# ---------------------------------------------------------------------------
def download_one(
    ref: VideoRef,
    dest_root: Path,
    resume: bool = True,
    chunk: int = 1 << 20,  # 1 MiB
    show_progress: bool = True,
) -> Path:
    """Download a single video. Returns the final path.

    Behavior:
      - Creates the destination dir tree.
      - HEAD to get total size; if the local file already equals that
        size, skips (fast-path for repeated runs).
      - Otherwise streams bytes in chunks, appending to a ``.part`` file,
        then atomically renames to the final name when complete.
      - ``resume=True`` preserves existing bytes in ``.part``; ``False``
        starts from scratch.
    """
    final = ref.local_path(dest_root)
    partial = final.with_suffix(final.suffix + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)

    total = -1
    try:
        total = _remote_size(ref.url)
    except (urllib.error.URLError, OSError) as e:  # network offline / 4xx / 5xx
        raise RuntimeError(f"HEAD failed for {ref.url}: {e}") from e

    if final.exists() and total > 0 and final.stat().st_size == total:
        if show_progress:
            tqdm.write(f"[skip] {ref.set_id}/{ref.filename} already complete")
        return final

    start = partial.stat().st_size if (resume and partial.exists()) else 0
    if not resume and partial.exists():
        partial.unlink()

    mode = "ab" if start > 0 else "wb"
    desc = f"{ref.set_id}/{ref.filename}"
    with _open_range(ref.url, start) as resp, open(partial, mode) as fh:
        pbar = tqdm(
            total=total if total > 0 else None,
            initial=start,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            disable=not show_progress,
            leave=False,
        )
        try:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                fh.write(buf)
                pbar.update(len(buf))
        finally:
            pbar.close()

    if total > 0 and partial.stat().st_size != total:
        raise RuntimeError(
            f"size mismatch for {desc}: got {partial.stat().st_size} "
            f"bytes, expected {total}"
        )
    os.replace(partial, final)
    return final


# ---------------------------------------------------------------------------
# Batch entry points
# ---------------------------------------------------------------------------
def download_videos(
    dest: str | os.PathLike,
    sets: list[str] | None = None,
    videos: list[str] | None = None,
    all_: bool = False,
    workers: int = 1,
    resume: bool = True,
    dry_run: bool = False,
    show_progress: bool = True,
) -> list[Path]:
    """Download a selection of PIE videos to ``dest``."""
    if all_:
        refs = all_videos()
    else:
        refs = select(sets=sets, videos=videos)

    if not refs:
        raise ValueError(
            "No videos selected. Pass one of: sets=..., videos=..., all_=True"
        )

    dest_root = Path(dest).expanduser().resolve()
    if dry_run:
        total_bytes = 0
        for r in refs:
            try:
                n = _remote_size(r.url)
            except OSError:
                n = -1
            total_bytes += max(n, 0)
            local = r.local_path(dest_root)
            status = "cached" if local.exists() else "download"
            print(f"[{status}] {r.set_id}/{r.filename}  ({n} bytes)")
        print(
            f"--- total: {len(refs)} files, "
            f"~{total_bytes / 1024 ** 3:.1f} GB (server-reported)"
        )
        return []

    out_paths: list[Path] = []
    if workers <= 1:
        for r in tqdm(refs, desc="videos", disable=not show_progress):
            out_paths.append(
                download_one(r, dest_root, resume=resume, show_progress=show_progress)
            )
    else:
        with _cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(download_one, r, dest_root, resume, 1 << 20, show_progress)
                for r in refs
            ]
            for fut in tqdm(
                _cf.as_completed(futures),
                total=len(futures),
                desc="videos",
                disable=not show_progress,
            ):
                out_paths.append(fut.result())
    return out_paths


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------
def _fetch_to_buffer(
    url: str,
    timeout: int = 300,
    show_progress: bool = True,
    desc: str = "download",
) -> bytes:
    """Stream a URL into memory with a progress bar."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        total_hdr = r.headers.get("Content-Length")
        total = int(total_hdr) if total_hdr else None
        buf = io.BytesIO()
        pbar = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            disable=not show_progress,
            leave=False,
        )
        try:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                buf.write(chunk)
                pbar.update(len(chunk))
        finally:
            pbar.close()
        return buf.getvalue()


def download_annotations(
    dest: str | os.PathLike,
    url: str = ANNOTATIONS_TARBALL_URL,
    overwrite: bool = False,
    show_progress: bool = True,
) -> list[Path]:
    """Download aras62/PIE tarball and extract only the annotation dirs to ``dest``.

    Result layout:
        {dest}/annotations/setXX/*.xml
        {dest}/annotations_attributes/setXX/*.xml
        {dest}/annotations_vehicle/setXX/*.xml

    Idempotent: if ``{dest}/annotations/`` already exists and ``overwrite`` is
    False, this function is a no-op and returns ``[]``.
    """
    dest_root = Path(dest).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    existing = [d for d in ANNOTATION_DIRS if (dest_root / d).is_dir()]
    if existing and not overwrite:
        if show_progress:
            print(
                f"[skip] annotations already present under {dest_root} "
                f"({', '.join(existing)}). Pass overwrite=True to replace."
            )
        return []

    raw = _fetch_to_buffer(
        url,
        show_progress=show_progress,
        desc="annotations.tar.gz",
    )

    extracted: list[Path] = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Tarball layout: 'PIE-master/annotations/...'
            name = member.name
            parts = name.split("/", 2)
            if len(parts) < 2:
                continue
            top_rel = parts[1]  # 'annotations' or similar
            if top_rel not in ANNOTATION_DIRS:
                continue
            # Re-root: drop the 'PIE-master/' prefix.
            target_rel = "/".join(parts[1:])
            target = dest_root / target_rel
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            # Atomic-ish write via a temp file
            tmp = target.with_suffix(target.suffix + ".part")
            with open(tmp, "wb") as out:
                out.write(fobj.read())
            os.replace(tmp, target)
            extracted.append(target)

    if not extracted:
        raise RuntimeError(
            f"no annotation files found in tarball at {url}. "
            "Upstream repo layout may have changed."
        )
    if show_progress:
        print(f"[ok] extracted {len(extracted)} annotation files to {dest_root}")
    return extracted


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_videos_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "videos",
        help="Download PIE video clips (resumable).",
    )
    p.add_argument(
        "--dest",
        required=True,
        help="Root output directory. Videos land under {dest}/PIE_clips/setXX/.",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true", help="Download every set.")
    grp.add_argument("--sets", nargs="+", metavar="setXX", help="Whole-set downloads.")
    grp.add_argument(
        "--videos",
        nargs="+",
        metavar="setXX:video_YYYY",
        help="Specific videos, e.g. set03:video_0001",
    )
    p.add_argument("--workers", type=int, default=1, help="Parallel workers.")
    p.add_argument("--no-resume", action="store_true", help="Restart partials.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would download and total size, without fetching.",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress tqdm bars.")
    p.set_defaults(_handler=_run_videos)


def _add_annotations_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "annotations",
        help="Download PIE XML annotations from aras62/PIE.",
    )
    p.add_argument("--dest", required=True, help="Root output directory.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing annotation dirs (default: skip if present).",
    )
    p.add_argument(
        "--url",
        default=ANNOTATIONS_TARBALL_URL,
        help=f"Tarball URL (default: {ANNOTATIONS_TARBALL_URL}).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    p.set_defaults(_handler=_run_annotations)


def _run_videos(args: argparse.Namespace) -> int:
    download_videos(
        dest=args.dest,
        sets=args.sets,
        videos=args.videos,
        all_=args.all,
        workers=args.workers,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        show_progress=not args.quiet,
    )
    return 0


def _run_annotations(args: argparse.Namespace) -> int:
    download_annotations(
        dest=args.dest,
        url=args.url,
        overwrite=args.overwrite,
        show_progress=not args.quiet,
    )
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pie-download",
        description="Download PIE dataset assets (videos and annotations).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    _add_videos_parser(sub)
    _add_annotations_parser(sub)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return args._handler(args)
    except (ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
