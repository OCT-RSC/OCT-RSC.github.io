"""Check the static website and published assets using the Python standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DOWNLOADS = ROOT / "downloads"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            require(attrs["id"] not in self.ids, f"Duplicate HTML ID: {attrs['id']}")
            self.ids.add(attrs["id"])
        for name in ("href", "src", "poster"):
            if name in attrs:
                self.links.append(attrs[name])


def site_file(url):
    path = (ROOT / unquote(urlsplit(url).path)).resolve()
    require(path.is_relative_to(ROOT), f"Asset outside website: {url}")
    require(path.is_file(), f"Missing asset: {url}")
    return path


def check_page():
    page = Page()
    text = (ROOT / "index.html").read_text(encoding="utf-8")
    page.feed(text)
    for link in page.links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            continue
        if parsed.path:
            site_file(link)
        else:
            require(parsed.fragment in page.ids, f"Missing section: {link}")
    require("volume-viewer.js" in text, "Interactive viewer is not loaded")
    require((ROOT / ".nojekyll").is_file(), "Missing GitHub Pages .nojekyll file")
    return len(page.links)


def check_downloads(write_checksums):
    checksum_file = DOWNLOADS / "checksums.txt"
    files = sorted(path for path in DOWNLOADS.iterdir()
                   if path.is_file() and path != checksum_file)
    checksums = "".join(f"{digest(path)}  {path.name}\n" for path in files)
    if write_checksums:
        checksum_file.write_text(checksums, encoding="utf-8")
    require(checksum_file.read_text(encoding="utf-8") == checksums,
            "Download checksums changed; review replacements before regenerating")
    return len(files)


def check_videos():
    demos = json.loads((ASSETS / "demos.json").read_text())
    media = json.loads((ASSETS / "media-verification.json").read_text())
    require(set(demos) == {"depth", "tilt", "exvivo"}, "Incorrect experiment tabs")
    require(media["all_targets"] and media["case_count"] == 26,
            "Incorrect demonstration coverage")
    require(media["recorded_OCT_states"] == 338 and media["same_state_RGB"] == 312,
            "Incorrect recorded-state counts")
    require(media["steps"] == list(range(13)), "Missing recorded OCT steps")
    conditions = {
        "depth": [2, 2.5, 3, 3.5, 4, 4.5, 5],
        "tilt": [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180],
        "exvivo": [2, 2.5, 3, 3.5, 4, 0, 22.5, 45, 67.5, 90],
    }
    for key, values in conditions.items():
        demo = demos[key]
        require(demo["target_values"] == values == media["target_conditions"][key],
                f"Incomplete target coverage: {key}")
        require(demo["case_count"] == len(values), f"Incorrect case count: {key}")
        require(demo["recorded_steps"] == list(range(13)), f"Missing steps: {key}")
        require(demo["resolution"] == [3840, 2160], f"Incorrect export resolution: {key}")
        require(digest(site_file(demo["video"])) == media["video_sha256"][key],
                f"Video checksum changed: {key}")
        require(site_file(demo["poster"]).stat().st_size > 1000,
                f"Missing video poster: {key}")


def check_viewer():
    style = json.loads((ASSETS / "paper-render-style.json").read_text())
    require(style["shape_dhw"] == [256, 256, 256], "Unexpected display volume shape")
    require(style["depth_crop"] is None, "Display volume is unexpectedly cropped")
    require(style["shared_parallel_scale"] == 6.8, "Shared camera scale changed")
    require(style["display_only_mask"] is True, "Display-mask scope is not recorded")
    require(style["mask_voxels"] == 618637, "Display-mask voxel count changed")
    voxel_count = math.prod(style["shape_dhw"])
    for name in ("paper-display-volume-f16.bin", "paper-display-tissue-f16.bin"):
        data = (ASSETS / name).read_bytes()
        require(len(data) == voxel_count * 2, f"Unexpected float16 array size: {name}")
        require(all(math.isfinite(value) for (value,) in struct.iter_unpack("<e", data)),
                f"Nonfinite intensity: {name}")
    mask = (ASSETS / "paper-display-mask.bin").read_bytes()
    require(len(mask) == voxel_count and set(mask) == {0, 1}, "Invalid display mask")
    require(sum(mask) == style["mask_voxels"], "Display-mask occupancy changed")
    lut = (ASSETS / "paper-transfer-lut.bin").read_bytes()
    require(len(lut) == 4096 * 2 * 4 * 4, "Incorrect transfer-function size")
    require(all(math.isfinite(value) and 0 <= value <= 1
                for (value,) in struct.iter_unpack("<f", lut)), "Invalid transfer function")
    points = json.loads((ASSETS / "point-cloud.json").read_text())
    require(len(points["positions"]) == 1024 * 3, "Point-cloud sample count changed")
    require(all(math.isfinite(value) and -1.001 <= value <= 1.001
                for value in points["positions"]), "Invalid normalized point coordinates")
    for name, expected in style["asset_sha256"].items():
        require(digest(ASSETS / name) == expected, f"Viewer asset checksum changed: {name}")
    logos = json.loads((ASSETS / "logo-sources.json").read_text())
    for logo in logos.values():
        path = ASSETS / logo["file"]
        require(ET.parse(path).getroot().tag.endswith("svg"), f"Invalid logo: {path.name}")
        require(digest(path) == logo["sha256"], f"Logo checksum changed: {path.name}")


def check_published_content():
    # Report filenames only if a scan fails; never print a matched credential.
    patterns = {
        "private key": re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
        "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
        "local data path": re.compile(rb"(?:/" + rb"Users/[^\s\"']+|/" + rb"gpfs/accounts/[^\s\"']+|/" + rb"nfs/turbo/[^\s\"']+|[SH]:\\\\?chaoyz)", re.I),
    }

    def scan(name, data):
        for label, pattern in patterns.items():
            require(not pattern.search(data), f"Review {label} in published file: {name}")

    files = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in {".git", "__pycache__", "node_modules"} for part in relative.parts):
            continue
        if path.is_dir():
            continue
        require(not path.is_symlink(), f"Unexpected published symlink: {relative}")
        require(path.stat().st_size < 100 * 1024 * 1024,
                f"File exceeds normal GitHub upload limit: {relative}")
        require(path.name not in {".env", "id_rsa", "id_ed25519"}
                and path.suffix not in {".log", ".pem", ".key"},
                f"Unexpected private/runtime file: {relative}")
        files.append(path)
        if path.suffix in {".html", ".js", ".css", ".json", ".svg", ".md", ".txt", ".py"}:
            scan(str(relative), path.read_bytes())
        if path.name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as archive:
                for member in archive.getmembers():
                    require(not member.name.startswith("/") and ".." not in Path(member.name).parts,
                            f"Unsafe archive entry in {relative}")
                    require(not member.issym() and not member.islnk(),
                            f"Unexpected archive link in {relative}")
                    if member.isfile() and not member.name.endswith(".npy"):
                        scan(f"{relative}:{member.name}", archive.extractfile(member).read())
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                require(archive.testzip() is None, f"Invalid wheel archive: {relative}")
                for name in archive.namelist():
                    scan(f"{relative}:{name}", archive.read(name))
    return {"published_files": len(files), "published_bytes": sum(p.stat().st_size for p in files),
            "largest_file_bytes": max(p.stat().st_size for p in files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-checksums", action="store_true")
    args = parser.parse_args()
    downloads = check_downloads(args.write_checksums)
    links = check_page()
    check_videos()
    check_viewer()
    inventory = check_published_content()
    print(json.dumps({"status": "passed", "local_links": links, "downloads": downloads,
                      "video_cases": 26, "OCT_states": 338, "paired_RGB": 312,
                      "point_cloud_samples": 1024, "display_mask_voxels": 618637,
                      **inventory}, indent=2))


if __name__ == "__main__":
    main()
