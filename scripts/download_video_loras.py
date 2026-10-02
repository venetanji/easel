"""Install pinned, checksum-verified safetensors from the curated video catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from easel.video_loras import VIDEO_LORAS


class HFRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        redirected = super().redirect_request(request, response, code, message, headers, newurl)
        if redirected is not None and urlsplit(newurl).hostname != urlsplit(request.full_url).hostname:
            redirected.remove_header("Authorization")
        if urlsplit(newurl).scheme != "https":
            raise urllib.error.URLError("refusing a non-HTTPS download redirect")
        return redirected


def open_download(request, timeout):
    return urllib.request.build_opener(HFRedirectHandler()).open(request, timeout=timeout)


def token_from_env_file(filename: Path) -> str | None:
    for line in filename.read_text().splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and name.strip() in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
            return value.strip().strip("\"'") or None
    return None


def checksum(filename: Path) -> str:
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_asset(entry: dict, asset: dict, destination: Path, token: str | None) -> dict:
    filename = asset["filename"]
    if Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise ValueError("unsafe catalog filename")
    target = destination / filename
    if target.exists():
        if target.stat().st_size != asset["bytes"] or checksum(target) != asset["sha256"]:
            raise ValueError(f"existing file does not match the pinned asset: {filename}; not overwritten")
        return {"id": entry["id"], "filename": filename, "status": "already_verified"}
    url = f"https://huggingface.co/{entry['repo_id']}/resolve/{entry['revision']}/{filename}"
    headers = {"Authorization": "Bearer " + token} if token else {}
    request = urllib.request.Request(url, headers=headers)
    temporary = None
    try:
        with open_download(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(dir=destination, prefix="." + filename + ".",
                                             suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                digest = hashlib.sha256()
                size = 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    digest.update(block)
                    size += len(block)
        if size != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
            raise ValueError(f"download checksum/size mismatch: {filename}")
        if target.exists():
            raise ValueError(f"destination appeared during download: {filename}; not overwritten")
        temporary.chmod(0o644)
        temporary.replace(target)
        temporary = None
        return {"id": entry["id"], "filename": filename, "status": "installed",
                "bytes": size, "sha256": digest.hexdigest()}
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            return {"id": entry["id"], "filename": filename, "status": "access_denied",
                    "http_status": error.code,
                    "action": "Use a valid HF token and approve access to " + entry["repo_id"]}
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--collection", default="ltx-25-creative-lab")
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--token-env-file", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.ids:
        unknown = set(args.ids) - VIDEO_LORAS.keys()
        if unknown:
            parser.error("unknown ids: " + ", ".join(sorted(unknown)))
        entries = [VIDEO_LORAS[model_id] for model_id in args.ids]
    else:
        entries = [entry for entry in VIDEO_LORAS.values() if entry["collection"] == args.collection]
        if not entries:
            parser.error("unknown or empty collection")
    token = token_from_env_file(args.token_env_file) if args.token_env_file else \
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    args.destination.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for entry in entries:
        for asset in entry["files"]:
            try:
                result = install_asset(entry, asset, args.destination, token)
            except (OSError, ValueError, urllib.error.URLError) as error:
                result = {"id": entry["id"], "filename": asset["filename"],
                          "status": "failed", "error": str(error)}
            outcomes.append(result)
            print(json.dumps(result), flush=True)
    if args.report:
        args.report.write_text(json.dumps(outcomes, indent=2) + "\n")
    return int(any(result["status"] not in ("installed", "already_verified") for result in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
