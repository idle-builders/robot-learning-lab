"""Warm pinned Arena USD dependencies concurrently on a disposable GPU worker."""

from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import urlopen
import os
import sys
import tempfile

from isaaclab.utils import assets


ROOT_URL = (
    "https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/"
    "Isaac/IsaacLab/Arena/assets/background_library/galileo_locomanip/galileo_locomanip.usd"
)


def fetch(url):
    target = assets._mirror_path(url, tempfile.gettempdir())
    if assets._usable_mirror(url):
        return url, target, None
    os.makedirs(os.path.dirname(target), exist_ok=True)
    for attempt in range(2):
        partial = target + ".prefetch"
        try:
            with urlopen(url, timeout=90) as source, open(partial, "wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            os.replace(partial, target)
            assets._write_mirror_fingerprint(url, target)
            return url, target, None
        except Exception as exc:
            try:
                os.remove(partial)
            except OSError:
                pass
            if isinstance(exc, HTTPError) and exc.code == 404:
                return url, None, "404"
            if attempt == 1:
                return url, None, repr(exc)


def main():
    root_url = sys.argv[1] if len(sys.argv) > 1 else ROOT_URL
    _, root_path, error = fetch(root_url)
    if error:
        raise RuntimeError(f"Unable to fetch root asset: {error}")
    queue = [assets._resolve_reference_url(root_url, ref) for ref in assets._find_asset_dependencies(root_path)]
    seen = {root_url}
    level = 0
    while queue:
        batch = [url for url in dict.fromkeys(queue) if url.startswith("https://") and url not in seen]
        seen.update(batch)
        queue = []
        if not batch:
            break
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(fetch, batch))
        good = 0
        for url, path, error in results:
            if error:
                print("SKIP", error, url, flush=True)
                continue
            good += 1
            for ref in assets._find_asset_dependencies(path):
                queue.append(assets._resolve_reference_url(url, ref))
        print("LEVEL", level, "downloaded_or_cached", good, "seen", len(seen), "next", len(queue), flush=True)
        level += 1


if __name__ == "__main__":
    main()
