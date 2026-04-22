"""Scrape Go source files from MIT-licensed GitHub repos for AST pretraining.

Usage:
    export GITHUB_TOKEN=ghp_...   # optional but strongly recommended
    python scrape.py --max-pages 10 --output-dir scraped_code

Writes files to <output-dir>/<owner>__<repo>/<path/in/repo>. Re-runs skip repos
already listed in <output-dir>/.seen-repos so incremental scrapes are safe.
"""
import argparse
import asyncio
import os
import random
import time
from typing import Callable, Optional
from urllib.parse import quote_plus

import aiohttp


DEFAULT_QUERY = "language:Go license:mit"
DEFAULT_OUTPUT = "scraped_code"
DEFAULT_MAX_PAGES = 10
DEFAULT_MAX_FILE_BYTES = 500 * 1024
ALLOWED_EXTENSIONS = {"go"}
SEEN_REPOS_FILE = ".seen-repos"

request_queue: list = []
seen_repos: set = set()
config: dict = {}


def auth_headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "gopanic-zero-scraper",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def _request(session: aiohttp.ClientSession, url: str, use_json: bool):
    """GET with GitHub rate-limit awareness. Returns (body, headers)."""
    while True:
        async with session.get(url) as response:
            if response.status in (403, 429):
                remaining = response.headers.get("X-RateLimit-Remaining")
                reset = response.headers.get("X-RateLimit-Reset")
                if remaining == "0" and reset:
                    wait = max(0, int(reset) - int(time.time())) + random.uniform(1, 5)
                    print(f"rate-limited; sleeping {wait:.1f}s until reset", flush=True)
                    await asyncio.sleep(wait)
                    continue
                body = await response.text()
                raise RuntimeError(f"GitHub {response.status}: {body[:200]}")
            response.raise_for_status()
            headers = dict(response.headers)
            if use_json:
                return await response.json(), headers
            return await response.text(), headers


def add_request(
    url: str,
    callback: Callable,
    error_callback: Optional[Callable] = None,
    use_json: bool = True,
):
    request_queue.append({
        "url": url,
        "callback": callback,
        "error_callback": error_callback,
        "use_json": use_json,
    })


async def fetch_data_callback(session, url, callback, error, use_json=True):
    try:
        body, _ = await _request(session, url, use_json=use_json)
        callback(body)
    except Exception as e:
        (error or (lambda ex: print(f"error {url}: {ex}", flush=True)))(e)


def save_file(save_path: str, contents: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(contents)


def make_download_handler(save_path: str):
    def handle(result):
        save_file(save_path, result)
        print(f"saved {save_path}", flush=True)
    return handle


def parse_contents_res(result, repo_slug: str):
    items = result if isinstance(result, list) else [result]
    for entry in items:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("type")
        if etype == "dir":
            add_request(
                entry["url"],
                lambda r, rs=repo_slug: parse_contents_res(r, rs),
            )
            continue
        if etype != "file":
            continue
        name = entry.get("name", "")
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if ext not in ALLOWED_EXTENSIONS:
            continue
        size = entry.get("size", 0)
        if size and size > config["max_file_bytes"]:
            print(f"skip (size {size} > cap): {entry.get('path')}", flush=True)
            continue
        download_url = entry.get("download_url")
        if not download_url:
            continue
        rel_path = entry.get("path", name)
        save_path = os.path.join(config["output_dir"], repo_slug, rel_path)
        add_request(download_url, make_download_handler(save_path), use_json=False)


def append_seen_repo(full_name: str):
    path = os.path.join(config["output_dir"], SEEN_REPOS_FILE)
    with open(path, "a") as f:
        f.write(full_name + "\n")


def load_seen_repos():
    path = os.path.join(config["output_dir"], SEEN_REPOS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    seen_repos.add(line)
    print(f"loaded {len(seen_repos)} previously scraped repos", flush=True)


def parse_repo_list(result, page: int):
    items = result.get("items", [])
    print(f"page {page}: {len(items)} repos", flush=True)
    for item in items:
        full = item.get("full_name", "")
        if not full:
            continue
        if full in seen_repos:
            print(f"skip: already scraped {full}", flush=True)
            continue
        seen_repos.add(full)
        append_seen_repo(full)
        slug = full.replace("/", "__")
        contents_url = item["contents_url"].removesuffix("{+path}")
        add_request(
            contents_url,
            lambda r, rs=slug: parse_contents_res(r, rs),
        )


def _next_link(link_header: str) -> Optional[str]:
    if not link_header:
        return None
    for part in link_header.split(","):
        chunk = part.strip()
        if not chunk:
            continue
        sections = chunk.split(";")
        target = sections[0].strip()
        rel = None
        for s in sections[1:]:
            s = s.strip()
            if s.startswith("rel="):
                rel = s[4:].strip('"')
        if rel == "next" and target.startswith("<") and target.endswith(">"):
            return target[1:-1]
    return None


async def seed_queue_with_search(session, query: str, max_pages: int):
    url = (
        "https://api.github.com/search/repositories"
        f"?q={quote_plus(query)}&per_page=30&page=1"
    )
    page = 0
    while url and page < max_pages:
        page += 1
        body, headers = await _request(session, url, use_json=True)
        parse_repo_list(body, page)
        url = _next_link(headers.get("Link", ""))


async def drain_queue(session):
    while request_queue:
        batch = []
        while request_queue:
            t = request_queue.pop()
            batch.append(
                fetch_data_callback(
                    session,
                    t["url"],
                    t["callback"],
                    t["error_callback"],
                    use_json=t.get("use_json", True),
                )
            )
        await asyncio.gather(*batch)


async def main_async(args):
    config["output_dir"] = args.output_dir
    config["max_file_bytes"] = args.max_file_bytes
    os.makedirs(args.output_dir, exist_ok=True)
    load_seen_repos()

    if not os.environ.get("GITHUB_TOKEN"):
        print("warning: GITHUB_TOKEN not set; 60 req/hr limit applies", flush=True)

    async with aiohttp.ClientSession(headers=auth_headers()) as session:
        body, _ = await _request(session, "https://api.github.com/rate_limit", use_json=True)
        core = body.get("resources", {}).get("core", {})
        search = body.get("resources", {}).get("search", {})
        print(
            f"rate-limit: core {core.get('remaining')}/{core.get('limit')}, "
            f"search {search.get('remaining')}/{search.get('limit')}",
            flush=True,
        )

        await seed_queue_with_search(session, args.query, args.max_pages)
        await drain_queue(session)


def parse_args():
    p = argparse.ArgumentParser(
        description="Scrape Go source files from MIT-licensed GitHub repos."
    )
    p.add_argument("--query", default=DEFAULT_QUERY,
                   help="GitHub search query (default: %(default)s)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                   help="Output directory (default: %(default)s)")
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES,
                   help="Max search pages @ 30 repos/page (default: %(default)s)")
    p.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES,
                   help="Skip files larger than N bytes (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
