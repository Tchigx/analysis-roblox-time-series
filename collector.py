"""Sample public Roblox experience metrics and write them to Supabase.

Two stages:
  1. Discovery -- walk the Discover-page sorts to collect universe IDs.
  2. Detail    -- batch-fetch metrics for those IDs and store a snapshot.

The old games.roblox.com/v1/games/list + /v1/games/sorts endpoints are
deprecated; discovery now goes through apis.roblox.com/explore-api/v1.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from supabase import create_client

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

EXPLORE_BASE = "https://apis.roblox.com/explore-api/v1"
GAMES_BASE = "https://games.roblox.com/v1"

# Conservative starting batch size. 100 returns 400 in practice; the real cap
# is lower and undocumented. Failed batches are bisected (see fetch_batch), so
# this is a starting guess rather than a hard requirement.
DETAIL_CHUNK_SIZE = 50

# Each sort returns roughly 100 experiences, so the number of sorts you walk
# sets your ceiling. Walking all of them and de-duplicating typically lands
# somewhere in the 300-800 unique games range.
MAX_SORTS = 25

REQUEST_TIMEOUT = 15  # seconds -- without this a hung socket stalls the runner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("collector")


# --------------------------------------------------------------------------
# HTTP session with retries
# --------------------------------------------------------------------------

def build_session() -> requests.Session:
    """Session that retries transient failures with exponential backoff.

    429 matters most here: Roblox rate-limits by IP, and CI runners share
    datacenter address space that tends to be throttled harder than home
    connections. respect_retry_after_header honours the server's own backoff
    hint when it sends one.
    """
    retry = Retry(
        total=4,
        backoff_factor=1.5,          # 0s, 1.5s, 3s, 6s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "roblox-ccu-collector/1.0"})
    return session


# --------------------------------------------------------------------------
# Stage 1 -- discovery
# --------------------------------------------------------------------------

def harvest_universe_ids(payload) -> set[int]:
    """Recursively pull every universeId out of a JSON payload.

    Deliberately shape-agnostic. The explore API's response structure has
    changed before and is undocumented; walking for the key is far more
    durable than indexing into payload["sorts"][i]["games"][j].
    """
    found: set[int] = set()

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "universeId" and isinstance(value, int):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def discover(session: requests.Session) -> set[int]:
    """Collect universe IDs from the Discover-page sorts."""
    # sessionId groups related requests server-side. Any UUID works; generate
    # a fresh one per run.
    session_id = str(uuid.uuid4())
    universe_ids: set[int] = set()

    try:
        resp = session.get(
            f"{EXPLORE_BASE}/get-sorts",
            params={"sessionId": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        sorts_payload = resp.json()
    except Exception as exc:
        log.error("get-sorts failed: %s", exc)
        return universe_ids

    # get-sorts often embeds the first page of each sort -- take those for free.
    universe_ids |= harvest_universe_ids(sorts_payload)

    sort_ids = [
        s.get("sortId")
        for s in sorts_payload.get("sorts", [])
        if isinstance(s, dict) and s.get("sortId")
    ][:MAX_SORTS]
    log.info("discovered %d sorts", len(sort_ids))

    for sort_id in sort_ids:
        try:
            resp = session.get(
                f"{EXPLORE_BASE}/get-sort-content",
                params={"sessionId": session_id, "sortId": sort_id},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            universe_ids |= harvest_universe_ids(resp.json())
        except Exception as exc:
            # One bad sort shouldn't sink the run.
            log.warning("sort %s failed: %s", sort_id, exc)

    log.info("discovery found %d unique universe ids", len(universe_ids))
    return universe_ids


# --------------------------------------------------------------------------
# Stage 2 -- detail
# --------------------------------------------------------------------------

def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class BadBatch(Exception):
    """The server rejected this specific set of IDs (HTTP 400)."""


def request_batch(session: requests.Session, ids: list[int]) -> list[dict]:
    """One detail request.

    The URL is built by hand rather than via params= because requests
    percent-encodes the separators into %2C, and some Roblox endpoints reject
    encoded commas.
    """
    url = f"{GAMES_BASE}/games?universeIds=" + ",".join(map(str, ids))
    resp = session.get(url, timeout=REQUEST_TIMEOUT)

    if resp.status_code == 400:
        # Roblox puts a real explanation in the body -- surface it.
        raise BadBatch(resp.text[:300])

    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_batch(session: requests.Session, ids: list[int], depth: int = 0) -> list[dict]:
    """Fetch a batch, halving it on rejection until the cause is isolated.

    A 400 has two causes that look identical from outside: the batch is over
    the server's size limit, or it contains a universe ID the API won't serve
    (deleted, private, malformed). Bisection resolves both -- an oversized
    batch keeps halving until it fits, and a poisonous ID keeps halving until
    it's alone and gets dropped. Without this, one dead ID costs you every
    other game in its batch.
    """
    try:
        return request_batch(session, ids)
    except BadBatch as exc:
        if len(ids) == 1:
            log.warning("dropping universe id %s: %s", ids[0], exc)
            return []
        if depth == 0:
            log.info("batch of %d rejected, bisecting", len(ids))
        mid = len(ids) // 2
        return (fetch_batch(session, ids[:mid], depth + 1)
                + fetch_batch(session, ids[mid:], depth + 1))
    except Exception as exc:
        log.warning("batch starting %s failed: %s", ids[0], exc)
        return []


def fetch_details(session: requests.Session, universe_ids: set[int]) -> list[dict]:
    """Batch-fetch full game records."""
    records: list[dict] = []

    for chunk in chunked(sorted(universe_ids), DETAIL_CHUNK_SIZE):
        records.extend(fetch_batch(session, chunk))

    log.info("fetched details for %d of %d games", len(records), len(universe_ids))
    return records


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def to_rows(records: list[dict], sampled_at: str):
    """Split API records into dimension rows and fact rows."""
    games, metrics = [], []

    for game in records:
        universe_id = game.get("id")
        if universe_id is None:
            continue

        creator = game.get("creator") or {}
        games.append({
            "universe_id": universe_id,
            "root_place_id": game.get("rootPlaceId"),
            "name": game.get("name"),
            "creator_name": creator.get("name"),
            "creator_type": creator.get("type"),
            "genre": game.get("genre"),
            "genre_l1": game.get("genre_l1"),
            "genre_l2": game.get("genre_l2"),
            "max_players": game.get("maxPlayers"),
            "price": game.get("price"),
            "created_at": game.get("created"),
            "updated_at": game.get("updated"),
            "last_seen": sampled_at,
        })
        metrics.append({
            "universe_id": universe_id,
            "sampled_at": sampled_at,
            # Explicit None rather than 0 on a missing field: a real zero and
            # an absent value are different things, and collapsing them would
            # quietly corrupt the time series.
            "ccu": game.get("playing"),
            "visits": game.get("visits"),
            "favorites": game.get("favoritedCount"),
        })

    return games, metrics


def persist(supabase, games: list[dict], metrics: list[dict]) -> int:
    """Upsert dimensions first (FK parent), then append the snapshot."""
    for chunk in chunked(games, 500):
        supabase.table("games").upsert(chunk, on_conflict="universe_id").execute()

    written = 0
    for chunk in chunked(metrics, 500):
        supabase.table("game_metrics").insert(chunk).execute()
        written += len(chunk)

    return written


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> int:
    started_at = datetime.now(timezone.utc)
    sampled_at = started_at.isoformat()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        log.error("SUPABASE_URL and SUPABASE_KEY must be set")
        return 1

    supabase = create_client(url, key)
    session = build_session()

    status, error, games_seen, rows_written = "ok", None, 0, 0

    try:
        universe_ids = discover(session)
        if not universe_ids:
            raise RuntimeError("discovery returned no universe ids")

        records = fetch_details(session, universe_ids)
        games_seen = len(records)
        if not records:
            raise RuntimeError("detail fetch returned no records")

        games, metrics = to_rows(records, sampled_at)
        rows_written = persist(supabase, games, metrics)

        # Fewer details than IDs means some chunks failed. Still a useful
        # sample, but flag it so you can filter these runs out later.
        if games_seen < len(universe_ids) * 0.9:
            status = "partial"

        log.info("wrote %d metric rows (status=%s)", rows_written, status)

    except Exception as exc:
        status, error = "failed", str(exc)[:500]
        log.exception("collector run failed")

    finally:
        try:
            supabase.table("collection_runs").insert({
                "started_at": started_at.isoformat(),
                "games_seen": games_seen,
                "rows_written": rows_written,
                "status": status,
                "error": error,
            }).execute()
        except Exception:
            log.exception("could not write run log")

    return 0 if status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
