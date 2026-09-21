"""Run one job end to end against a local stack, and report what came back.

Exists because the obvious way to do this by hand does not work. Sessions are cookie
based and a job belongs to the session that created it, so submitting with one `curl`
and polling with another returns 404 — which reads as a missing job when it is really a
missing cookie. One process, one jar.

Usage::

    uv run python infra/scripts/run_job.py
    uv run python infra/scripts/run_job.py --prompt "..." --voice "Morgan Freeman"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from http.cookiejar import CookieJar
from typing import Any

TERMINAL = {"done", "failed", "cancelled"}


def say(line: str) -> None:
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument(
        "--prompt",
        default="A lighthouse keeper finds the lamp cold on a night with no moon.",
    )
    parser.add_argument("--voice", default="Morgan Freeman")
    parser.add_argument("--mode", default="narration_with_dialogue")
    parser.add_argument("--length", default="short")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def get(path: str) -> Any:
        with opener.open(f"{args.api}{path}", timeout=30) as response:
            return json.load(response)

    voices = get("/v1/voices")["items"]
    if not voices:
        say("no voices seeded")
        return 1

    narrator = next((v for v in voices if args.voice.lower() in v["name"].lower()), voices[0])
    other = next((v for v in voices if v["id"] != narrator["id"]), narrator)
    third = next(
        (v for v in voices if v["id"] not in {narrator["id"], other["id"]}),
        None,
    )
    say(f"narrator   {narrator['name']} ({narrator['duration_seconds']:.0f}s reference)")
    if args.mode == "narration_with_dialogue":
        say(f"character  {other['name']}")
        if third:
            say(f"character  {third['name']}")

    payload = json.dumps(
        {
            "prompt": args.prompt,
            "length": args.length,
            "mode": args.mode,
            "emotion": "neutral",
            "language": "en",
            "speed": 1.0,
            "voice_id": narrator["id"],
            "dialogue_voice_id": other["id"] if args.mode == "narration_with_dialogue" else None,
            "second_dialogue_voice_id": (
                third["id"] if args.mode == "narration_with_dialogue" and third else None
            ),
        }
    ).encode()

    request = urllib.request.Request(  # noqa: S310 - the API base is a CLI argument
        f"{args.api}/v1/jobs",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    with opener.open(request, timeout=30) as response:
        job_id = json.load(response)["id"]
    say(f"job    {job_id}")
    say("")

    previous: tuple[str, int | None] | None = None
    job: dict[str, Any] = {}

    while time.monotonic() - started < args.timeout:
        job = get(f"/v1/jobs/{job_id}")
        state = (job["status"], job.get("segment_count"))
        if state != previous:
            say(f"{time.monotonic() - started:7.1f}s  {job['status']:<14} segments={state[1]}")
            previous = state
        if job["status"] in TERMINAL:
            break
        time.sleep(5)

    say("")
    say(f"final  {job.get('status')}")
    if job.get("error"):
        say(f"error  {job['error']}")
        return 1

    audio = job.get("audio") or []
    if audio:
        say(f"audio  {audio[0]['duration_seconds']:.1f}s")

    segments = job.get("segments") or []
    say(f"timeline  {len(segments)} entries")
    for segment in segments[:5]:
        say(
            f"  [{segment['index']:>2}] {segment['kind']:<9} "
            f"{(segment.get('speaker') or '-'):<8} "
            f"{segment['start_seconds']:7.2f}-{segment['end_seconds']:7.2f}s  "
            f"chars {segment['start_char']}-{segment['end_char']}  "
            f"{segment['text'][:48]!r}"
        )
    if len(segments) > 5:
        say(f"  ... {len(segments) - 5} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
