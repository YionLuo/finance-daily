#!/usr/bin/env python3
"""
Y Daily Health Check & Auto-Recovery

Checks if Breaking News has been updated recently.
If stale (>2h since last auto commit), runs update_breaking.py locally
and pushes the result.

Designed to run hourly via WorkBuddy automation as a failover
when GitHub Actions is down.
"""

import os
import sys
import subprocess
import time
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from utils import FALLBACK_ENDPOINTS

# ============ Configuration ============

SITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STALE_THRESHOLD_MINUTES = 120  # 2 hours
MAX_RETRY = 2

# NovAI config (fallback if env not set)
DEFAULT_BASE_URL = "https://once.novai.su/v1"
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def run(cmd, cwd=None):
    """Run a shell command, return (success, stdout)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=300, cwd=cwd or SITE_DIR
        )
        return r.returncode == 0, r.stdout.strip() + "\n" + r.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def git_pull():
    """Pull latest from remote."""
    ok, out = run("git pull --rebase origin main")
    if not ok:
        # If rebase conflicts, abort and try reset
        run("git rebase --abort")
        run("git stash")
        ok, out = run("git pull --rebase origin main")
        if ok:
            run("git stash pop")
    return ok


def get_last_auto_commit_age():
    """
    Get the age (in minutes) of the most recent 'auto:' commit.
    Returns None if no auto commit found.
    """
    ok, out = run(
        'git log --format="%ct %s" --since="48 hours ago" -50'
    )
    if not ok:
        return None

    for line in out.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue
        try:
            ts = int(parts[0])
            subject = parts[1]
        except (ValueError, IndexError):
            continue

        if subject.startswith("auto:"):
            age_sec = time.time() - ts
            return age_sec / 60.0

    return None


def test_api():
    """Quick test if any LLM API endpoint is responsive."""
    import urllib.request
    import json

    api_key = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    if not api_key:
        log("No API key available")
        return False

    # Build endpoint list: env var first, then fallbacks
    endpoints = []
    env_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    if env_url:
        endpoints.append(env_url)
    for ep in FALLBACK_ENDPOINTS:
        if ep not in endpoints:
            endpoints.append(ep)

    for base_url in endpoints:
        url = f"{base_url}/chat/completions"
        data = json.dumps({
            "model": "gpt-5.1-low",
            "messages": [{"role": "user", "content": "say ok"}],
            "max_tokens": 5
        }).encode()

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    log(f"API test OK: {base_url}")
                    # Update env for subsequent scripts to use the working endpoint
                    os.environ["OPENAI_BASE_URL"] = base_url
                    return True
        except Exception as e:
            log(f"API test failed [{base_url}]: {e}")
            continue

    log("All API endpoints failed")
    return False


def run_update():
    """Run update_breaking.py and return success."""
    env_extra = ""
    if not os.environ.get("OPENAI_BASE_URL"):
        env_extra += f'OPENAI_BASE_URL="{DEFAULT_BASE_URL}" '
    if not os.environ.get("OPENAI_API_KEY") and DEFAULT_API_KEY:
        env_extra += f'OPENAI_API_KEY="{DEFAULT_API_KEY}" '

    cmd = f"{env_extra}python3 scripts/update_breaking.py"
    log(f"Running update script...")
    ok, out = run(cmd)

    # Check if any new items were actually produced
    if ok and "Done!" in out:
        log("Update script completed successfully")
        # Print summary lines
        for line in out.split("\n"):
            if any(k in line for k in ["New unique", "Updated time", "Error"]):
                log(f"  {line.strip()}")
        return True
    else:
        log(f"Update script failed")
        for line in out.split("\n")[-5:]:
            if line.strip():
                log(f"  {line.strip()}")
        return False


def git_push():
    """Commit and push changes."""
    # Check if there are changes
    ok, out = run("git diff --stat index.html")
    if not ok or not out.strip():
        log("No changes to push")
        return True

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M CST")
    commit_msg = f"auto: update breaking news {now_str}"

    ok1, _ = run("git add index.html")
    ok2, _ = run(f'git commit -m "{commit_msg}"')
    if not ok2:
        log("Commit failed")
        return False

    for attempt in range(MAX_RETRY):
        ok, out = run("git push origin main")
        if ok:
            log("Push successful")
            return True
        # If rejected, pull and retry
        log(f"Push rejected (attempt {attempt+1}), pulling...")
        run("git pull --rebase origin main")

    log("Push failed after retries")
    return False


def main():
    log("=" * 50)
    log("Y Daily Health Check")
    log("=" * 50)

    # Step 1: Pull latest
    log("Pulling latest...")
    if not git_pull():
        log("ERROR: git pull failed, aborting")
        return 1

    # Step 2: Check last auto commit age
    age = get_last_auto_commit_age()
    if age is None:
        log("WARNING: No auto commits found in last 48h")
        age = 9999  # Force update
    else:
        log(f"Last auto commit: {age:.0f} minutes ago")

    if age < STALE_THRESHOLD_MINUTES:
        log(f"OK — within {STALE_THRESHOLD_MINUTES}min threshold, no action needed")
        return 0

    # Step 3: Stale! Test API first
    log(f"STALE — {age:.0f}min since last update (threshold: {STALE_THRESHOLD_MINUTES}min)")
    log("Testing API availability...")

    if not test_api():
        log("API is down — cannot recover, will retry next cycle")
        return 1

    log("API is up — running recovery update")

    # Step 4: Run update
    if not run_update():
        log("Update failed — will retry next cycle")
        return 1

    # Step 5: Push
    if not git_push():
        log("Push failed — changes are local only")
        return 1

    log("Recovery complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
