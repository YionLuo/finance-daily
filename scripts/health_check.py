#!/usr/bin/env python3
"""
Y Daily Health Check & Auto-Recovery v2

Checks THREE content streams:
1. Breaking News — should have auto commit within 2 hours
2. Finance Daily Report — should have today's issue (after 22:00 CST)
3. AI Daily Report — should have today's issue (after 22:10 CST)

If any stream is stale, attempts auto-recovery by running the
corresponding update script locally and pushing the result.

Designed to run hourly via WorkBuddy automation as a failover
when GitHub Actions is down.
"""

import os
import sys
import subprocess
import time
import re
import json
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from utils import FALLBACK_ENDPOINTS

# ============ Configuration ============

SITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CST = timezone(timedelta(hours=8))

# Thresholds
BREAKING_STALE_MINUTES = 120      # Breaking News: 2 hours without commit
DAILY_REPORT_HOUR = 22            # Finance daily runs at 22:00 CST
AI_REPORT_HOUR = 22               # AI daily runs at 22:10 CST
DAILY_GRACE_MINUTES = 60          # Allow 60 min grace after scheduled time

MAX_RETRY = 2

# NovAI config (fallback if env not set)
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def run(cmd, cwd=None, timeout_sec=300):
    """Run a shell command, return (success, stdout)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout_sec, cwd=cwd or SITE_DIR
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
        run("git rebase --abort")
        run("git stash")
        ok, out = run("git pull --rebase origin main")
        if ok:
            run("git stash pop")
    return ok


# ============ Check Functions ============

def get_last_auto_commit_age():
    """Get age (minutes) of most recent 'auto:' commit."""
    ok, out = run('git log --format="%ct %s" --since="48 hours ago" -50')
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


def check_daily_report_in_html(report_type="finance"):
    """
    Check if today's daily report exists in index.html.

    Args:
        report_type: "finance" or "ai"

    Returns:
        (has_today, latest_date_str) — whether today's report exists, and the latest date found
    """
    try:
        html_path = os.path.join(SITE_DIR, "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        now = datetime.now(CST)
        today_str = now.strftime("%Y-%m-%d")

        if report_type == "finance":
            # Look for issue id like "2026-04-12" in const issues = [...]
            pattern = r'"id"\s*:\s*"(\d{4}-\d{2}-\d{2})"'
            var_pattern = r'const\s+issues\s*='
        else:
            # Look for AI issue id like "ai-2026-04-12" in const aiIssues = [...]
            pattern = r'"id"\s*:\s*"ai-(\d{4}-\d{2}-\d{2})"'
            var_pattern = r'const\s+aiIssues\s*='

        # Find the relevant section
        var_match = re.search(var_pattern, html)
        if not var_match:
            return False, "N/A"

        # Search for dates after the variable declaration
        section = html[var_match.start():var_match.start() + 5000]
        dates = re.findall(pattern, section)

        if not dates:
            return False, "N/A"

        latest_date = dates[0]  # First one should be the latest (prepended)
        has_today = today_str in dates

        return has_today, latest_date

    except Exception as e:
        log(f"Error checking {report_type} report: {e}")
        return False, "error"


def should_check_daily(report_hour):
    """
    Determine if we should expect today's daily report to exist.
    Only check after the scheduled time + grace period.
    """
    now = datetime.now(CST)
    scheduled_time = now.replace(hour=report_hour, minute=0, second=0, microsecond=0)
    grace = timedelta(minutes=DAILY_GRACE_MINUTES)

    return now >= scheduled_time + grace


# ============ API Test ============

def test_api():
    """Quick test if any LLM API endpoint is responsive."""
    import urllib.request

    api_key = os.environ.get("OPENAI_API_KEY", DEFAULT_API_KEY)
    if not api_key:
        log("No API key available")
        return False

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
            "model": "google/gemini-2.5-flash",
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
                    log(f"API OK: {base_url}")
                    os.environ["OPENAI_BASE_URL"] = base_url
                    return True
        except Exception as e:
            log(f"API failed [{base_url}]: {e}")
            continue

    log("All API endpoints failed")
    return False


# ============ Recovery Functions ============

def build_env_string():
    """Build environment variable prefix for subprocess calls."""
    env_extra = ""
    if not os.environ.get("OPENAI_BASE_URL"):
        env_extra += f'OPENAI_BASE_URL="{DEFAULT_BASE_URL}" '
    if not os.environ.get("OPENAI_API_KEY") and DEFAULT_API_KEY:
        env_extra += f'OPENAI_API_KEY="{DEFAULT_API_KEY}" '
    env_extra += 'TZ=Asia/Shanghai '
    return env_extra


def run_breaking_update():
    """Run update_breaking.py and return success."""
    env = build_env_string()
    cmd = f"{env}python3 scripts/update_breaking.py"
    log("Running Breaking News update...")
    ok, out = run(cmd, timeout_sec=120)

    if ok and "Done!" in out:
        log("Breaking News update OK")
        return True
    else:
        log("Breaking News update FAILED")
        for line in out.split("\n")[-5:]:
            if line.strip():
                log(f"  {line.strip()}")
        return False


def run_finance_daily():
    """Run update_finance.py and return success."""
    env = build_env_string()
    cmd = f"{env}python3 scripts/update_finance.py"
    log("Running Finance Daily Report...")
    ok, out = run(cmd, timeout_sec=600)  # Daily reports take longer

    if ok and "Done!" in out:
        log("Finance Daily Report OK")
        return True
    else:
        log("Finance Daily Report FAILED")
        for line in out.split("\n")[-5:]:
            if line.strip():
                log(f"  {line.strip()}")
        return False


def run_ai_daily():
    """Run update_ai.py and return success."""
    env = build_env_string()
    cmd = f"{env}python3 scripts/update_ai.py"
    log("Running AI Daily Report...")
    ok, out = run(cmd, timeout_sec=600)

    if ok and "Done!" in out:
        log("AI Daily Report OK")
        return True
    else:
        log("AI Daily Report FAILED")
        for line in out.split("\n")[-5:]:
            if line.strip():
                log(f"  {line.strip()}")
        return False


def git_push(commit_msg=None):
    """Commit and push changes."""
    ok, out = run("git diff --stat index.html")
    if not ok or not out.strip():
        log("No changes to push")
        return True

    if not commit_msg:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M CST")
        commit_msg = f"auto: health check recovery {now_str}"

    ok1, _ = run("git add index.html")
    ok2, _ = run(f'git commit -m "{commit_msg}"')
    if not ok2:
        log("Commit failed")
        return False

    for attempt in range(MAX_RETRY):
        ok, out = run("git push origin main")
        if ok:
            log("Push OK")
            return True
        log(f"Push rejected (attempt {attempt+1}), pulling...")
        run("git pull --rebase origin main")

    log("Push failed after retries")
    return False


# ============ Main ============

def main():
    log("=" * 50)
    log("Y Daily Health Check v2")
    log("=" * 50)

    now = datetime.now(CST)
    log(f"Current time: {now.strftime('%Y-%m-%d %H:%M CST')}")

    # Step 1: Pull latest
    log("\n--- Git Pull ---")
    if not git_pull():
        log("ERROR: git pull failed, aborting")
        return 1

    issues_found = []
    recovery_needed = []

    # ============ Check 1: Breaking News ============
    log("\n--- Check: Breaking News ---")
    age = get_last_auto_commit_age()
    if age is None:
        log("WARNING: No auto commits found in last 48h")
        issues_found.append("Breaking News: no commits in 48h")
        recovery_needed.append("breaking")
    elif age >= BREAKING_STALE_MINUTES:
        log(f"STALE — {age:.0f}min since last commit (threshold: {BREAKING_STALE_MINUTES}min)")
        issues_found.append(f"Breaking News: stale ({age:.0f}min)")
        recovery_needed.append("breaking")
    else:
        log(f"OK — last commit {age:.0f}min ago")

    # ============ Check 2: Finance Daily Report ============
    log("\n--- Check: Finance Daily Report ---")
    if should_check_daily(DAILY_REPORT_HOUR):
        has_today, latest_date = check_daily_report_in_html("finance")
        if has_today:
            log(f"OK — today's report exists (latest: {latest_date})")
        else:
            log(f"MISSING — today's report not found (latest: {latest_date})")
            issues_found.append(f"Finance Daily: missing today (latest: {latest_date})")
            recovery_needed.append("finance")
    else:
        log(f"SKIP — not yet past {DAILY_REPORT_HOUR}:00 CST + grace period")

    # ============ Check 3: AI Daily Report ============
    log("\n--- Check: AI Daily Report ---")
    if should_check_daily(AI_REPORT_HOUR):
        has_today, latest_date = check_daily_report_in_html("ai")
        if has_today:
            log(f"OK — today's report exists (latest: {latest_date})")
        else:
            log(f"MISSING — today's report not found (latest: {latest_date})")
            issues_found.append(f"AI Daily: missing today (latest: {latest_date})")
            recovery_needed.append("ai")
    else:
        log(f"SKIP — not yet past {AI_REPORT_HOUR}:00 CST + grace period")

    # ============ Summary ============
    log("\n--- Summary ---")
    if not issues_found:
        log("ALL OK — no issues detected")
        return 0

    log(f"Issues found ({len(issues_found)}):")
    for issue in issues_found:
        log(f"  ⚠️  {issue}")

    # ============ Recovery ============
    log("\n--- Recovery ---")
    log("Testing API availability...")

    if not test_api():
        log("API is down — cannot recover, will retry next cycle")
        return 1

    log("API is up — starting recovery")
    recovered = []

    if "breaking" in recovery_needed:
        if run_breaking_update():
            recovered.append("breaking")

    if "finance" in recovery_needed:
        if run_finance_daily():
            recovered.append("finance")

    if "ai" in recovery_needed:
        if run_ai_daily():
            recovered.append("ai")

    if recovered:
        parts = []
        if "breaking" in recovered:
            parts.append("breaking news")
        if "finance" in recovered:
            parts.append("finance daily")
        if "ai" in recovered:
            parts.append("AI daily")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M CST")
        commit_msg = f"auto: recovery {' + '.join(parts)} {now_str}"

        if git_push(commit_msg):
            log(f"\nRecovery complete: {', '.join(parts)}")
        else:
            log("\nRecovery ran but push failed")
            return 1
    else:
        log("\nNo successful recovery")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
