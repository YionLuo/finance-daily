"""
Shared utilities for Y Daily automation scripts.
Handles reading/writing JS data blocks embedded in index.html.
Provides shared LLM client creation with fallback endpoints and retry logic.
"""

import re
import json
import os
import time as _time
from datetime import datetime, timezone, timedelta

CST = timezone(timedelta(hours=8))
INDEX_PATH = os.path.join(os.path.dirname(__file__), '..', 'index.html')


def read_html():
    """Read the full index.html content."""
    with open(INDEX_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def write_html(content):
    """Write the full index.html content."""
    with open(INDEX_PATH, 'w', encoding='utf-8') as f:
        f.write(content)


def _find_matching_bracket(html, start_pos, open_char, close_char):
    """
    Find the matching closing bracket, properly handling strings.
    Returns the position of the closing bracket.
    """
    depth = 0
    i = start_pos
    while i < len(html):
        ch = html[i]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return i
        elif ch == '"':
            i += 1
            while i < len(html) and html[i] != '"':
                if html[i] == '\\':
                    i += 1
                i += 1
        elif ch == "'":
            i += 1
            while i < len(html) and html[i] != "'":
                if html[i] == '\\':
                    i += 1
                i += 1
        elif ch == '`':
            i += 1
            while i < len(html) and html[i] != '`':
                if html[i] == '\\':
                    i += 1
                i += 1
        i += 1
    return -1


def _parse_js_value(text):
    """
    Parse a raw JS text block into Python objects using node.
    This is more reliable than regex-based parsing for complex JS.
    Uses stdin to avoid 'Argument list too long' errors with large data.
    """
    import subprocess
    import tempfile

    # Write JS data to a temp file, then have node read and parse it.
    # This avoids both ARG_MAX limits (command line) and pipe buffer issues (stdin).
    node_paths = [
        "node",
        os.path.expanduser("~/.workbuddy/binaries/node/versions/22.12.0/bin/node"),
    ]

    for node in node_paths:
        tmp_file = None
        try:
            # Write JS value to a temp file
            tmp_file = tempfile.NamedTemporaryFile(
                mode='w', suffix='.js', delete=False, encoding='utf-8'
            )
            tmp_file.write(text)
            tmp_file.close()

            node_script = f"""
            const fs = require('fs');
            try {{
                const raw = fs.readFileSync({json.dumps(tmp_file.name)}, 'utf8');
                const data = eval('(' + raw + ')');
                console.log(JSON.stringify(data));
            }} catch(e) {{
                console.error('PARSE_ERROR: ' + e.message);
                process.exit(1);
            }}
            """

            result = subprocess.run(
                [node, "-e", node_script],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
            continue
        finally:
            if tmp_file:
                try:
                    os.unlink(tmp_file.name)
                except OSError:
                    pass

    return None


def extract_js_array(html, var_name):
    """
    Extract a JS array from `const varName = [...]` in the HTML.
    Uses node for reliable parsing of complex JS (template literals, emoji, etc.).
    """
    pattern = rf'const\s+{var_name}\s*=\s*\['
    match = re.search(pattern, html)
    if not match:
        return []

    bracket_pos = html.index('[', match.start())
    end_pos = _find_matching_bracket(html, bracket_pos, '[', ']')
    if end_pos == -1:
        return []

    array_str = html[bracket_pos:end_pos + 1]
    result = _parse_js_value(array_str)
    return result if isinstance(result, list) else []


def extract_js_object(html, var_name):
    """
    Extract a JS object from `const varName = {...}` in the HTML.
    Uses node for reliable parsing.
    """
    pattern = rf'const\s+{var_name}\s*=\s*\{{'
    match = re.search(pattern, html)
    if not match:
        return {}

    bracket_pos = html.index('{', match.start())
    end_pos = _find_matching_bracket(html, bracket_pos, '{', '}')
    if end_pos == -1:
        return {}

    obj_str = html[bracket_pos:end_pos + 1]
    result = _parse_js_value(obj_str)
    return result if isinstance(result, dict) else {}


def extract_js_string(html, var_name):
    """Extract a JS string from `const varName = "..."`."""
    pattern = rf'const\s+{var_name}\s*=\s*"([^"]*)"'
    match = re.search(pattern, html)
    if match:
        return match.group(1)
    return ""


def replace_js_array(html, var_name, new_data, indent=0):
    """
    Replace a JS array in the HTML with new data.
    new_data: Python list of dicts.
    Returns the modified HTML string.
    """
    js_str = python_to_js_array(new_data, indent)

    match = re.search(rf'const\s+{var_name}\s*=\s*\[', html)
    if not match:
        return html

    bracket_pos = html.index('[', match.start())
    end_bracket = _find_matching_bracket(html, bracket_pos, '[', ']')
    if end_bracket == -1:
        return html

    # Also consume the trailing semicolon
    end = end_bracket + 1
    if end < len(html) and html[end] == ';':
        end += 1

    prefix = html[match.start():bracket_pos]
    replacement = prefix + js_str + ';'
    return html[:match.start()] + replacement + html[end:]


def replace_js_string(html, var_name, new_value):
    """Replace a JS string constant."""
    pattern = rf'(const\s+{var_name}\s*=\s*")[^"]*(")'
    return re.sub(pattern, rf'\g<1>{new_value}\2', html)


def replace_js_object(html, var_name, new_data):
    """Replace a JS object in the HTML."""
    js_str = python_to_js_object(new_data)

    match = re.search(rf'const\s+{var_name}\s*=\s*\{{', html)
    if not match:
        return html

    bracket_pos = html.index('{', match.start())
    end_bracket = _find_matching_bracket(html, bracket_pos, '{', '}')
    if end_bracket == -1:
        return html

    end = end_bracket + 1
    if end < len(html) and html[end] == ';':
        end += 1

    prefix = html[match.start():bracket_pos]
    replacement = prefix + js_str + ';'
    return html[:match.start()] + replacement + html[end:]


# ============ JS <-> Python Conversion ============

def js_array_to_json(js_str):
    """
    Convert JS array notation to valid JSON.
    Handles: unquoted keys, single quotes, trailing commas.
    """
    result = js_str

    # Remove JS comments
    result = re.sub(r'//[^\n]*', '', result)

    # Handle template literals (backtick strings) -> double-quoted strings
    # This is complex; for our use case the data is relatively simple
    def replace_template(m):
        content = m.group(1)
        # Escape double quotes inside
        content = content.replace('\\', '\\\\').replace('"', '\\"')
        content = content.replace('\n', '\\n')
        return '"' + content + '"'

    result = re.sub(r'`([^`]*)`', replace_template, result)

    # Replace single-quoted strings with double-quoted
    # (simplified - assumes no nested quotes)
    result = re.sub(r"'([^']*)'", r'"\1"', result)

    # Quote unquoted keys: word: -> "word":
    result = re.sub(r'(?<=[\{,\n])\s*(\w+)\s*:', r' "\1":', result)

    # Remove trailing commas before ] or }
    result = re.sub(r',\s*([\]\}])', r'\1', result)

    return result


def js_object_to_json(js_str):
    """Convert JS object notation to valid JSON."""
    return js_array_to_json(js_str)


def python_to_js_array(data, indent=0):
    """
    Convert Python list of dicts to JS array string.
    Uses unquoted keys for cleaner output.
    """
    if not data:
        return '[]'

    lines = ['[']
    for item in data:
        lines.append('  ' + python_to_js_object_inline(item) + ',')
    lines.append(']')
    return '\n'.join(lines)


def python_to_js_object(data):
    """Convert Python dict to JS object string (with unquoted keys)."""
    return _dict_to_js(data, indent=0)


def python_to_js_object_inline(data):
    """Convert a simple flat dict to a single-line JS object."""
    parts = []
    for key, value in data.items():
        parts.append(f'{key}: {_js_value(value)}')
    return '{ ' + ', '.join(parts) + ' }'


def _js_value(val):
    """Convert a Python value to JS literal."""
    if val is None:
        return 'null'
    if isinstance(val, bool):
        return 'true' if val else 'false'
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        # Use double quotes, escape special chars
        escaped = val.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        return f'"{escaped}"'
    if isinstance(val, list):
        items = ', '.join(_js_value(v) for v in val)
        return f'[{items}]'
    if isinstance(val, dict):
        return python_to_js_object_inline(val)
    return json.dumps(val)


def _dict_to_js(data, indent=0):
    """Convert dict to multi-line JS object with proper indentation."""
    pad = '  ' * indent
    inner_pad = '  ' * (indent + 1)
    lines = ['{']
    items = list(data.items())
    for i, (key, value) in enumerate(items):
        comma = ',' if i < len(items) - 1 else ''
        if isinstance(value, dict):
            lines.append(f'{inner_pad}{key}: {_dict_to_js(value, indent + 1)}{comma}')
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            lines.append(f'{inner_pad}{key}: [')
            for j, item in enumerate(value):
                item_comma = ',' if j < len(value) - 1 else ''
                if isinstance(item, dict):
                    lines.append(f'{inner_pad}  {python_to_js_object_inline(item)}{item_comma}')
                else:
                    lines.append(f'{inner_pad}  {_js_value(item)}{item_comma}')
            lines.append(f'{inner_pad}]{comma}')
        elif isinstance(value, list):
            items_str = ', '.join(_js_value(v) for v in value)
            lines.append(f'{inner_pad}{key}: [{items_str}]{comma}')
        else:
            lines.append(f'{inner_pad}{key}: {_js_value(value)}{comma}')
    lines.append(f'{pad}}}')
    return '\n'.join(lines)


def now_cst():
    """Get current time in CST."""
    return datetime.now(CST)


def format_date_cst(dt=None):
    """Format datetime as '2026年4月2日 16:29 CST'."""
    if dt is None:
        dt = now_cst()
    return f"{dt.year}年{dt.month}月{dt.day}日 {dt.hour:02d}:{dt.minute:02d} CST"


def is_within_hours(time_str, hours=24):
    """
    Check if a HH:MM time string is within the last N hours.
    Since breaking news only stores time (not date), we use the breakingDate
    to determine the reference date. For simplicity, return True (caller handles).
    """
    return True  # Actual filtering done at a higher level


# ============ LLM Client & Retry ============

# Fallback API endpoints — tried in order if the primary fails with connection errors
FALLBACK_ENDPOINTS = [
    "https://api.deepseek.com",
]


def default_model_for_base_url(base_url):
    """Return a working default model for the configured OpenAI-compatible gateway."""
    if "openrouter" in base_url:
        return "deepseek/deepseek-v4-flash"
    return "deepseek-chat"


# Default model — auto-detect based on API endpoint
_base = os.environ.get("OPENAI_BASE_URL", FALLBACK_ENDPOINTS[0])
DEFAULT_LLM_MODEL = default_model_for_base_url(_base)


def create_llm_client(required=True):
    """
    Create an OpenAI-compatible client with multi-endpoint fallback.

    Args:
        required: If True, raise SystemExit when API key is missing.
                  If False, return None (for breaking news cleanup-only mode).

    Returns:
        OpenAI client instance, or None if not required and unavailable.
    """
    try:
        from openai import OpenAI
    except ImportError:
        if required:
            print("ERROR: openai package not installed.")
            raise SystemExit(1)
        print("WARNING: openai package not installed. Cleanup-only mode.")
        return None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if required:
            print("ERROR: OPENAI_API_KEY is required.")
            raise SystemExit(1)
        print("WARNING: OPENAI_API_KEY not set. Cleanup-only mode.")
        return None

    # Build endpoint list: env var first, then fallbacks
    endpoints = []
    env_url = os.environ.get("OPENAI_BASE_URL")
    if env_url and env_url.strip():
        endpoints.append(env_url.strip())
    for ep in FALLBACK_ENDPOINTS:
        if ep not in endpoints:
            endpoints.append(ep)

    print(f"LLM endpoints to try ({len(endpoints)}): {endpoints}")

    # Try each endpoint with a quick chat completion test (short timeout)
    working_ep = None
    working_model = None
    explicit_model = os.environ.get("LLM_MODEL")
    for ep in endpoints:
        try:
            test_client = OpenAI(api_key=api_key, base_url=ep, timeout=15)
            model = explicit_model or default_model_for_base_url(ep)
            # Use a minimal chat completion instead of models.list()
            # This tests the actual path we'll use for report generation
            test_resp = test_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "say ok"}],
                max_tokens=5,
            )
            if test_resp.choices and test_resp.choices[0].message.content:
                print(f"LLM endpoint OK: {ep} ({model})")
                working_ep = ep
                working_model = model
                break
            else:
                print(f"LLM endpoint {ep}: empty response")
        except Exception as e:
            print(f"LLM endpoint {ep} failed: {e}")
            continue

    # Use the working endpoint, or fall back to first one
    chosen_ep = working_ep or (endpoints[0] if endpoints else None)
    if chosen_ep:
        if not working_ep:
            print(f"WARNING: All endpoints failed connectivity test, using {chosen_ep} anyway")
            working_model = explicit_model or default_model_for_base_url(chosen_ep)
        os.environ["OPENAI_BASE_URL"] = chosen_ep
        if not explicit_model and working_model:
            os.environ["LLM_MODEL"] = working_model
        # Create actual client with long timeout for LLM generation
        return OpenAI(api_key=api_key, base_url=chosen_ep, timeout=300)

    if required:
        print("ERROR: No working LLM endpoint found.")
        raise SystemExit(1)
    return None


def llm_chat_with_retry(client, messages, model=None, max_tokens=4000,
                        temperature=0.2, max_retries=3, backoff_base=2):
    """
    LLM chat completion with exponential backoff retry.

    Retries on: network errors, timeouts, HTTP 5xx, empty responses.
    Does NOT retry on: 401 (auth), 404 (model not found), 400 (bad request).

    Args:
        client: OpenAI client instance
        messages: List of message dicts
        model: Model name (defaults to LLM_MODEL env var or deepseek-chat)
        max_tokens: Max response tokens
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        backoff_base: Base seconds for exponential backoff

    Returns:
        Response content string

    Raises:
        Exception: If all retries exhausted
    """
    if model is None:
        model = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=180,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("LLM returned empty content")
            return content.strip()

        except Exception as e:
            last_error = e
            err_str = str(e).lower()

            # Non-retryable errors
            if any(code in err_str for code in ["401", "unauthorized", "invalid api key"]):
                print(f"ERROR: Authentication failed — {e}")
                raise
            if "404" in err_str and "model" in err_str:
                print(f"ERROR: Model not found — {e}")
                raise
            if "400" in err_str and "bad request" in err_str:
                print(f"ERROR: Bad request — {e}")
                raise

            # Retryable
            if attempt < max_retries:
                wait = backoff_base ** (attempt + 1)
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"Retrying in {wait}s...")
                _time.sleep(wait)
            else:
                print(f"LLM call failed after {max_retries + 1} attempts: {e}")

    raise last_error
