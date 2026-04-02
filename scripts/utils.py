"""
Shared utilities for Y Daily automation scripts.
Handles reading/writing JS data blocks embedded in index.html.
"""

import re
import json
import os
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
    """
    import subprocess

    # Use node to evaluate JS and output JSON
    node_script = f"""
    try {{
        const data = {text};
        console.log(JSON.stringify(data));
    }} catch(e) {{
        console.error('PARSE_ERROR: ' + e.message);
        process.exit(1);
    }}
    """

    node_paths = [
        "node",
        os.path.expanduser("~/.workbuddy/binaries/node/versions/22.12.0/bin/node"),
    ]

    for node in node_paths:
        try:
            result = subprocess.run(
                [node, "-e", node_script],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
            continue

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
