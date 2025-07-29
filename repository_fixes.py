#!/usr/bin/env python3
"""
Quick fix script to add None checks to all repository methods.
This will add proper error handling to prevent None attribute access errors.
"""

import re


def add_connection_checks(file_path):
    """Add connection checks to repository methods."""

    with open(file_path, "r") as f:
        content = f.read()

    # Pattern 1: Add connection check for store_event method (SQLite)
    content = re.sub(
        r'(async def store_event.*?""".*?""")\s*async with self\._lock:',
        r"\1\n        if not self._ensure_connection():\n            return False\n            \n        async with self._lock:",
        content,
        flags=re.DOTALL,
    )

    # Pattern 2: Add session check for _store_paradox that's still missing the check
    content = re.sub(
        r'(async def _store_paradox.*?""".*?""")\s*self\.session\.execute\(',
        r"\1\n        if not self._ensure_session():\n            return\n            \n        self.session.execute(",
        content,
        flags=re.DOTALL,
    )

    # Pattern 3: Wrap all remaining cursor accesses with connection checks
    cursor_pattern = r"(\s+)(cursor = self\._connection\.cursor\(\))"
    content = re.sub(
        cursor_pattern,
        r"\1if not self._ensure_connection():\n\1    return False\n\1cursor = self._connection.cursor()",
        content,
    )

    # Pattern 4: Wrap commit calls with connection checks
    commit_pattern = r"(\s+)(self\._connection\.commit\(\))"
    content = re.sub(
        commit_pattern,
        r"\1if self._connection is not None:\n\1    self._connection.commit()",
        content,
    )

    with open(file_path, "w") as f:
        f.write(content)

    print(f"✅ Applied connection/session checks to {file_path}")


if __name__ == "__main__":
    add_connection_checks(
        "/Users/chanduchitikam/omega-phr/services/timeline_lattice/app/repository.py"
    )
    print("🎯 Repository fixes completed!")
