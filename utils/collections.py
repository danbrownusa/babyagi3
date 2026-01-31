"""
Thread-Safe Collections

Provides thread-safe wrappers for common collections.
"""

import threading


class ThreadSafeList:
    """Thread-safe list wrapper for concurrent access.

    Provides safe iteration (via copy) and atomic operations.
    Used for global shared state like MEMORIES and NOTES.

    Example:
        items = ThreadSafeList()
        items.append("hello")
        for item in items:  # Iterates over a copy
            print(item)
    """

    def __init__(self):
        self._list: list = []
        self._lock = threading.Lock()

    def append(self, item):
        """Thread-safe append."""
        with self._lock:
            self._list.append(item)

    def pop(self, index: int = -1):
        """Thread-safe pop."""
        with self._lock:
            return self._list.pop(index)

    def __getitem__(self, index):
        """Thread-safe index access."""
        with self._lock:
            return self._list[index]

    def __setitem__(self, index, value):
        """Thread-safe index assignment."""
        with self._lock:
            self._list[index] = value

    def __len__(self):
        """Thread-safe length."""
        with self._lock:
            return len(self._list)

    def __iter__(self):
        """Return iterator over a copy (safe for concurrent modification)."""
        with self._lock:
            return iter(self._list.copy())

    def copy(self) -> list:
        """Return a copy of the list."""
        with self._lock:
            return self._list.copy()

    def __bool__(self):
        """Thread-safe bool check."""
        with self._lock:
            return bool(self._list)
