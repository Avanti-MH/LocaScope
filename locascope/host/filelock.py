"""An exclusive lock on an open file, the same call on Linux and on Windows.

    with open(path, 'w') as lock, locked(lock):
        ...read, change, write the files the lock stands for...

Two processes finishing together each read a file before the other wrote it, and one loses its rows; the
lock makes the second wait. Linux takes `fcntl.flock` (released when the file is closed, as before); Windows
has no `fcntl`, so the first byte of the file is locked with `msvcrt.locking`.
"""
from __future__ import annotations

import contextlib
from typing import IO, Iterator

try:
    import fcntl
except ImportError:                                  # Windows
    fcntl = None
    import msvcrt


@contextlib.contextmanager
def locked(handle: IO) -> Iterator[None]:
    """Hold an exclusive lock on the open file `handle` until the block ends (waiting for it if another process holds it)."""
    if fcntl is not None:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
        return
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)       # retries for ~10 s, then raises OSError
    try:
        yield
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
