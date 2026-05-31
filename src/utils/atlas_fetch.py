"""
SSL-robust atlas fetching for nilearn.

The Craddock-2012 (CC200) atlas is hosted on nitrc.org, which ships an
**invalid TLS certificate**. A normal ``datasets.fetch_atlas_craddock_2012``
therefore fails with an ``SSLError`` and leaves a half-written cache directory
behind, which then makes every subsequent attempt fail too.

``robust_fetch`` wraps any nilearn fetcher so that a single command works
unattended on local / Colab / Kaggle:

  1. Try the fetch normally (uses the system certificate store — the happy path
     for atlases hosted on GitHub/S3 with valid certs).
  2. On *any* failure, clear the empty/partial cache directories that nilearn
     created, disable TLS verification **for the duration of the retry only**
     (urllib3 warning suppression + a ``requests.Session.send`` wrapper +
     ``ssl`` default-context override), then retry once.

It is self-contained and idempotent: the SSL patch is always restored in a
``finally`` block, and partial caches are only removed on the failure path so a
valid cache is never deleted.
"""

from __future__ import annotations

import shutil
import ssl
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable


@contextmanager
def ssl_unverified():
    """Temporarily disable TLS verification for nilearn/requests downloads.

    Restores the original ``requests.Session.send`` and ``ssl`` default context
    on exit, so the relaxed verification never leaks beyond the ``with`` block.
    """
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    orig_send = requests.Session.send
    orig_ctx = ssl._create_default_https_context  # type: ignore[attr-defined]

    def _send(self, *args, **kwargs):
        kwargs["verify"] = False
        return orig_send(self, *args, **kwargs)

    try:
        requests.Session.send = _send  # type: ignore[assignment]
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]
        yield
    finally:
        requests.Session.send = orig_send  # type: ignore[assignment]
        ssl._create_default_https_context = orig_ctx  # type: ignore[attr-defined]


def _clear_partial(dirs: Iterable[str | Path]) -> None:
    """Remove cache directories left half-written by a failed download."""
    for d in dirs:
        p = Path(d)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)


def robust_fetch(
    fetch_fn: Callable,
    *args,
    partial_dirs: Iterable[str | Path] = (),
    **kwargs,
):
    """Call a nilearn atlas fetcher, retrying once with TLS verification off.

    Parameters
    ----------
    fetch_fn     : nilearn fetcher (e.g. ``datasets.fetch_atlas_craddock_2012``)
    partial_dirs : cache directories to delete before the retry if the first
                   attempt fails (so nilearn re-downloads cleanly)
    *args, **kwargs : forwarded to ``fetch_fn``
    """
    try:
        return fetch_fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — SSLError and friends
        print(f"  [atlas_fetch] '{getattr(fetch_fn, '__name__', fetch_fn)}' "
              f"failed ({type(exc).__name__}: {exc}); "
              f"clearing partial cache + retrying with TLS verification off ...")
        _clear_partial(partial_dirs)
        with ssl_unverified():
            return fetch_fn(*args, **kwargs)
