"""Database API. Exercises:
  - relative imports (`from . import cache`)
  - class with __init__ / methods / __slots__
  - closures (the lambda passed to search, plus our memoized count)
  - module-level globals() coverage (counter for record IDs)
  - PEP 695 type alias
"""
from . import cache

# Module-level state, incremented every put. Mainly here to make sure
# globals() inside a member method sees the right module dict.
_NEXT_ID = 0


type RecordId = int
type Record = dict[str, object]


class Database:
    """Tiny in-memory key/value store with closure-aware querying."""

    __slots__ = ('_records', '_ids')

    def __init__(self) -> None:
        self._records: dict[str, Record] = {}
        self._ids: dict[str, RecordId] = {}

    def put(self, key: str, value: Record) -> RecordId:
        global _NEXT_ID
        _NEXT_ID += 1
        self._records[key] = dict(value)
        self._ids[key] = _NEXT_ID
        return _NEXT_ID

    def get(self, key: str) -> Record | None:
        return self._records.get(key)

    def delete(self, key: str) -> None:
        self._records.pop(key, None)
        self._ids.pop(key, None)

    @cache.memoize(maxsize=4)
    def count(self) -> int:
        return len(self._records)

    def search(self, predicate) -> list[str]:
        return [k for k, v in self._records.items() if predicate(v)]

    def __repr__(self) -> str:
        return f'Database(n={len(self._records)})'


def latest_id() -> RecordId:
    """Reads the module global. Confirms globals() inside the bundle
    refers to *this* module, not the bundle file."""
    return globals().get('_NEXT_ID', -1)
