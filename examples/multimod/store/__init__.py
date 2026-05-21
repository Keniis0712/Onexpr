"""store package — tiny in-memory record store with serializable entries."""
__version__ = '0.1'

# Surface the most-used names; the bundler should pick these up so
# `from store import api` works after bundling too.
from store import api
from store import cache

__all__ = ['api', 'cache', '__version__']
