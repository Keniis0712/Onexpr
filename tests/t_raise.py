import sys


def _excepthook(exc_type, exc_value, tb):
    print('hook', exc_type.__name__, exc_value)
    sys.exit(0)


sys.excepthook = _excepthook


def trigger():
    raise KeyError('k') from ValueError('v')


trigger()
