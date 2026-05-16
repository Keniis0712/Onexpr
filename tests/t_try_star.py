def basic_except_star():
    log = []
    try:
        raise ExceptionGroup('g', [ValueError('v'), TypeError('t')])
    except* ValueError as eg:
        log.append(('value', [str(e) for e in eg.exceptions]))
    except* TypeError as eg:
        log.append(('type', [str(e) for e in eg.exceptions]))
    return log


print(basic_except_star())


def except_star_partial_handle():
    log = []
    try:
        try:
            raise ExceptionGroup('g', [
                ValueError('v'), TypeError('t'), KeyError('k'),
            ])
        except* ValueError as eg:
            log.append(('value', len(eg.exceptions)))
    except ExceptionGroup as outer:
        log.append(('outer', sorted(type(e).__name__ for e in outer.exceptions)))
    return log


print(except_star_partial_handle())


def except_star_single_exception_promoted():
    # A non-group raise inside try* gets wrapped to a group, but if it
    # falls through unhandled, the result should still propagate.
    log = []
    try:
        try:
            raise ValueError('lone')
        except* TypeError as eg:
            log.append('type-handler')
    except ExceptionGroup as outer:
        log.append(('group', sorted(type(e).__name__ for e in outer.exceptions)))
    except ValueError as e:
        log.append(('value', str(e)))
    return log


print(except_star_single_exception_promoted())


def except_star_handler_raises_new():
    log = []
    try:
        try:
            raise ExceptionGroup('g', [ValueError('v')])
        except* ValueError:
            raise RuntimeError('handler-raised')
    except ExceptionGroup as outer:
        # When the handler raises a non-group, Python wraps it back
        # into a group together with anything still unmatched.
        log.append(('outer-group', sorted(type(e).__name__ for e in outer.exceptions)))
    except RuntimeError as e:
        log.append(('outer-runtime', str(e)))
    return log


print(except_star_handler_raises_new())


def except_star_with_finally():
    log = []
    try:
        try:
            raise ExceptionGroup('g', [ValueError('v')])
        except* ValueError as eg:
            log.append(('handled', len(eg.exceptions)))
        finally:
            log.append('finally')
    except ExceptionGroup:
        log.append('outer-shouldnt-fire')
    return log


print(except_star_with_finally())


def except_star_no_exception():
    log = []
    try:
        log.append('try-body')
    except* ValueError as eg:
        log.append('handler-shouldnt')
    log.append('after')
    return log


print(except_star_no_exception())


def except_star_tuple_types():
    log = []
    try:
        raise ExceptionGroup('g', [
            ValueError('v'), TypeError('t'), KeyError('k'),
        ])
    except* (ValueError, TypeError) as eg:
        log.append(('vt', sorted(type(e).__name__ for e in eg.exceptions)))
    except* KeyError as eg:
        log.append(('k', sorted(type(e).__name__ for e in eg.exceptions)))
    return log


print(except_star_tuple_types())
