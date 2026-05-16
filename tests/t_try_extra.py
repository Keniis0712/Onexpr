"""Cases that stress the try implementation beyond the basics in
t_try.py: exception type matching with subclasses / tuples,
implicit / explicit __context__ chains, exceptions raised inside
each clause, try inside a class body."""


def case_subclass_match():
    log = []
    try:
        raise ValueError('v')
    except LookupError:
        log.append('lookup')
    except Exception:
        log.append('exception-catchall')
    return log


print(case_subclass_match())


def case_tuple_exception_types():
    log = []
    for exc_cls in (ValueError, TypeError, KeyError):
        try:
            raise exc_cls('e')
        except (ValueError, TypeError) as e:
            log.append(('vt', type(e).__name__))
        except KeyError as e:
            log.append(('k', type(e).__name__))
    return log


print(case_tuple_exception_types())


def case_raise_from():
    try:
        try:
            raise ValueError('original')
        except ValueError as e:
            raise RuntimeError('wrapped') from e
    except RuntimeError as outer:
        return (
            type(outer).__name__,
            type(outer.__cause__).__name__,
            outer.__cause__.args,
        )


print(case_raise_from())


def case_implicit_context_handler():
    try:
        try:
            raise ValueError('orig')
        except ValueError:
            raise RuntimeError('new')
    except RuntimeError as e:
        return (
            type(e).__name__,
            type(e.__context__).__name__,
            e.__context__.args,
        )


print(case_implicit_context_handler())


def case_implicit_context_finally():
    try:
        try:
            raise ValueError('first')
        finally:
            raise RuntimeError('finally-wins')
    except RuntimeError as e:
        return (
            type(e).__name__,
            e.args,
            type(e.__context__).__name__,
            e.__context__.args,
        )


print(case_implicit_context_finally())


def case_exception_in_handler():
    log = []
    try:
        try:
            raise ValueError('first')
        except ValueError:
            log.append('handler-running')
            raise TypeError('second')
    except TypeError as e:
        log.append(('outer-caught', str(e)))
    return log


print(case_exception_in_handler())


def case_exception_in_else():
    log = []
    try:
        try:
            log.append('try-body')
        except ValueError:
            log.append('inner-handler-not-run')
        else:
            log.append('else-running')
            raise RuntimeError('from-else')
    except RuntimeError as e:
        log.append(('outer-caught', str(e)))
    return log


print(case_exception_in_else())


def case_exception_in_finally():
    log = []
    try:
        try:
            log.append('try-body')
        finally:
            log.append('finally-running')
            raise RuntimeError('from-finally')
    except RuntimeError as e:
        log.append(('outer-caught', str(e)))
    return log


print(case_exception_in_finally())


def case_try_in_class_body():
    class C:
        try:
            x = 'try-set'
        except:
            x = 'handler-set'

        try:
            raise ValueError
        except ValueError:
            y = 'caught'
    return (C.x, C.y)


print(case_try_in_class_body())


def case_alias_match():
    MyExc = ValueError
    try:
        raise ValueError('v')
    except MyExc:
        return 'matched-via-alias'


print(case_alias_match())


def case_nested_reraise_to_parent():
    log = []
    try:
        try:
            raise FileNotFoundError('missing')
        except FileNotFoundError:
            log.append('inner-handler')
            raise
    except OSError as e:
        log.append(('outer', type(e).__name__, e.args))
    return log


print(case_nested_reraise_to_parent())


def case_finally_overrides_return():
    def f():
        try:
            return 'try-return'
        finally:
            return 'finally-return'
    return f()


print(case_finally_overrides_return())


def case_try_in_called_lambda():
    def safe_div(a, b):
        try:
            return a / b
        except ZeroDivisionError:
            return None
    return [safe_div(10, x) for x in (1, 2, 0, 5, 0)]


print(case_try_in_called_lambda())


def case_break_in_try_handler_finally():
    log = []
    for i in range(5):
        try:
            log.append(('try', i))
            if i == 2:
                raise ValueError
        except ValueError:
            log.append('handler')
            break
        finally:
            log.append(('finally', i))
        log.append(('after-try', i))
    log.append('done')
    return log


print(case_break_in_try_handler_finally())


def case_continue_in_finally():
    log = []
    for i in range(4):
        try:
            log.append(('try', i))
        finally:
            if i % 2 == 0:
                continue
        log.append(('after-try', i))
    return log


print(case_continue_in_finally())


def case_args_preserved():
    try:
        raise ValueError('original args', 42)
    except ValueError as e:
        return e.args


print(case_args_preserved())
