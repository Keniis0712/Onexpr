def basic_try():
    out = []
    try:
        out.append('try-before')
        raise ValueError('boom')
        out.append('try-after')
    except ValueError as e:
        out.append(('handler', str(e)))
    out.append('after')
    return out


print(basic_try())


def no_exception():
    out = []
    try:
        out.append('try-only')
    except ValueError:
        out.append('handler')
    out.append('after')
    return out


print(no_exception())


def multiple_handlers(kind):
    try:
        if kind == 'value':
            raise ValueError('v')
        elif kind == 'type':
            raise TypeError('t')
        elif kind == 'key':
            raise KeyError('k')
        return 'no exc'
    except ValueError as e:
        return ('value', str(e))
    except TypeError as e:
        return ('type', str(e))


print(multiple_handlers('value'))
print(multiple_handlers('type'))
print(multiple_handlers('none'))


def bare_except():
    try:
        raise RuntimeError('whatever')
    except:
        return 'caught'


print(bare_except())


def with_finally():
    log = []
    try:
        log.append('try')
        raise ValueError('boom')
    except ValueError:
        log.append('handler')
    finally:
        log.append('finally')
    log.append('after')
    return log


print(with_finally())


def finally_with_no_exc():
    log = []
    try:
        log.append('try')
    finally:
        log.append('finally')
    return log


print(finally_with_no_exc())


def with_else():
    log = []
    try:
        log.append('try')
    except ValueError:
        log.append('handler')
    else:
        log.append('else')
    finally:
        log.append('finally')
    return log


print(with_else())


def else_skipped_when_handled():
    log = []
    try:
        log.append('try')
        raise ValueError('boom')
    except ValueError:
        log.append('handler')
    else:
        log.append('else')
    finally:
        log.append('finally')
    return log


print(else_skipped_when_handled())


def unhandled_propagates():
    inner_log = []
    try:
        try:
            inner_log.append('inner-try')
            raise ValueError('boom')
        except KeyError:
            inner_log.append('inner-handler-keyerror')
    except ValueError as e:
        inner_log.append(('outer-handler', str(e)))
    return inner_log


print(unhandled_propagates())


def re_raise():
    log = []
    try:
        try:
            log.append('inner')
            raise TypeError('boom')
        except TypeError:
            log.append('caught-and-reraise')
            raise
    except TypeError as e:
        log.append(('outer-caught', str(e)))
    return log


print(re_raise())


def handler_can_raise_new():
    log = []
    try:
        try:
            log.append('inner')
            raise ValueError('original')
        except ValueError:
            log.append('handler-raises-new')
            raise RuntimeError('chained')
    except RuntimeError as e:
        log.append(('outer', str(e)))
    return log


print(handler_can_raise_new())


def return_in_try():
    log = []

    def f():
        try:
            log.append('try')
            return 'try-return'
        except ValueError:
            log.append('handler')
        finally:
            log.append('finally')
        log.append('after-try-block-unreachable')
        return 'fall-through'
    result = f()
    return (result, log)


print(return_in_try())


def return_in_handler():
    log = []

    def f():
        try:
            log.append('try')
            raise ValueError('boom')
        except ValueError:
            log.append('handler')
            return 'handler-return'
        finally:
            log.append('finally')
        return 'unreachable'
    result = f()
    return (result, log)


print(return_in_handler())


def finally_overrides_return():
    def f():
        try:
            return 'try-return'
        finally:
            return 'finally-return'
    return f()


print(finally_overrides_return())


def finally_swallows_exc():
    def f():
        try:
            raise ValueError('boom')
        finally:
            return 'finally-eats'
    return f()


print(finally_swallows_exc())


def try_in_loop():
    log = []
    for i in range(5):
        try:
            if i == 2:
                raise ValueError(i)
            log.append(('try', i))
        except ValueError as e:
            log.append(('handler', e.args[0]))
    return log


print(try_in_loop())


def break_in_try_then_finally():
    log = []
    for i in range(5):
        try:
            if i == 2:
                break
            log.append(('try', i))
        finally:
            log.append(('finally', i))
    log.append('after-loop')
    return log


print(break_in_try_then_finally())


def nested_try_handlers():
    log = []
    try:
        try:
            raise ValueError('inner-boom')
        except KeyError:
            log.append('inner-keyerror-handler')
        except ValueError as e:
            log.append(('inner-value-handler', str(e)))
            raise TypeError('rethrown-as-type')
    except TypeError as e:
        log.append(('outer-type-handler', str(e)))
    return log


print(nested_try_handlers())


def assignment_in_try_visible_outside():
    x = 'before'
    try:
        x = 'inside-try'
    except:
        pass
    return x


print(assignment_in_try_visible_outside())


def assignment_in_handler_visible_outside():
    x = 'before'
    try:
        raise ValueError('boom')
    except ValueError:
        x = 'inside-handler'
    return x


print(assignment_in_handler_visible_outside())


def continue_in_try_skips_after():
    out = []
    for i in range(4):
        try:
            if i % 2 == 0:
                continue
        except:
            pass
        out.append(('after-try', i))
    return out


print(continue_in_try_skips_after())


def break_in_try_exits_loop_immediately():
    out = []
    for i in range(5):
        try:
            if i == 2:
                break
        except:
            pass
        out.append(('after-try', i))
    out.append('after-loop')
    return out


print(break_in_try_exits_loop_immediately())


def continue_in_handler_skips_after():
    out = []
    for i in range(4):
        try:
            if i % 2 == 0:
                raise ValueError('skip')
        except ValueError:
            continue
        out.append(('after-try', i))
    return out


print(continue_in_handler_skips_after())


def break_in_handler_exits_loop():
    out = []
    for i in range(5):
        try:
            if i == 2:
                raise ValueError('stop')
            out.append(('try', i))
        except ValueError:
            break
        out.append(('after-try', i))
    out.append('done')
    return out


print(break_in_handler_exits_loop())


def continue_in_finally_skips_after():
    out = []
    for i in range(4):
        try:
            out.append(('try', i))
            if i == 1:
                raise ValueError
        except ValueError:
            pass
        finally:
            if i % 2 == 0:
                continue
        out.append(('after-try', i))
    return out


print(continue_in_finally_skips_after())


def nested_try_break():
    out = []
    for i in range(5):
        try:
            try:
                if i == 2:
                    break
            except:
                pass
            out.append(('inner-after-try', i))
        except:
            pass
        out.append(('outer-after-try', i))
    out.append('done')
    return out


print(nested_try_break())


def while_with_continue_in_try():
    out = []
    n = 0
    while n < 5:
        n += 1
        try:
            if n % 2 == 0:
                continue
        except:
            pass
        out.append(('after-try', n))
    return out


print(while_with_continue_in_try())
