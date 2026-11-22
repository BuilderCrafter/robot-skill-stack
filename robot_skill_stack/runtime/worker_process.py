"""Private subprocess entry point; never starts Kit or imports a native main."""
import sys


class _NoSdkImports:
    """Fail visibly if an accidentally captured processor requests simulator SDK."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'omni', 'pxr', 'carb', 'isaacsim', 'isaac'}:
            raise ImportError(f'Simulator SDK import forbidden in observation worker: {fullname}')
        return None


def main():
    incoming, outgoing = sys.stdin.buffer, sys.stdout.buffer
    # Processor imports and computations may use print(); protocol owns stdout.
    sys.stdout = sys.stderr
    sys.meta_path.insert(0, _NoSdkImports())
    from .process_worker import _receive, _send
    try:
        command, process = _receive(incoming)
        if command != 'initialize' or not callable(process):
            raise ValueError('Expected one callable SDK-free processor')
    except Exception as exc:
        _send(outgoing, ('error', f'{type(exc).__name__}: {exc}'))
        return
    _send(outgoing, ('ready', None))
    while True:
        try:
            command, payload = _receive(incoming)
        except EOFError:
            return
        try:
            if command != 'compute':
                raise ValueError('Expected compute command')
            result = process(payload)
            _send(outgoing, ('result', result))
        except Exception as exc:
            _send(outgoing, ('error', f'{type(exc).__name__}: {exc}'))


if __name__ == '__main__':
    main()
