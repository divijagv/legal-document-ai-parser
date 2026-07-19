# ADK's `adk web` discovery expects `from . import agent` here. Guarded so
# that environments without google-adk installed (e.g. the CI job that only
# unit-tests the pure helpers in .validation) can still import this package.
try:
    from . import agent
except ModuleNotFoundError:
    agent = None
