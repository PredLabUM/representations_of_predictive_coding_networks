def __getattr__(name):
    _submodules = {"load_dataset", "load_eeg", "training", "visualization", "analysis"}
    if name in _submodules:
        import importlib
        module = importlib.import_module(f".{name}", __package__)
        globals()[name] = module  # cache it so __getattr__ isn't called again
        return module
    raise AttributeError(f"module {__package__!r} has no attribute {name!r}")