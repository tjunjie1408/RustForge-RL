def test_import_core():
    import rustforge
    from rustforge import _core

    assert _core is not None
