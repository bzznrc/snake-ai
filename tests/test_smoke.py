from snake_ai.train.model import LinearQNet


def test_import_package():
    import snake_ai

    assert snake_ai.__version__


def test_model_forward_shape():
    model = LinearQNet(11, [32], 3)
    output = model.forward(__import__("torch").randn(11))
    assert output.shape[-1] == 3
