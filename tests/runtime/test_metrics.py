from open_cortex.runtime.metrics import parse_prometheus_value


def test_parse_prometheus_value_reads_single_metric():
    text = """
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
"""

    value = parse_prometheus_value(text, "llamacpp:requests_processing")

    assert value == 1.0


def test_parse_prometheus_value_returns_none_when_missing():
    text = "llamacpp:requests_processing 1\n"

    value = parse_prometheus_value(text, "llamacpp:requests_deferred")

    assert value is None