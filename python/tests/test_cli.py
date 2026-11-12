"""
Unit tests for MobileLLM CLI command parsing and execution.
"""

from mobilellm.cli.main import create_parser, main


def test_cli_parser_commands() -> None:
    parser = create_parser()

    # status
    args = parser.parse_args(["status"])
    assert args.command == "status"

    # models
    args = parser.parse_args(["models"])
    assert args.command == "models"

    # load
    args = parser.parse_args(["load", "test.gguf", "--context-len", "4096"])
    assert args.command == "load"
    assert args.model_path == "test.gguf"
    assert args.context_len == 4096

    # unload
    args = parser.parse_args(["unload", "m1"])
    assert args.command == "unload"
    assert args.model_id == "m1"

    # generate
    args = parser.parse_args(["generate", "Hello world", "--max-tokens", "64", "--temp", "0.5"])
    assert args.command == "generate"
    assert args.prompt == "Hello world"
    assert args.max_tokens == 64
    assert args.temp == 0.5

    # benchmark
    args = parser.parse_args(["benchmark", "--model", "test.gguf", "--runs", "5", "--tokens", "32"])
    assert args.command == "benchmark"
    assert args.model == "test.gguf"
    assert args.runs == 5
    assert args.tokens == 32

    # serve
    args = parser.parse_args(["serve", "--port", "9000", "--allow-remote-connections"])
    assert args.command == "serve"
    assert args.port == 9000
    assert args.allow_remote_connections is True


def test_cli_status_native() -> None:
    ret = main(["--native", "status"])
    assert ret == 0


def test_cli_models_native() -> None:
    ret = main(["--native", "models"])
    assert ret == 0


def test_cli_security_violation_serve() -> None:
    # Attempting to serve on 0.0.0.0 without API key must return error code 2
    ret = main(["serve", "--host", "0.0.0.0", "--allow-remote-connections"])
    assert ret == 2

