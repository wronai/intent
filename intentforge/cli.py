import argparse
import json
import sys


def _validate_schemas() -> int:
    from jsonschema import Draft7Validator

    from .schema_registry import SCHEMAS

    for schema in SCHEMAS.values():
        Draft7Validator.check_schema(schema)
    return 0


def _validate_samples() -> int:
    return 0


def _dsl_run(args) -> int:
    """Run DSL from file or stdin"""
    from .dsl import DSLRunner

    if args.file:
        with open(args.file) as f:
            source = f.read()
    elif args.command:
        source = args.command
    else:
        source = sys.stdin.read()

    runner = DSLRunner()
    try:
        result = runner.run_sync(source)
        if result is not None:
            if isinstance(result, dict):
                print(json.dumps(result, indent=2, ensure_ascii=False))
            else:
                print(result)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _dsl_generate(args) -> int:
    """Generate code from DSL"""
    from .dsl import DSLCodeGenerator, DSLRunner

    if args.file:
        with open(args.file) as f:
            source = f.read()
    else:
        source = sys.stdin.read()

    runner = DSLRunner()
    program = runner.parse(source)
    generator = DSLCodeGenerator()

    if args.target == "python":
        print(generator.to_python(program))
    elif args.target == "shell":
        print(generator.to_shell(program))
    else:
        print(f"Unknown target: {args.target}", file=sys.stderr)
        return 1

    return 0


def _dsl_call(args) -> int:
    """Call a service action directly"""
    import asyncio

    from .services import services

    service = services.get(args.service)
    if service is None:
        print(f"Unknown service: {args.service}", file=sys.stderr)
        return 1

    method = getattr(service, args.action, None)
    if method is None:
        print(f"Unknown action '{args.action}' for service '{args.service}'", file=sys.stderr)
        return 1

    # Parse arguments
    kwargs = {}
    if args.args:
        try:
            kwargs = json.loads(args.args)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON arguments: {e}", file=sys.stderr)
            return 1

    # Call method
    import inspect

    try:
        result = method(**kwargs)
        if inspect.isawaitable(result):
            result = asyncio.run(result)

        if isinstance(result, dict):
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _dsl_services(args) -> int:
    """List available services and actions"""
    from .services import services

    if args.service:
        # Show actions for specific service
        svc = services.get(args.service)
        if svc is None:
            print(f"Unknown service: {args.service}", file=sys.stderr)
            return 1

        print(f"Service: {args.service}")
        print(f"Class: {svc.__class__.__name__}")
        print("\nActions:")
        for name in dir(svc):
            if not name.startswith("_"):
                method = getattr(svc, name)
                if callable(method):
                    import inspect

                    sig = inspect.signature(method)
                    params = ", ".join(
                        f"{p.name}={p.default!r}"
                        if p.default != inspect.Parameter.empty
                        else p.name
                        for p in sig.parameters.values()
                        if p.name != "self"
                    )
                    doc = method.__doc__ or ""
                    doc_line = doc.split("\n")[0].strip() if doc else ""
                    print(f"  {name}({params})")
                    if doc_line:
                        print(f"    {doc_line}")
    else:
        # List all services
        print("Available services:")
        for name in services.list():
            svc = services.get(name)
            doc = svc.__class__.__doc__ or ""
            doc_line = doc.split("\n")[0].strip() if doc else ""
            print(f"  {name}: {doc_line}")

    return 0


def _dsl_repl(args) -> int:
    """Interactive DSL REPL"""
    from .dsl import DSLRunner

    runner = DSLRunner()
    print("IntentForge DSL REPL")
    print("Type 'help' for commands, 'exit' to quit")
    print()

    while True:
        try:
            line = input("dsl> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

        if line in {"exit", "quit"}:
            break

        if line == "help":
            print("""
IntentForge DSL REPL - Interactive shell for service calls

SYNTAX:
  service.action(param="value", ...)   - Call a service action
  $var = service.action(...)           - Store result in variable
  $var.field                           - Access variable field

AVAILABLE SERVICES:
  chat      - LLM chat (send, models)
  analytics - Stats and NLP queries (stats, query, chart_data, products)
  voice     - Voice command processing (process)
  file      - File/image analysis (analyze, ocr, process_document, describe)
  data      - Data operations (list, get, create, update, delete)

COMMANDS:
  help      - Show this help
  services  - List all services
  vars      - Show stored variables
  exit      - Exit REPL

EXAMPLES:
  chat.models()
  chat.send(message="Cześć!")
  $result = chat.send(message="Hello")
  $result.response
  analytics.stats(period="today")
  voice.process(command="Włącz światło")
  file.ocr(image_base64="...")

NOTE: Use service.action() syntax, not natural language commands.
""")
            continue

        if line == "vars":
            for name, value in runner.get_variables().items():
                print(f"  ${name} = {value}")
            continue

        if line == "services":
            from .services import services

            for name in services.list():
                print(f"  {name}")
            continue

        # Check if user typed just a service name (common mistake)
        from .services import services as svc_registry

        available_services = svc_registry.list()
        if line in available_services:
            print(f"'{line}' is a service. Use: {line}.<action>()")
            print(f"Available actions for {line}:")
            service = svc_registry.get(line)
            if service:
                for method_name in dir(service):
                    if not method_name.startswith("_"):
                        method = getattr(service, method_name)
                        if callable(method):
                            print(f"  {line}.{method_name}()")
            continue

        try:
            result = runner.run_sync(line)
            if result is not None:
                if isinstance(result, dict):
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                else:
                    print(result)
        except SyntaxError as e:
            error_str = str(e)
            print(f"Syntax error: {error_str}")

            # Provide helpful hints based on error type
            if "Expected '.'" in error_str:
                # User typed something like "voice" without .action()
                word = line.split()[0] if line else ""
                if word in available_services:
                    print(f"Hint: '{word}' is a service. Try: {word}.send() or {word}.process()")
                else:
                    print('Hint: Use service.action() syntax, e.g.: chat.send(message="Hello")')
            elif "Expected '='" in error_str:
                print("Hint: For assignment use: $var = service.action()")
            else:
                print('Hint: Use service.action(args) syntax, e.g.: chat.send(message="Hello")')
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            if "Unknown service" in error_msg:
                print("Hint: Type 'services' to see available services")
            elif "Unknown action" in error_msg:
                print("Hint: Run 'intentforge services <name>' in shell to see actions")
            elif "connection" in error_msg.lower() or "connect" in error_msg.lower():
                print("Hint: Check if Ollama is running: ollama serve")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="intentforge", description="IntentForge CLI - NLP-driven code generation framework"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Schema validation
    sub.add_parser("validate-schemas", help="Validate JSON schemas")
    sub.add_parser("validate-samples", help="Validate sample files")

    # DSL commands
    dsl_run = sub.add_parser("dsl", help="Run DSL script")
    dsl_run.add_argument("-f", "--file", help="DSL file to run")
    dsl_run.add_argument("-c", "--command", help="DSL command to run")

    dsl_gen = sub.add_parser("dsl-gen", help="Generate code from DSL")
    dsl_gen.add_argument("-f", "--file", help="DSL file")
    dsl_gen.add_argument(
        "-t", "--target", choices=["python", "shell"], default="python", help="Target language"
    )

    dsl_call = sub.add_parser("dsl-call", help="Call service action directly")
    dsl_call.add_argument("service", help="Service name")
    dsl_call.add_argument("action", help="Action name")
    dsl_call.add_argument("args", nargs="?", help="JSON arguments")

    dsl_services = sub.add_parser("services", help="List available services")
    dsl_services.add_argument("service", nargs="?", help="Service name for details")

    sub.add_parser("repl", help="Interactive DSL REPL")

    args = parser.parse_args(argv)

    if args.command == "validate-schemas":
        return _validate_schemas()
    if args.command == "validate-samples":
        return _validate_samples()
    if args.command == "dsl":
        return _dsl_run(args)
    if args.command == "dsl-gen":
        return _dsl_generate(args)
    if args.command == "dsl-call":
        return _dsl_call(args)
    if args.command == "services":
        return _dsl_services(args)
    if args.command == "repl":
        return _dsl_repl(args)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
