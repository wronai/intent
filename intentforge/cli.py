import argparse


def _validate_schemas() -> int:
    from jsonschema import Draft7Validator

    from .schema_registry import SCHEMAS

    for schema in SCHEMAS.values():
        Draft7Validator.check_schema(schema)
    return 0


def _validate_samples() -> int:
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="intentforge")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-schemas")
    sub.add_parser("validate-samples")

    args = parser.parse_args(argv)

    if args.command == "validate-schemas":
        return _validate_schemas()
    if args.command == "validate-samples":
        return _validate_samples()

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
