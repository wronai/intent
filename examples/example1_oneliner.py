#!/usr/bin/env python3
"""
Example 1: One-Liner API
========================

Najprostsze możliwe użycie IntentForge - jedna linia kodu do wygenerowania:
- API endpoint
- CRUD operacji
- Formularza z backendem

Uruchomienie:
    python example1_oneliner.py

Lub z Docker:
    docker-compose run --rm intentforge python examples/example1_oneliner.py
"""

from intentforge.simple import Forge, crud, form, generate, query, save


def main():
    print("=" * 60)
    print("IntentForge - Example 1: One-Liner API")
    print("=" * 60)

    # =========================================================================
    # 1. Najprostsza forma - jedna linia
    # =========================================================================
    print("\n1. Generate code from natural language:")
    print("-" * 40)

    code = generate("Create REST API endpoint to list products with pagination")
    print(code[:500] + "...\n")

    # =========================================================================
    # 2. CRUD w jednej linii
    # =========================================================================
    print("\n2. Generate complete CRUD:")
    print("-" * 40)

    files = crud("products")

    print(f"Generated {len(files)} files:")
    for name in files.keys():
        print(f"  - {name}")

    # Zapisz do plików
    paths = save(files, "generated/products")
    print(f"\nSaved to: {paths[0].rsplit('/', 1)[0]}/")

    # =========================================================================
    # 3. Formularz z integracją
    # =========================================================================
    print("\n3. Generate form with backend:")
    print("-" * 40)

    files = form(
        "newsletter",
        [
            {"name": "email", "type": "email", "required": True},
            {"name": "name", "type": "text"},
            {"name": "interests", "type": "select", "options": ["tech", "business", "design"]},
        ],
    )

    # Pokaż wygenerowany HTML
    print("Generated HTML form:")
    print(files["frontend_html"][:400] + "...")

    # =========================================================================
    # 4. Bezpieczne zapytanie SQL
    # =========================================================================
    print("\n4. Generate safe SQL query:")
    print("-" * 40)

    sql, params = query(
        "Pobierz aktywnych użytkowników posortowanych po dacie rejestracji", table="users"
    )
    print(f"SQL: {sql[:200]}...")

    # =========================================================================
    # 5. Fluent Builder API
    # =========================================================================
    print("\n5. Fluent Builder API:")
    print("-" * 40)

    # Dla bardziej złożonych przypadków
    result = (
        Forge("Create REST API for orders management")
        .platform("fastapi")
        .with_auth("jwt")
        .with_pagination(100)
        .fields(["customer_id", "total", "status", "items"])
        .constraint("Use async/await")
        .constraint("Include rate limiting")
        .generate()
    )

    print(f"Generated {len(result)} chars of code")
    print(result[:300] + "...")

    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
