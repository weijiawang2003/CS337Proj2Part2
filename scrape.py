# scrape.py

from __future__ import annotations

import json
from typing import Any, Dict, List, Set

from recipe_api import (
    parse_recipe_from_url,
    Recipe,
    Step,
    Ingredient,
    PRIMARY_METHODS,
)


def ingredient_to_dict(ing: Ingredient) -> Dict[str, Any]:
    return {
        "name": ing.name,
        "quantity": ing.quantity,
        "measurement": ing.unit,
        "descriptor": ing.descriptor,
        "preparation": ing.preparation,
    }


def step_to_dict(step: Step) -> Dict[str, Any]:
    primary = [m for m in step.methods if m in PRIMARY_METHODS]
    other = [m for m in step.methods if m not in PRIMARY_METHODS]

    duration = step.time.get("duration") if step.time else None
    time_obj = {"duration": duration}

    temp_obj = dict(step.temperature) if step.temperature else {}

    return {
        "step_number": step.step_number,
        "description": step.description,
        "ingredients": list(step.ingredients),
        "tools": list(step.tools),
        "methods": {
            "primary": primary,
            "other": other,
        },
        "time": time_obj,
        "temperature": temp_obj,
    }


def recipe_to_dict(recipe: Recipe) -> Dict[str, Any]:
    primary_global: Set[str] = set()
    other_global: Set[str] = set()

    for step in recipe.steps:
        for m in step.methods:
            if m in PRIMARY_METHODS:
                primary_global.add(m)
            else:
                other_global.add(m)

    return {
        "title": recipe.title,
        "ingredients": [ingredient_to_dict(ing) for ing in recipe.ingredients],
        "tools": list(recipe.tools),
        "methods": {
            "primary_cooking_methods": sorted(primary_global),
            "other_methods": sorted(other_global),
        },
        "steps": [step_to_dict(s) for s in recipe.steps],
    }


def scrape_recipe(url: str) -> str:
    recipe = parse_recipe_from_url(url)
    data = recipe_to_dict(recipe)
    return json.dumps(data, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python scrape.py <recipe_url>")
        raise SystemExit(1)

    url = sys.argv[1].strip()
    print(scrape_recipe(url))
