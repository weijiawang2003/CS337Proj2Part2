import requests
from bs4 import BeautifulSoup


def scrape_recipe(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    parts = []
    title = soup.find("h1")

    if title:
        parts.append(f"Title: {title.get_text(strip=True)}")

    ingredients = soup.select("li.ingredients-item, span.ingredients-item-name, li.ingredient, li[itemprop='recipeIngredient']")
    
    if ingredients:
        parts.append("Ingredients:")
        for li in ingredients:
            text = li.get_text(" ", strip=True)
            if text:
                parts.append(f"- {text}")
    steps = soup.select("li.instructions-section-item, li.step, li[itemprop='recipeInstructions']")
    
    if steps:
        parts.append("Steps:")
        for i, li in enumerate(steps, start=1):
            text = li.get_text(" ", strip=True)
            if text:
                parts.append(f"{i}. {text}")


    if not ingredients and not steps:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        parts.extend(paragraphs[:20])


    return "\n".join(parts)