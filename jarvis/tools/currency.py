import requests


def schema() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": (
                "Convert money between currencies using latest exchange rates. "
                "Use this for GBP, USD, EUR, CNY, JPY and other currency conversions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount of money to convert."},
                    "from_currency": {"type": "string", "description": "Source currency code, for example GBP."},
                    "to_currency": {"type": "string", "description": "Target currency code, for example CNY."},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
        },
    }


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()

    try:
        response = requests.get(
            "https://api.frankfurter.dev/v2/rates",
            params={"base": from_currency, "quotes": to_currency},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()

        rate = data.get("rates", {}).get(to_currency)
        if rate is None:
            return {"ok": False, "error": f"No rate returned for {from_currency} to {to_currency}."}

        converted = float(amount) * float(rate)

        return {
            "ok": True,
            "amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "rate": rate,
            "converted": round(converted, 2),
            "date": data.get("date"),
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
