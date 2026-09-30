from popline.stance import messages, parse, quote_check


def test_prompt_has_rule_and_text_but_no_label():
    m = messages("Paneuropa ist eine Utopie.")
    assert m[0]["role"] == "system" and "The voice that counts is the text's" in m[1]["content"]
    assert "Paneuropa ist eine Utopie." in m[1]["content"]


def test_parse_accepts_fenced_json_and_rejects_bad_stance():
    ans, err = parse('```json\n{"stance": "contra", "reason": "r", "quote": "q"}\n```')
    assert err is None and ans["stance"] == "CONTRA"
    assert parse('{"stance": "UNSURE"}')[1].startswith("stance not one of")
    assert parse("I think it is positive.")[1] == "no JSON object"


def test_quote_check():
    text = "Das Europa von gestern meint man im allgemeinen zu kennen."
    assert quote_check("Europa von  gestern", text) == "exact"
    assert quote_check("Das Europa von gesterm meint man", text) == "close"
    assert quote_check("Die Vereinigten Staaten von Europa", text) == "not_found"
    assert quote_check("", text) == "empty"
