from tracemeet.evaluation.asr_metrics import normalize_gold, normalize_hyp


def test_gold_markup():
    g = "bon + euh * @ {rire} j(e) suis [ok,oké] ex- $ Marie $ =n= est"
    assert normalize_gold(g) == "bon euh je suis ok marie est"


def test_hyp_numbers_and_punct():
    assert normalize_hyp("Il y a 22 personnes, c'est bien !") == \
        "il y a vingt deux personnes c' est bien"


def test_apostrophes_match():
    assert normalize_hyp("C’est") == normalize_gold("c'est")
