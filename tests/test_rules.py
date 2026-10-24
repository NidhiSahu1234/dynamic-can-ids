import tempfile
from pathlib import Path

from backend.app.services.rules import RuleService


def test_add_list_remove_rule():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        svc = RuleService(db)
        rule = svc.add_rule("abc|dlc=8|00:00:00:00:00:00:00:00", "dos", status="confirmed")
        assert len(svc.list_rules()) == 1
        assert svc.is_confirmed_attack(rule.signature)
        assert svc.remove_rule(rule.rule_id)
        assert not svc.is_confirmed_attack(rule.signature)


def test_suspected_rule_not_in_bloom_confirmed():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        svc = RuleService(db)
        sig = "018f|dlc=8|fe:5b:00:00:00:3c:00:00"
        svc.add_rule(sig, "fuzzy", status="suspected")
        assert sig not in svc.get_confirmed_signatures()
