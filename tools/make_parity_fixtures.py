# /// script
# requires-python = ">=3.10"
# dependencies = ["laya", "numpy"]
# ///
"""Record what upstream Laya answers, for the Ruby test suite to match.

Everything here is pure Python: no weights are loaded and nothing is downloaded. The fixtures
pin language detection, email cleaning, routing, option rendering, the calibration arithmetic
and the embedding shortlist to upstream's own output, so a behavior change in either codebase
shows up as a failing Ruby test rather than a silent divergence.

Usage:
    uv run tools/make_parity_fixtures.py test/fixtures/parity
"""
import inspect
import json
import os
import sys

import laya
from laya import lang as L
from laya.agent import Agent
from laya.common import (QTYPES, confidence_from_probs, ece_score, render_criterion,
                         render_options, serialize_state, temp_bucket, clamp_temperature)
from laya.email import clean_email_body, email_state
from laya.router import Router, match_typed_decisions_workflow, normalise_name, _repo_str
from laya.shortlist import shortlist_choice, predict_shortlist

OUT = sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/parity"

TEXTS = [
    "", "12345 6789", "   ", "?!.,", "refund me", "Care este ora in Tokyo?",
    "The customer was charged twice and wants a refund.",
    "Please refund the duplicate charge on invoice 4411 today.",
    "We visited a cafe in Zurich and the naive assumption about the invoice was wrong, so please refund",
    "Le client a été facturé deux fois et demande un remboursement.",
    "Le client a ete facture deux fois et il demande un remboursement pour la facture",
    "Der Kunde wurde zweimal belastet und möchte eine Rückerstattung für die Rechnung",
    "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung die nicht korrekt ist",
    "El cliente fue cobrado dos veces y quiere que le devuelvan el dinero por la factura",
    "Me cobraron dos veces la factura, por favor devuelvanme el dinero hoy",
    "Fui cobrado duas vezes na fatura de março, quero o dinheiro de volta",
    "Voce pode me mandar a nota fiscal? Preciso dela hoje para o financeiro",
    "Deu erro 500 no endpoint de login depois do update, alguem pode ver isso",
    "Il cliente è stato addebitato due volte e vuole il rimborso della fattura",
    "Ho ricevuto la fattura sbagliata, vorrei il rimborso subito per favore",
    "De klant is twee keer in rekening gebracht en wil geld terug voor de factuur",
    "Gătește-mi o rețetă de sarmale de post pentru mâine.",
    "Am fost taxat de două ori pentru factura din luna martie și vreau banii",
    "Exportă APK-ul pentru Android și pune-l pe Drive ca să-l instalez.",
    "Cât e ora acum la Tokyo",
    "Klient został obciążony dwukrotnie i chce zwrot pieniędzy za fakturę",
    "Zákazníkovi byla částka účtována dvakrát a žádá o vrácení peněz",
    "Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım",
    "Khách hàng đã bị thu phí hai lần và muốn được hoàn tiền ngay",
    "Հայերեն", "ՀԱՅԵՐԵՆ", "։֊", "Հայերեն abc",
    "ग्राहक से दो बार शुल्क लिया गया और वह धनवापसी चाहता है।",
    "お客様は二重に請求されたため返金を希望しています。", "客户被重复扣款要求退款",
    "고객이 두 번 청구되어 환불을 원합니다", "تم خصم المبلغ مرتين من العميل ويريد استرداد الأموال",
    "வாடிக்கையாளரிடம் இருமுறை கட்டணம் வசூலிக்கப்பட்டது",
    "С клиента дважды сняли деньги и он хочет возврат",
    "ลูกค้าถูกเรียกเก็บเงินสองครั้งและต้องการเงินคืน",
    "Ο πελάτης χρεώθηκε δύο φορές και θέλει επιστροφή χρημάτων",
    "הלקוח חויב פעמיים ורוצה החזר כספי", "ᚠᚢᚦᚨᚱᚲ ᚷᛖᛒᛟ", "ＡＢＣ ｄｅｆ",
    "See github.com and user@acme.com for the invoice details we discussed",
    "Envie para financeiro@empresa.com.br o boleto v1.2.3 com.br",
    "Ich brauche die Rechnung",
]
STATES = [
    None, {}, [], "plain string", {"body": "charged twice", "n": 3}, {"a": {"b": ["deep"]}},
    ["x", {"y": "z"}], {"subject": "नमस्ते", "body": "ग्राहक से दो बार शुल्क लिया गया"},
    {"message": "I was charged twice, please refund."},
    {"message": "मुझसे दो बार शुल्क लिया गया"}, {"message": "Quero cancelar minha assinatura"},
    {"from": "a@b.com", "subject": "Fatura", "body": "Fui cobrado duas vezes, quero reembolso agora"},
    {"body": 42}, {"body": True}, {"body": None},
]
EMAILS = [
    "", None, "My account is locked.\nThis email is confidential and intended solely for the named addressee.\nPlease unlock it.",
    "My account is locked\nThis email is confidential and intended solely for the named addressee.\nPlease unlock it.",
    "My account is locked. This email is confidential and intended solely for the named addressee.",
    "My account is locked.\n\nThis email is confidential and intended solely for the named addressee.",
    "My account is locked.\n\nThis email and any files transmitted with it are\nconfidential and intended solely for the named addressee.",
    "Please reopen ticket 4411.\n\nIf you have received this message in error, delete it.",
    "Thanks for the update.\nOn Mon, Sep 20, Bob wrote:\n> original text",
    "Hi team,\nCan you confirm the refund?\nRegards,\nAlice",
    "Thanks for the update.\r\n-- \r\nBob\r\nSent from my iPhone",
    "Hello\\n> quoted\\n-----Original Message-----\\nFrom: x",
    "Preciso da nota fiscal.\nEm 12/09/2025, Maria Souza escreveu:\n> mensagem antiga",
    "Preciso da nota fiscal.\nEm resposta ao que você escreveu: seguem os dados",
    "Necesito la factura.\nEl 12/09/2025, Juan escribió:\n> mensaje anterior",
    "Segue o pedido.\n-----Mensagem original-----\nDe: alguem@x.com",
    "Bom dia, preciso do boleto.\nAtenciosamente,\nJoão",
    "Obrigado pelo retorno, mas ainda preciso do boleto de setembro para pagar hoje",
    "Preciso do boleto.\nEnviado do meu smartphone Samsung Galaxy.",
    "Enviado do meu celular o comprovante ontem. Preciso conferir o valor agora.",
    "Segue em anexo.\nDe: Maria Souza\nEnviado: segunda-feira, 12 de setembro\nPara: financeiro",
    "De: 10/09 a 15/09 estarei de férias, favor redirecionar os chamados para o suporte",
    "Preciso do contrato confidencial que discutimos ontem na reunião com o cliente",
    "Segue o contrato.\n\nEsta mensagem é confidencial e destinada exclusivamente ao destinatário.",
    "Antes de imprimir o boleto, confira o valor com o financeiro por favor",
    "Segue.\n\nPense no meio ambiente antes de imprimir esta mensagem.",
    "Long thread.\nOn Tue, Sep 21, 2025 at 4:12 PM Someone Very Long Name\nsomeone@example.com> wrote:\n> old",
    "x" * 4000,
]
QUESTIONS = {
    "choice_dict": {"type": "choice", "instructions": "Which team?",
                    "criteria": {"billing": {"desc": "payments"}, "tech": None, "sales": "", "zero": 0, "no": False}},
    "choice_list": {"type": "choice", "instructions": "Pick", "criteria": ["alpha", "beta"]},
    "score": {"type": "score", "instructions": "How urgent?", "criteria": [{"d": "low"}, "high", 2]},
    "noul_bare": {"type": "noul", "instructions": "Is it?"},
    "noul_dict": {"type": "noul", "instructions": "Is this phishing?",
                  "criteria": {"true": {"desc": "phishing, scam or fraud"}, "false": {"desc": "legitimate"}}},
    "noul_bool_keys": {"type": "noul", "instructions": "Is it?", "criteria": {True: "yes it is", False: "no"}},
    "instructions_not_str": {"type": "noul", "instructions": {"q": "é"}},
}
BAD_QUESTIONS = {
    "not_a_dict": "nope",
    "no_type": {"instructions": "x"},
    "unknown_type": {"type": "essay", "instructions": "x"},
    "no_instructions": {"type": "noul"},
    "choice_no_criteria": {"type": "choice", "instructions": "x"},
    "choice_empty": {"type": "choice", "instructions": "x", "criteria": {}},
    "score_not_list": {"type": "score", "instructions": "x", "criteria": {"a": 1}},
    "score_empty": {"type": "score", "instructions": "x", "criteria": []},
    "noul_bad_criteria": {"type": "noul", "instructions": "x", "criteria": ["a"]},
}
TD = {
    "agent_trace_observability": ["action", "needs_review", "outcome", "risk", "urgency"],
    "customer_service": ["action", "category", "churn_risk", "needs_human", "urgency"],
    "invoice_processing": ["discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"],
    "security_incidents": ["credential_compromise", "disposition", "severity", "true_positive", "urgency"],
}
Q_GENERIC = {"dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "tech": None}}}
ROUTE_CALLS = [
    ({"body": "I was charged twice, please refund."}, "generic", {}),
    ({"body": "Հայերեն"}, "generic", {}), ({"body": "Հայերեն"}, "generic", {"model": "english"}),
    ({"body": "मुझसे दो बार शुल्क लिया गया"}, "generic", {}),
    ({"body": "二重に請求されました"}, "generic", {}), ({"body": "두 번 청구되었습니다"}, "generic", {}),
    ({"body": "تم خصم المبلغ مرتين"}, "generic", {}),
    ({"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die Rechnung"}, "generic", {}),
    ({"body": "anything"}, "generic", {"model": "multilingual"}),
    ({"body": "मुझसे दो बार"}, "generic", {"model": "english"}),
    ({"body": "x"}, "generic", {"task": "typed_decisions"}), ({"body": "x"}, "generic", {"task": "typed-decisions"}),
    ({"body": "मुझसे दो बार"}, "generic", {"lang": "en"}), ({"body": "मुझसे दो बार"}, "generic", {"lang": "en_US.UTF-8"}),
    ({"body": "hello there"}, "generic", {"lang": "de"}), ({"body": "hello"}, "generic", {"lang": ""}),
    ({"body": "I was charged twice"}, "td", {}),
    ({}, "generic", {}), (None, "generic", {}), ("12345", "generic", {}),
    ("Quero cancelar", "generic", {}), ("Esqueci minha senha", "generic", {}),
    ("Gătește-mi o rețetă de sarmale de post pentru mâine.", "generic", {}),
    ("Müşteriden iki kez ücret alındı ve para iadesi istiyor lütfen yardım", "generic", {}),
    ("Please refund the duplicate charge on invoice 4411 today.", "generic", {}),
    ("refund me", "generic", {}),
    ({"body": "hello"}, "generic", {"lang_guess": "pt"}), ({"body": "olá"}, "generic", {"lang_guess": "en"}),
    ({"body": "hello"}, "generic", {"lang_guess": "zzz"}),
]
SHORTLIST_VECTORS = {
    "pay me": [1.0, 0.0], "alpha": [1.0, 0.0], "beta": [0.0, 1.0],
    "gamma: mid": [0.6, 0.8], "delta: same": [1.0, 0.0],
}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def lang_fixture():
    rows = []
    for text in TEXTS:
        rows.append({"text": text, "script": L.detect_script(text),
                     "profile": {k: round(v, 10) for k, v in L.script_profile(text).items()},
                     "guess": L.guess_latin_language(text), "is_english": L.is_english(text),
                     "analyse": jsonable(L.analyse(text))})
    states = [{"state": jsonable(s), "text": L.state_text(s), "analyse": jsonable(L.analyse(s))} for s in STATES]
    return {"texts": rows, "states": states,
            "long_state_text_length": len(L.state_text("x" * 5000))}


def email_fixture():
    rows = [{"body": body, "clean": clean_email_body(body)} for body in EMAILS]
    states = [
        {"args": ["  Subject  ", "Body\nRegards,\nBob", "a@b.c"], "kwargs": {"account_tier": "enterprise", "empty": None},
         "state": email_state("  Subject  ", "Body\nRegards,\nBob", "a@b.c", account_tier="enterprise", empty=None)},
        {"args": ["s", "Body\nRegards,\nBob", None], "kwargs": {"clean": False},
         "state": email_state("s", "Body\nRegards,\nBob", None, clean=False)},
        {"args": [None, None, None], "kwargs": {}, "state": email_state(None, None, None)},
    ]
    return {"bodies": rows, "states": states,
            "max_chars": {"body": "x" * 4000, "length": len(clean_email_body("x" * 4000)),
                          "custom": len(clean_email_body("x" * 10, max_chars=5))}}


def questions_fixture():
    rows = {}
    for name, qdef in QUESTIONS.items():
        internal = Agent._to_internal(qdef)
        rows[name] = {"definition": jsonable(qdef), "internal_type": internal["t"],
                      "instructions": internal["ins"], "options": render_options(internal)}
    errors = {}
    for name, qdef in BAD_QUESTIONS.items():
        try:
            Agent._check_question(name, qdef)
            errors[name] = None
        except ValueError as e:
            errors[name] = str(e)
    criteria = {"str": render_criterion("phishing or scam"), "dict": render_criterion({"desc": "phishing"}),
                "list": render_criterion(["a", "b"]), "int": render_criterion(3),
                "bool": render_criterion(False), "nonascii": render_criterion({"d": "münchen"})}
    return {"questions": rows, "errors": errors, "criteria": criteria,
            "serialize_state": {json.dumps(jsonable(s)): serialize_state(s) for s in
                                ["plain", {"a": 1, "b": [True, None, "x"]}, ["x", {"y": "z"}]]}}


def calibration_fixture():
    confidences = []
    for probs, k in [([1.0], 1), ([0.5, 0.5], 2), ([1.0, 0.0], 2), ([0.7, 0.2, 0.1], 3),
                     ([0.4, 0.3, 0.3], 3), ([0.25] * 4, 4), ([1.0, 0.0, 0.7], 2)]:
        import numpy as np
        confidences.append({"p": probs, "k": k, "confidence": confidence_from_probs(np.array(probs), k)})
    import numpy as np
    eces = []
    for conf, correct, bins in [([0.9, 0.9], [1, 0], 15), ([0.0, 1.0], [0, 1], 15),
                                ([0.1, 0.2, 0.9], [0, 1, 1], 10), ([], [], 15),
                                ([0.0666, 0.0667], [1, 0], 15)]:
        value = ece_score(np.array(conf, dtype=float), np.array(correct, dtype=float), bins)
        eces.append({"conf": conf, "correct": correct, "bins": bins,
                     "ece": None if value != value else value})
    buckets = [{"qtype": t, "k": k, "bucket": temp_bucket(QTYPES[t], k)}
               for t in QTYPES for k in (1, 2, 3, 5, 6, 10, 11, 13, 77)]
    clamps = [{"input": repr(v), "clamped": clamp_temperature(v)} for v in
              [0.1006, 0.10058280825614929, 1.7601518630981445, 1.0, 9.0, 0.0, -3.0, None, "x",
               float("nan"), float("inf"), "1.5", True]]
    return {"confidence": confidences, "ece": eces, "buckets": buckets, "clamps": clamps}


def router_fixture():
    questions = {"generic": Q_GENERIC, "td": {i: {"type": "noul", "instructions": "x"} for i in TD["customer_service"]}}
    rows = []
    router = Router()
    auto = Router(auto_task_detection=True)
    standalone = Router(standalone_repos=True)
    local = Router(models={"english": "/tmp/en", "multilingual": "/tmp/ml"})
    for state, qname, kwargs in ROUTE_CALLS:
        for label, r in [("default", router), ("auto", auto), ("standalone", standalone), ("local", local)]:
            decision = r.route(state, questions[qname], **kwargs)
            rows.append({"router": label, "state": jsonable(state), "questions": qname,
                         "kwargs": jsonable(kwargs), "decision": jsonable(dict(decision))})
    defaults = {"max_loaded": Router().max_loaded, "default": Router().default}
    custom_default = Router(default="multilingual").route("12345", Q_GENERIC)
    workflows = {name: match_typed_decisions_workflow({i: {} for i in ids}) for name, ids in TD.items()}
    workflows["partial"] = match_typed_decisions_workflow({"urgency": {}, "category": {}})
    workflows["superset"] = match_typed_decisions_workflow({i: {} for i in TD["customer_service"] + ["extra"]})
    workflows["empty"] = match_typed_decisions_workflow({})
    aliases = {a: normalise_name(a) for a in
               ["en", "laya", "multi", "ML", "typed", "typed_decisions", "English", " laya ", "decisions"]}
    repos = {"root": _repo_str(("convaiinnovations/laya", None)),
             "sub": _repo_str(("convaiinnovations/laya", "multilingual")),
             "plain": _repo_str("some/repo")}
    return {"routes": rows, "defaults": defaults, "workflows": workflows, "aliases": aliases,
            "repo_str": repos, "custom_default": jsonable(dict(custom_default))}


def presets_fixture():
    return {name: jsonable(getattr(laya, name)()) for name in
            ["triage_questions", "email_questions", "guard_questions", "moderation_questions",
             "router_questions"]} | {"email_questions_custom": jsonable(
                 laya.email_questions({"vip": "important", "rest": "everyone else"}))}


def shortlist_fixture():
    class Table:
        def __init__(self, vectors):
            self.vectors = vectors
            self.calls = []

        def __call__(self, texts):
            self.calls.append(list(texts))
            return [self.vectors[t] for t in texts]

    criteria = {"alpha": None, "beta": "", "gamma": "mid", "delta": "same"}
    rows = []
    for k in (1, 2, 3, 4, 20):
        embed = Table(SHORTLIST_VECTORS)
        labels = shortlist_choice("pay me", criteria, embed, k=k)
        rows.append({"k": k, "labels": labels, "calls": embed.calls})

    zero = Table({**SHORTLIST_VECTORS, "pay me": [0.0, 0.0], "delta: same": [3.0, 4.0]})
    rows.append({"k": 2, "labels": shortlist_choice("pay me", criteria, zero, k=2), "calls": zero.calls,
                 "note": "zero query"})
    nan = Table({"pay me": [1.0, 0.0], "alpha": [float("nan")] * 2, "beta": [1.0, 0.0]})
    rows.append({"k": 1, "labels": shortlist_choice("pay me", {"alpha": None, "beta": None}, nan, k=1),
                 "calls": nan.calls, "note": "nan vector"})

    class Recorder:
        def __init__(self):
            self.questions = None

        def predict(self, state, questions, **kwargs):
            self.questions = jsonable(questions)
            return {"model": "fake", "answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}}

    full = {"billing": {"desc": "payments"}, "tech": "bugs", "sales": None, "other": "misc"}
    vectors = {"Which desk?\nI was charged twice": [1.0, 0.0], 'billing: {"desc": "payments"}': [0.0, 1.0],
               "tech: bugs": [1.0, 0.0], "sales": [0.2, 0.2], "other: misc": [0.0, 1.0]}
    agent = Recorder()
    out = predict_shortlist(agent, "I was charged twice",
                            {"intent": {"type": "choice", "instructions": "Which desk?", "criteria": full}},
                            Table(vectors), k=2)
    return {"cases": rows, "predict": {"shortlist": jsonable(out["shortlist"]),
                                       "questions_seen": agent.questions}}


os.makedirs(OUT, exist_ok=True)
fixtures = {
    "lang": lang_fixture(), "email": email_fixture(), "questions": questions_fixture(),
    "calibration": calibration_fixture(), "router": router_fixture(),
    "presets": presets_fixture(), "shortlist": shortlist_fixture(),
}
for name, data in fixtures.items():
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"laya_version": laya.__version__, "data": data}, f, indent=1, ensure_ascii=False)
    print("wrote", path)
