import collections
import re
import unicodedata


GENERIC_PREFIXES = (
    "ngay",
    "nam",
    "co",
    "la",
)

RANKING_PREFIXES = (
    "top",
    "nhom",
)

GENERIC_NUMERIC_SUFFIXES = (
    "cai",
    "chiec",
)


def _normalize_unicode(text):
    return unicodedata.normalize("NFKC", text or "")


def _strip_accents(text):
    decomposed = unicodedata.normalize("NFD", text)
    filtered = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return filtered.replace("đ", "d").replace("Đ", "D")


def _strip_parenthetical_content(text):
    return re.sub(r"\([^)]*\)", " ", text)


def _strip_number_separators(text):
    # Treat "5.281" and "5,281" as the same surface form for QA answers.
    return re.sub(r"(?<=\d)[\.,](?=\d)", "", text)


def _remove_punctuation(text):
    text = text.replace("_", " ")
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)


def _white_space_fix(text):
    return " ".join(text.split())


def _lower(text):
    return text.lower()


def normalize_answer(text):
    normalized = _normalize_unicode(text)
    normalized = _strip_parenthetical_content(normalized)
    normalized = _strip_number_separators(normalized)
    normalized = _remove_punctuation(normalized)
    normalized = _lower(normalized)
    normalized = _strip_accents(normalized)
    normalized = _white_space_fix(normalized)
    return normalized


def _strip_generic_prefixes(text):
    changed = True
    while changed:
        changed = False
        for prefix in GENERIC_PREFIXES:
            new_text = re.sub(rf"^{prefix}\s+(.+)$", r"\1", text)
            if new_text != text:
                text = new_text
                changed = True
    return text


def _canonicalize_numeric_surface(text):
    for prefix in RANKING_PREFIXES:
        text = re.sub(rf"^{prefix}\s+(\d+)$", r"\1", text)

    for suffix in GENERIC_NUMERIC_SUFFIXES:
        text = re.sub(rf"^(\d+)\s+{suffix}$", r"\1", text)
        text = re.sub(rf"^(?:co\s+)?(\d+)\s+{suffix}$", r"\1", text)

    return text


def _strip_trailing_context(text):
    # "khoa kinh te thuoc dai hoc quoc gia ha noi" -> "khoa kinh te"
    text = re.sub(r"^(khoa\s+.+?)\s+thuoc\s+.+$", r"\1", text)
    return text


def canonicalize_answer(text):
    canonical = normalize_answer(text)
    canonical = _strip_generic_prefixes(canonical)
    canonical = _canonicalize_numeric_surface(canonical)
    canonical = _strip_trailing_context(canonical)
    return _white_space_fix(canonical)


def get_tokens(text, canonical=False):
    if not text:
        return []
    normalized = canonicalize_answer(text) if canonical else normalize_answer(text)
    return normalized.split()


def compute_exact(a_gold, a_pred, canonical=False):
    normalizer = canonicalize_answer if canonical else normalize_answer
    return int(normalizer(a_gold) == normalizer(a_pred))


def compute_f1(a_gold, a_pred, canonical=False):
    gold_toks = get_tokens(a_gold, canonical=canonical)
    pred_toks = get_tokens(a_pred, canonical=canonical)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_recall(a_gold, a_pred, canonical=False):
    gold_toks = get_tokens(a_gold, canonical=canonical)
    pred_toks = get_tokens(a_pred, canonical=canonical)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0:
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0.0

    return 1.0 * num_same / len(gold_toks)


def aggregate_mean(values):
    return sum(values) / len(values) if values else 0.0
