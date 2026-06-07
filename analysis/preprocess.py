import re
import unicodedata
from collections import Counter


LETTER_PATTERN = re.compile(r"[^a-z]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"[\n\r.!?;:]+")


IMPORTANT_WORDS = {
    "pt": {"nao", "sim", "muito", "pouco", "mais", "menos", "mas", "sem", "contra", "sobre"},
    "en": {"no", "not", "yes", "very", "little", "more", "less", "but", "without", "against", "about"},
    "es": {"no", "si", "muy", "poco", "mas", "menos", "pero", "sin", "contra", "sobre"},
}


STOPWORDS = {
    "pt": {
        "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "ate", "com",
        "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "do", "dos", "e",
        "ela", "elas", "ele", "eles", "em", "entre", "era", "eram", "essa", "essas", "esse", "esses",
        "esta", "estao", "estas", "estava", "estavam", "este", "estes", "eu", "foi", "foram", "ha",
        "isso", "isto", "ja", "lhe", "lhes", "me", "mesmo", "meu", "meus", "minha", "minhas", "na",
        "nas", "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os",
        "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "porque", "qual", "quando", "que",
        "quem", "se", "sem", "seu", "seus", "sua", "suas", "tambem", "te", "tem", "tendo", "ter",
        "teu", "teus", "tua", "tuas", "um", "uma", "voce", "voces",
    },
    "en": {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
        "me", "more", "most", "my", "myself", "nor", "of", "off", "on", "once", "only", "or", "other",
        "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some",
        "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
        "with", "you", "your", "yours", "yourself", "yourselves",
    },
    "es": {
        "a", "al", "algo", "algunas", "algunos", "ante", "antes", "aquel", "aquella", "aquellas",
        "aquello", "aquellos", "aqui", "arriba", "asi", "atras", "aun", "aunque", "bajo", "bien",
        "cada", "casi", "como", "con", "contra", "cual", "cuando", "de", "del", "desde", "donde",
        "dos", "durante", "e", "el", "ella", "ellas", "ellos", "en", "entre", "era", "eran", "eres",
        "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estaban", "estado", "estais",
        "estamos", "estan", "estar", "estas", "este", "esto", "estos", "estoy", "fue", "fueron", "ha",
        "hace", "hacen", "hacer", "hacia", "han", "hasta", "hay", "la", "las", "le", "les", "lo",
        "los", "mas", "me", "mi", "mis", "mucha", "muchas", "mucho", "muchos", "muy", "nada", "ni",
        "nos", "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "o", "os", "otra",
        "otras", "otro", "otros", "para", "pero", "poco", "por", "porque", "que", "quien", "se", "sea",
        "segun", "ser", "si", "sido", "siendo", "sin", "sobre", "sois", "somos", "son", "soy", "su",
        "sus", "tambien", "te", "teneis", "tenemos", "tener", "tengo", "ti", "tiene", "tienen", "todo",
        "todos", "tu", "tus", "un", "una", "unas", "uno", "unos", "vosotras", "vosotros", "y", "ya",
        "yo",
    },
}


def strip_diacritics(text):
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_surface(text):
    ascii_text = strip_diacritics(text)
    return LETTER_PATTERN.sub(" ", ascii_text).strip()


def tokenize(text):
    return [token for token in normalize_surface(text).split() if len(token) > 1]


def normalize_term(text):
    tokens = tokenize(text)
    if len(tokens) != 1:
        return None
    return tokens[0]


def filter_tokens(tokens, language):
    stopwords = STOPWORDS.get(language, STOPWORDS["pt"])
    important = IMPORTANT_WORDS.get(language, IMPORTANT_WORDS["pt"])
    return [token for token in tokens if token in important or token not in stopwords]


def preprocess_text(text, language, max_tokens):
    sentences = []
    total_raw_tokens = 0
    total_clean_tokens = 0
    truncated = False

    for raw_sentence in SENTENCE_SPLIT_PATTERN.split(text):
        raw_tokens = tokenize(raw_sentence)
        if not raw_tokens:
            continue
        total_raw_tokens += len(raw_tokens)
        clean = filter_tokens(raw_tokens, language)
        if len(clean) < 2:
            continue
        remaining = max_tokens - total_clean_tokens
        if remaining <= 0:
            truncated = True
            break
        if len(clean) > remaining:
            clean = clean[:remaining]
            truncated = True
        sentences.append(clean)
        total_clean_tokens += len(clean)
        if total_clean_tokens >= max_tokens:
            truncated = True
            break

    return {
        "sentences": sentences,
        "raw_tokens": total_raw_tokens,
        "clean_tokens": total_clean_tokens,
        "truncated": truncated,
        "vocabulary": Counter(token for sentence in sentences for token in sentence),
    }


def apply_vocab_cap(sentences, vocabulary, max_vocab, required_terms):
    kept = set(required_terms)
    for token, _count in vocabulary.most_common(max_vocab):
        kept.add(token)
        if len(kept) >= max_vocab:
            break
    filtered = [[token for token in sentence if token in kept] for sentence in sentences]
    filtered = [sentence for sentence in filtered if len(sentence) >= 2]
    return filtered, Counter(token for sentence in filtered for token in sentence)
