import re
import string
import ssl
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('stopwords')
SW_ENG = stopwords.words('english')
 
# Make sure you have the necessary NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default to noun if no match


def lemmatize_text(words):
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    # Get POS tags for the words
    pos_tags = pos_tag(words)
    # Lemmatize each word with the appropriate POS tag
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in pos_tags]
    # Return the lemmatized words
    return lemmatized_words


def expand_contractions(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"can\'t", "can not", phrase, flags=re.IGNORECASE)

    # general
    phrase = re.sub(r"n\'t", " not", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\'re", " are", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\'s", " is", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\'d", " would", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\'ll", " will", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\'t", " not", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\'ve", " have", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\'m", " am", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"wanna", "want to", phrase, flags=re.IGNORECASE)

    return phrase


def remove_stopwords(tokens):
    return [w for w in tokens if not w in SW_ENG]


def tokenize_text(text):
    return word_tokenize(text)

# def lemmatize_text(tokens):    # commented out as it dose not work as intended
#     # Initialize the lemmatizer
#     lemmatizer = WordNetLemmatizer()
#     # Lemmatize each token
#     lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     # Return the lemmatized tokens
#     return lemmatized_tokens


def remove_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Substituting the matched URLs with an empty string
    return url_pattern.sub(' ', text)


def remove_img_and_gifs(text):
    # Regular expression to match the specific format ![gif](giphy|AgPt9udT567spxbSHf)
    pattern = r'!\[gif\]\(giphy\|[A-Za-z0-9]+\)'
    # Use re.sub to replace the matched patterns with an empty string
    return re.sub(pattern, ' ', text)


def remove_reddit_references(text):
    # Remove subreddit references
    text = re.sub(r'/r/\w+', ' ', text)
    # Remove user references
    text = re.sub(r'/u/\w+', ' ', text)
    return text


def remove_punctuation(text):
    # Remove images
    img_pattern = r"\!\[(?:img|gif)\]\([A-Za-z0-9]+\)"
    text = re.sub(img_pattern, r' ', text)

    # lowercase all text
    text = text.lower()

    # Separate closing/opening <<< E.G. )(, ](, etc >>> combinations of parenthesis
    parenthesis_mapping = {
        r'\)\(': ') (',
        r'\]\(': '] (',
        r'\)\[': ') [',
        r'\]\[': '] [',
        r'\}\(': '} (',
        r'\)\{': ') {',
        r'\}\{': '} {'
    }
    for parenthesis, replacement in parenthesis_mapping.items():
        text = re.sub(parenthesis, replacement, text)

    # # Substitute URLs with an empty string -> this is done in remove_urls function above
    # url_pattern = r'https?://\S+|www\.\S+'
    # text = re.sub(url_pattern, r' ', text)

    # Replace dotted acronyms with non-dotted acronyms (e.g. a.m.>>am, U.S.A>>USA )
    acronym_matches = re.findall(r"(?:\b\w\.)+", text)
    for match in acronym_matches:
        text = text.replace(match, match.replace(".", ""))

    # List of characters to replace with a space
    chars_to_replace = [
        '"', '#', '¢', '°', '³', 'µ', 'μ', '¹', 'º', 'ø', '⁻', '₂', '₄', 'ⓡ', '▪', '◊', '�',
        '(', ')', '<', '=', '@', '[', ']', '^', '_', '{', '}', '|', '°', '³', '¹', '–', '—',
        '†', '‡', '•', '⁽', '⁾', 'ⓒ', '▪', '◊', '✔', '\uf344', '�', '€', '%', '£', '…', 'ﬁ',
        '￼', '🫡', '%26', '✧', '\n'
    ]
    # Add all punctuation characters to the list of characters to replace
    chars_to_replace.extend(string.punctuation)
    # Replace each character in the list above with a space
    for char in chars_to_replace:
        text = text.replace(char, ' ')


    # Replace specific characters
    text = text.replace("’", "'").replace("‘", "'").replace("´", "'")
    text = text.replace("“", '"').replace("”", '"')

    # # Replace '' with nothing (including in abbreviations - don't>dont)  # commented out because not sure what this is meant to do
    # text = text.replace("'", "")

    # Replace diacritics with corresponding characters
    diacritics_mapping = {
        'à': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a', 'ā': 'a',
        'ç': 'c',
        'è': 'e', 'é': 'e',
        'í': 'i',
        'ñ': 'n',
        'ó': 'o', 'ô': 'o', 'ö': 'o',
        'ù': 'u', 'ü': 'u'
    }
    for diacritic, replacement in diacritics_mapping.items():
        text = text.replace(diacritic, replacement)

    # Remove special characters  - > uncomment if needed
    # Matches any character not in list, and any ':' that does not have numbers on either side
    text = re.sub(r"[^a-zA-Z0-9:%# ]|(?<!\d):(?!\d)", " ", text)

    # # Remove all digits -> uncomment if needed
    # text = re.sub(r'\d', ' ', text)

    # remove any repeating (more than twice) characters (ex. 'aaaaaaa' becomes 'aa')
    text = re.sub(r'(.)\1{2,}', lambda m: m.group(0)[:2], text)

    # Remove repeating spaces
    text = re.sub(' +', ' ', text)

    # Strip leading and trailing spaces
    text = text.strip()

    return text



def remove_emoticons(text):
    # Regular expression pattern for emoticons
    emoticon_pattern = re.compile(
        u'['
        u'\U0001F600-\U0001F64F'  # Emoticons
        u'\U0001F300-\U0001F5FF'  # Misc Symbols and Pictographs
        u'\U0001F680-\U0001F6FF'  # Transport and Map Symbols
        u'\U0001F1E0-\U0001F1FF'  # Flags (iOS)
        u'\U00002702-\U000027B0'  # Dingbats
        u'\U000024C2-\U0001F251'
        u'\U0001f926-\U0001f937'
        u'\U00010000-\U0010ffff'
        u'\u200d'  # Zero Width Joiner
        u'\u2640-\u2642'
        u'\u2600-\u2B55'
        u'\u23cf'
        u'\u23e9'
        u'\u231a'
        u'\u3030'
        u'\u23e9'
        u'\u1f918'
        u']+',
        re.UNICODE)

    # Substituting the emoticons with an empty string
    return emoticon_pattern.sub(r' ', text)


# Function to check if a string contains only English (ASCII) characters
def is_english(text):
    # Check if the text contains only ASCII characters (0-127)
    return all(ord(char) < 128 for char in text)