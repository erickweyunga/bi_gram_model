import os
import string

import nltk
from nltk.corpus import reuters

nltk.download('reuters')

# os.mkdir("data")

for field in reuters.fileids():
    article_text = ' '.join(reuters.words(field))

    filename = f'data/{field.replace("/", "_")}.txt'

    with open(filename, 'w', encoding="utf-8") as f:
        f.write(article_text)
