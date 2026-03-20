import nltk


def download_nltk_data():
    packages = ["punkt", "stopwords", "wordnet"]

    for pkg in packages:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
