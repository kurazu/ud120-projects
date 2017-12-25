from nltk.stem.snowball import SnowballStemmer
import io
import string
import os.path
import text_learning
import re


def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated)

        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)

        """

    f.seek(0)  # go back to beginning of file (annoying)
    all_text = f.read()

    # split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        # remove punctuation
        text_string = "".join(
            char for char in content[1] if char not in string.punctuation
        )
        # text_string = content[1].translate(
        #     str.maketrans("", ""), string.punctuation
        # )

        # project part 2: comment out the line below

        # split the text string into individual words, stem each word,
        # and append the stemmed word to words (make sure there's a single
        # space between each stemmed word)
        stemmer = SnowballStemmer("english")
        pattern = re.compile(r'\s+')
        words = ' '.join(
            stemmer.stem(word) for word in pattern.split(text_string) if word
        )

    return words


def main():
    file_path = os.path.join(
        os.path.dirname(text_learning.__file__), 'test_email.txt'
    )
    with io.open(file_path, 'r', encoding='utf-8') as f:
        text = parseOutText(f)
    print(text)


if __name__ == '__main__':
    main()
