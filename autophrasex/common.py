from naive_stopwords import Stopwords

STOPWORDS = Stopwords()


def ngrams(sequence, n=2):
    start, end = 0, 0
    while end < len(sequence):
        end = start + n
        if end > len(sequence):
            return
        yield (start, end), tuple(sequence[start: end])
        start += 1


if __name__ == "__main__":
    for window in ngrams('hello world', 2):
        print(window)
