import os

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


def load_input_files(input_files, callback):
    if not input_files:
        return
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            continue
        lino = 0
        with open(f, mode='rt', encoding='utf8') as fin:
            for line in fin:
                line = line.rstrip('\n')
                if callback:
                    callback(line, lino)
                lino += 1


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)


def uniform_chinese_text(text):
    if not text:
        return ""
    text = text.lower()
    return "".join([Q2B(c) for c in text])


if __name__ == "__main__":
    for window in ngrams('hello world', 2):
        print(window)
