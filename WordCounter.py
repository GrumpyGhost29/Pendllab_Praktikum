import SetupChecker
from Preprocessor import preprocess
from Transcriber import Transcriber


def count_words(data : list[tuple[int, str]], words : list[str]) -> dict:
    result = {}

    print(f"COUNTING WORDS: {words} in transcript")
    failed_bigrams = []
    for target in words:
        result[target] = 0

    for bigram in data:
        if bigram[1] in words:
            result[bigram[1]] += bigram[0]
        else:
            failed_bigrams.append(bigram)

    print("COUNTER CAUGHT MISMATCHED ABBREVIATIONS")
    print(failed_bigrams)

    for word in result:
        print(f"{word}: {result[word]}")
    print("DONE")
    print("--------------------------------------------------------")

    return result


if __name__ == "__main__":
    # SetupChecker.check()
    # t = Transcriber()
    # text = t.transcribe("test_audio.mp3")
    text = preprocess(open("test.txt", "r").read())
    words_to_count = ["het", "ly", "mo", "eo", "ba", "ndb"]
    counters = count_words(text, words_to_count)
