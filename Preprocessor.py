import re
from text_to_num import alpha2digit

def preprocess(text : str) -> list[tuple[int, str]]:
    print("PREPROCESSING")
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    text = re.sub(r'\s+', ' ', text)  # replace whitespace runs with one space
    text = text.strip() # Strip excess whitespace
    text = text.lower() # lowercase all
    text = alpha2digit(text, "de") # replace number words above 2
    text = re.sub(r'zwei', '2', text)
    text = re.sub(r'ein|eine', '1', text)
    words = text.split() # split into tokens

    choice_bigrams = [] # create list of token tuples
    failed_bigrams = [] # save failed bigrams for debugging

    for i in range(0, len(words)-1, 1):
        try:
            choice_bigrams.append((int(words[i]), words[i+1]))
        except ValueError:
            try:
                (int(words[i-1]), words[i])
            except ValueError:
                if i-1 >= 0:
                    failed_bigrams.append((words[i-1],words[i], words[i+1], "at pos" + str(i)))
                else:
                    failed_bigrams.append((words[i],words[i+1], "at pos" + str(i)))

    print("PREPROCESSOR COULDN'T PARSE:")
    print(failed_bigrams)
    print("PREPROCESSOR PARSED:")
    print(choice_bigrams)
    print("DONE")
    print("--------------------------------------------------------")

    return choice_bigrams
