from ahocorasick import Automaton
import pandas as pd
import numpy as np
import regex as re


def find_word_indices(texts, words, name_to_id) -> (list, list):
    '''
    Возвращает список id словарных слов встреченных в тексте и сами слова
    :params
    '''
    # Initialize Aho-Corasick automaton
    automaton = Automaton()

    # Add words to the automaton
    for idx, word in enumerate(words):
        automaton.add_word(word, idx)

    # Build the automaton
    automaton.make_automaton()

    id_list = []
    word_list = []
    # Iterate over each text
    for text in texts:
        # мапим issuerid к синониму, который встретился в сообщении
        ids_found_in_text = set()
        words_found_in_text = set()
        # Find occurrences of words in the text
        for end_index, word_index in automaton.iter(text.lower()):
            if name_to_id[words[word_index]] not in ids_found_in_text:
                ids_found_in_text.add(name_to_id[words[word_index]])
                words_found_in_text.add(words[word_index])
        id_list.append(list(ids_found_in_text))
        word_list.append(list(words_found_in_text))

    return id_list, word_list


def process_all_messages(texts, synonyms):
    '''
    Обёртка над функцией find_word_indices
    :params
      texts: list[str] - текст сообщения
      synonyms: (list[list[str]]) синонимы для каждой компании
    '''

    name_to_id = dict()  # по синониму к названию компании находим issuerid

    for issuerid, company_synonyms in enumerate(synonyms):
        for word in company_synonyms:
            if not word in name_to_id.keys():
                name_to_id[word] = issuerid + 1

    words = np.concatenate([x.ravel() for x in synonyms])

    return find_word_indices(texts, words, name_to_id)


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


class NER:
    def __init__(self, synonyms_path='./new_names_and_synonyms.csv'):
        synonyms_db = pd.read_csv(synonyms_path)
        stripped_series = synonyms_db['EMITENT_FULL_NAME'].str.strip().str.lower().dropna()
        # Split, strip, exclude empty, and explode to individual elements
        self.synonyms = stripped_series.str.split(',').apply(
            lambda lst: np.array([item.strip() for item in lst if item.strip()])).values

    def preprocessing_dataset(self, df):
        # df["preprocessed_MessageText"] = df["MessageText"].apply(lambda text: preprocessing_text(text))
        df = df.dropna()
        df["CompanyId"], df["CompanyName"] = process_all_messages(df["MessageText"], self.synonyms)
        df = df.explode(["CompanyId", "CompanyName"])
        return df
