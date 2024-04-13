import ner
import pandas as pd

if __name__ == '__main__':
    mentions = pd.read_pickle("../sentiment_dataset/mentions texts.pickle")
    sentiment = pd.read_pickle("../sentiment_dataset/sentiment_texts.pickle")
    sentiment = sentiment[sentiment["SentimentScore"] > 0]
    sentiment["SentimentScore"] -= 1
    preprocessor = ner.NER()
    sentiment = preprocessor.preprocessing_dataset(sentiment)
    successful = 0
    for idx, row in sentiment.iterrows():
        if row["issuerid"] == row["CompanyId"]:
            successful += 1
    print(successful / sentiment.index.nunique())
