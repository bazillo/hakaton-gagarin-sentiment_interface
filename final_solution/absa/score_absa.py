from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, TrainingArguments, Trainer, \
    pipeline
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from datasets import Dataset
import torch.nn.functional as F
from torch import nn
import torch
import pandas as pd
import final_solution.ner.ner as ner

mentions = pd.read_pickle("../sentiment_dataset/mentions texts.pickle")
sentiment = pd.read_pickle("../sentiment_dataset/sentiment_texts.pickle")
sentiment = sentiment[sentiment["SentimentScore"] > 0]
sentiment["SentimentScore"] -= 1
preprocessor = ner.NER()
sentiment = preprocessor.preprocessing_dataset(sentiment)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
df_step2 = sentiment[sentiment["issuerid"] == sentiment["CompanyId"]]

data = {
    'text': df_step2["MessageText"],
    'aspect': df_step2["CompanyName"],
    'label': df_step2["SentimentScore"]
}


df = pd.DataFrame(data)
df = df.reset_index()
df = df.drop(["index"], axis=1)

dataset = Dataset.from_pandas(df,  preserve_index=False)
dataset = dataset.train_test_split(test_size=0.1)
