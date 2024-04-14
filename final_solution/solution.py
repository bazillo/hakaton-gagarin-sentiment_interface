import pathlib
import typing as tp

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from final_solution.absa.model import TransformerClassificationModel, evaluate
from final_solution.ner import ner

EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,


#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]


def to_df(texts: tp.Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame({"MessageText": texts})


def score_text(message: str, preprocessor, model, tokenizer, device):
    sentiment = to_df(message)
    sentiment = preprocessor.preprocessing_dataset(sentiment)
    data = {
        'text': sentiment["MessageText"],
        'aspect': sentiment["CompanyName"]
    }

    df = pd.DataFrame(data)
    df = df.reset_index()
    df = df.drop(["index"], axis=1)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    return list(map(lambda x: (int(x[0]), float(x[1])),
                    list(zip(sentiment['CompanyId'], evaluate(model, tokenizer, dataset, device)))))


def score_texts(
        messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """
    # preprocessing

    path_to_solution = pathlib.Path(__file__).parent
    preprocessor = ner.NER(f"{path_to_solution}/ner/new_names_and_synonyms.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rubert_tiny = "cointegrated/rubert-tiny2"
    model = TransformerClassificationModel(rubert_tiny, num_classes=5, num_dense_layers=2).to(device)
    tokenizer_tiny = AutoTokenizer.from_pretrained(rubert_tiny)
    model.load_state_dict(torch.load(f"{path_to_solution}/absa/rubert_tiny_2fc", map_location=torch.device(device)))
    return [score_text([message], preprocessor, model, tokenizer_tiny, device) for message in messages]