import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel
import torch.nn as nn
from final_solution.absa.constants import *


class TransformerClassificationModel(nn.Module):
    """Надстройка над предобученой LLM, добавление классификационной головы с несколькими линейными слоями"""

    def __init__(self, base_transformer_model: str, num_classes: int, num_dense_layers: int):
        super(TransformerClassificationModel, self).__init__()
        config = AutoConfig.from_pretrained(base_transformer_model)

        self.backbone = AutoModel.from_pretrained(base_transformer_model, config=config)

        layers = []
        input_size = self.backbone.config.hidden_size
        for _ in range(num_dense_layers - 1):
            layers.append(nn.Linear(input_size, input_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(input_size, num_classes))
        self.classifier = nn.Sequential(*layers)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        probabilities = self.softmax(logits)
        return {'logits': logits, 'probabilities': probabilities, 'backbone outputs': outputs, }


def freeze_backbone_function(model: TransformerClassificationModel, freeze=True):
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    return model


def preprocess_data(tokenizer, texts, aspects):
    inputs = tokenizer([wrap(text, aspect) for text, aspect in zip(texts, aspects)],
                            max_length=256, truncation=True, padding=True, return_tensors="pt")
    return inputs

def evaluate(model, tokenizer, dataset, device):
    model.eval()
    test_texts, test_aspects = dataset["text"], dataset["aspect"]
    test_inputs = preprocess_data(tokenizer, test_texts, test_aspects)
    test_data = torch.utils.data.TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'])
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)

    return predictions





