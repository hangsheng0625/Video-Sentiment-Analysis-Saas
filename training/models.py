import torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze all BERT parameters so they are not updated during training
        for param in self.bert.parameters():
            param.requires_grad = False

        # Add a linear layer to project BERT's 768-dim output to 128-dim
        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # Extract  BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        pooler_output = outputs.pooler_output

        return self.projection(pooler_output)