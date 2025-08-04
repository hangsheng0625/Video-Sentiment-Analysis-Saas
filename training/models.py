import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
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
    

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)

        # Freeze all backbone parameters to use it as a fixed feature extractor
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the number of features output by the backbone's final fully connected layer
        num_features = self.backbone.fc.in_features

        # Replace the final classification layer with a new head:
        # - Linear layer to project features to 128 dimensions
        # - ReLU activation for non-linearity
        # - Dropout for regularization
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )  

    def forward(self, x):
        # [Batch size, frames, channels, height, width] => [Batch size, channels, frames, height, width]
        x = x.transpose(1, 2)  
        return self.backbone(x)