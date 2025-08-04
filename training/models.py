import torch
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
    

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Lower-level features
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            # Higher-level features
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        for param in self.conv_layers.parameters():
            param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # Remove the channel dimension
        x = x.squeeze(1)  

        features = self.conv_layers(x)


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoders for each modality
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # Final multimodal fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # Output a single sentiment score
        )

    def forward(self, input_ids, attention_mask, video_frames, audio_features):
        text_features = self.text_encoder(input_ids, attention_mask)
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate features from all modalities
        combined_features = torch.cat((text_features, video_features, audio_features), dim=1)

        return self.fusion_layer(combined_features)