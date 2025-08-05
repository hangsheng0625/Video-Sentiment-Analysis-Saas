import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import meld_dataset
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
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7)  # Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise
        )

        # Classification heads
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # Negative, Neutral, Positive
        )


    def forward(self, text_inputs, video_frames, audio_features):
        text_features = self.text_encoder(
            text_inputs['input_ids'], 
            attention_mask=text_inputs['attention_mask']
        )

        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)

        # Concatenate features from all modalities
        combined_features = torch.cat((text_features, video_features, audio_features), dim=1)

        fused_features = self.fusion_layer(combined_features)
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)

        return {'emotion': emotion_output, 'sentiment': sentiment_output}
    

class MultimodalTrainer(nn.Module):
    def __init__(self, model, train_loader, val_loader):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Log dataset sizes
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        print("\nDataset sizes:")
        print(f"Training samples: {train_size:,}")
        print(f"Validation samples: {val_size:,}")
        print(f"Batches per epoch: {len(train_loader):,}")

        # Very high:1 , high:0.1-0.01, medium:1e-1, low:1e-4, very low:1e-5
        self.optimizer = torch.optim.AdamW([
            {'params': model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': model.fusion_layer.parameters(), 'lr': 5e-4},
            {'params': model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': model.sentiment_classifier.parameters(), 'lr': 5e-4}
        ], weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.1, 
            patience=2
        )

        self.emotion_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.sentiment_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    def train_epoch(self):
        self.model.train()
        running_loss = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}

        for batch in self.train_loader:
            device = next(self.model.parameters()).device
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }   

            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_labels'].to(device)
            sentiment_labels = batch['sentiment_labels'].to(device)
            
            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(text_inputs, video_frames, audio_features)

            # Compute losses using raw digits
            emotion_loss = self.emotion_criterion(outputs['emotion'], emotion_labels)
            sentiment_loss = self.sentiment_criterion(outputs['sentiment'], sentiment_labels)

            total_loss = emotion_loss + sentiment_loss

            # Backward pass and calculate gradients
            total_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track the losses
            running_loss['total'] += total_loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

        return {k: v / len(self.train_loader) for k, v in running_loss.items()}

    def validate(self):
        self.model.eval()
        val_loss = {'total': 0.0, 'emotion': 0.0, 'sentiment': 0.0}
        all_emotions_predictions = []
        all_emotions_labels = []
        all_sentiments_predictions = []
        all_sentiments_labels = []

        with torch.inference_mode():
            for batch in self.val_loader:
                device = next(self.model.parameters()).device
                text_inputs = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device)
                }   

                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_labels'].to(device)
                sentiment_labels = batch['sentiment_labels'].to(device)

                # Forward pass
                outputs = self.model(text_inputs, video_frames, audio_features)

                # Compute losses
                emotion_loss = self.emotion_criterion(outputs['emotion'], emotion_labels)
                sentiment_loss = self.sentiment_criterion(outputs['sentiment'], sentiment_labels)

                total_loss = emotion_loss + sentiment_loss

                
if __name__ == "__main__":
    dataset = meld_dataset.MELDdataset(
        "../dataset/train/train_sent_emo.csv",
        "../dataset/train/train_splits"
    )

    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()

    text_inputs = 