import os
import argparse
import torchaudio
import torch
from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
import tqdm
import json
from install_ffmpeg import install_ffmpeg
import sys

# AWS Sagemaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '.')
SM_CHANNEL_TRAINING = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
SM_CHANNEL_VALIDATION = os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation')
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def parse_args():
    parser = argparse.ArgumentParser(description='Training script arguments')
    
    # Add your training arguments here
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # Data directories
    parser.add_argument('--train_dir', type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument('--val_dir', type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument('--test_dir', type=str, default=SM_CHANNEL_TEST)
    parser.add_argument('--model_dir', type=str, default=SM_MODEL_DIR)

    return parser.parse_args()

def main():
    # Install FFMPEG if not already installed
    if not install_ffmpeg():
        print("Error installing FFMPEG. Please ensure it is installed correctly.")
        sys.exit(1)

    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    # Track initial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv= os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir= os.path.join(args.train_dir, 'train_splits'),
        dev_csv= os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir= os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv= os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir= os.path.join(args.test_dir, 'output_repeated_splits_test')
        batch_size=args.batch_size,
    )

    print(f"""Training CSV path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training video directory: {os.path.join(args.train_dir, 'train_splits')}""")

    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader=train_loader, val_loader=val_loader)
    best_validation_loss = float('inf')

    metrics_data = {
        'train_loss': [],
        'val_loss': [],
        'epochs': [],
    }

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate()

        # Track metrics
        metrics_data['train_loss'].append(train_loss["total_loss"])
        metrics_data['val_loss'].append(val_loss["total_loss"])
        metrics_data['epochs'].append(epoch)

        # Log metrics in SameMaker format
        print(json.dumps({
            "metrics": [
                {
                    "Name": "train:loss",
                    "Value": train_loss["total_loss"]
                },
                {
                    "Name": "validation:loss",
                    "Value": val_loss["total_loss"]
                },
                {
                    "Name": "validation:emotion_precision",
                    "Value": val_metrics["emotion_precision"]
                },
                {
                    "Name": "validation:emotion_accuracy",
                    "Value": val_metrics["emotion_accuracy"]
                },
                {
                    "Name": "validation:sentiment_precision",
                    "Value": val_metrics["sentiment_precision"]
                },
                {
                    "Name": "validation:sentiment_accuracy",
                    "Value": val_metrics["sentiment_accuracy"]
                }
            ]
        }))

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Save the best model
        if val_loss["total_loss"] < best_validation_loss:
            best_validation_loss = val_loss["total_loss"]
            print(f"New best model found at epoch {epoch}, saving model...")
            torch.save(model.state_dict(), os.path.join(args.model_dir, "best_model.pth"))
    
    # After training is complete, evaluate on the test set
    print("Evaluating on the test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase='test')
    metrics_data['test_loss'] = test_loss["total_loss"]

    print(json.dumps({
    "metrics": [
        {
            "Name": "test:loss",
            "Value": test_loss["total_loss"]
        },
        {
            "Name": "test:emotion_accuracy",
            "Value": test_metrics["emotion_accuracy"]
        },
        {
            "Name": "test:emotion_precision",
            "Value": test_metrics["emotion_precision"]
        },
        {
            "Name": "test:sentiment_accuracy",
            "Value": test_metrics["sentiment_accuracy"]
        },
        {
            "Name": "test:sentiment_precision",
            "Value": test_metrics["sentiment_precision"]
        }
    ]
}))