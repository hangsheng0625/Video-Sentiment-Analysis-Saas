import torch
from torch.utils.data import Dataset  
import pandas as pd                   
from transformers import AutoTokenizer 
import os                             
import cv2
import numpy as np

class MELDdataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)  
        self.video_dir = video_dir         
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  

        # Map emotion labels from text to integer indices for model training
        self.emotion_map = {
            "anger": 0,
            "disgust": 1,
            "sadness": 2,
            "joy": 3,
            "neutral": 4,
            "surprise": 5,
            "fear": 6
        }

        # Map sentiment labels from text to integer indices for model training
        self.sentiment_map = {
            "negative": 0,
            "positive": 1,
            "neutral": 2
        }

    def _load_video_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Try to read the first frame to ensure the video file is valid
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Could not read frame from video file: {video_path}")
            
            # Reset index to not skip the first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224)) 

                # Normalize the RGB channel to [0, 1]
                frame = frame / 255.0  
                frames.append(frame)
        
        except Exception as e:
            print(f"Error loading video frames: {e}")
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError(f"No frames extracted from video file: {video_path}")
        
        # Pad or truncate the frames to ensure we always have 30 frames
        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))  # Pad with zeros
        else:
            frames = frames[:30]

        # Before permute: [frames, height, width, channels]
        # After permute: [channels, frames, height, width]
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)  # Convert to (C, T, H, W) format

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
         # Get the data for the idx-th sample (row from CSV)
        row = self.data.iloc[idx]
        # Before permute: [frames, height, width, channels]
        # After permute: [channels, frames, height, width]
        return torch.FloatTensor(np.array(frames)).permute(3, 0, 1, 2)  # Convert to (C, T, H, W) format

    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, idx):
         # Get the data for the idx-th sample (row from CSV)
        row = self.data.iloc[idx] 
        
        # Construct the video filename using Dialogue_ID and Utterance_ID from the CSV row ("dia0_utt0.mp4")
        video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
        
        # Build the full path to the video file (if self.video_dir is "../dataset/dev/dev_splits_complete", 
        # the full path will be "../dataset/dev/dev_splits_complete/dia0_utt0.mp4")
        path = os.path.join(self.video_dir, video_filename)

        # Check if the video file exists at the constructed path
        video_path_exists = os.path.exists(path)  

        if video_path_exists == False:
            raise FileNotFoundError(f"Video file not found: {path}")
        
        text_input = self.tokenizer(
            row['Utterance'], 
            padding='max_length', 
            truncation=True, 
            max_length=128, 
            return_tensors='pt'
        )
        # print(row['Utterance'])
        # print(text_input)

        video_frames = self._load_video_frame(path)
        print(video_frames)

if __name__ == "__main__":
    meld = MELDdataset("../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete")
    print(meld[0])