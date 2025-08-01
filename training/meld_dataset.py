from torch.utils.data import Dataset  
import pandas as pd                   
from transformers import AutoTokenizer 
import os                             
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

if __name__ == "__main__":
    meld = MELDdataset("../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete")
    hello = ["a", "b", "c"] 