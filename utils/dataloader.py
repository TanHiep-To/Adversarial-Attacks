import os
import numpy as np
import pandas as pd
import argparse

config = {
    'train_url': "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv" ,
    'test_url': "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
    'data_dir': '../data',
    'batch_size': 16,
}

class DatasetClassify:
    
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.batch_size = config['batch_size']

    def download(self):
        self.train_df = pd.read_csv(config['train_url'], header=None,names=["label", "title", "description"])
        self.test_df  = pd.read_csv(config['test_url'], header=None,names=["label", "title", "description"])
        
        data_dir = self.data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Classify and save the dataset into directories based on class
        for label in self.train_df['label'].unique():
            class_dir = os.path.join(data_dir, f"class_{label}")
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            class_data = self.train_df[self.train_df['label'] == label]
            class_data.to_csv(os.path.join(class_dir, f"class_{label}.csv"), index=False)

            self.train_df['text'] = self.train_df['title'] + " " + self.train_df['description']
            self.test_df['text'] = self.test_df['title'] + " " + self.test_df['description']
            return self.train_df, self.test_df
        
def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='Directory to store the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    args = parser.parse_args()
    return args

def main():
    args = argument_parser()
    dataset = DatasetClassify(args.data_dir)
    print('Downloading and Preprocessing the dataset')
    dataset.download()
    print("Data downloaded successfully")
    
if __name__ == '__main__':
    main()