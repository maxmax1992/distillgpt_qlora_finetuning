from typing import Tuple, Dict
import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset

class DocumentDataset(Dataset):
    def __init__(self, preprocessed_df, targets):
        self.targets = targets
        self.train_df = preprocessed_df
    
    def __getitem__(self, i):
        return dict(self.train_df.iloc[i, :]), self.targets[i]
    
    def __len__(self):
        return self.targets.size



    # Custom collate function that performs normalization for numerical features
def get_collate_fn_with_numerical_standardization(dataset_statistics):
    num_feats_means, num_feats_stds = dataset_statistics
    def collate_fn(batch):
        for input_obj, _ in batch:
            for key, val in input_obj.items():
                # print("Key with value type: ", type(val))
                if key in num_feats_means:
                    input_obj[key] = (float(val) - num_feats_means[key]) / num_feats_stds[key]
        return batch 
    return collate_fn


def get_dataset_statistics(df_preprocessed: pd.DataFrame) -> Tuple[Dict]:
    num_feats_means = dict(df_preprocessed[['days_since_update', 'n_versions', 'n_authors', 'n_categories']].mean())
    num_feats_stds = dict(df_preprocessed[['days_since_update', 'n_versions', 'n_authors', 'n_categories']].std())
    dataset_statistics = (num_feats_means, num_feats_stds)
    return dataset_statistics

def get_train_and_val_loaders(df_preprocessed: pd.DataFrame, \
                              targets: pd.DataFrame, train_frac: float,
                              batch_size: int) -> Tuple[DataLoader, DataLoader]:
    
    dataset = DocumentDataset(df_preprocessed, targets)
    dataset_statistics = get_dataset_statistics(df_preprocessed)
    n_train = int(train_frac*len(df_preprocessed)//1)
    n_val = len(df_preprocessed) - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])


    # Create DataLoader for each split
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, \
                            collate_fn=get_collate_fn_with_numerical_standardization(dataset_statistics))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, \
                            collate_fn=get_collate_fn_with_numerical_standardization(dataset_statistics))
    return train_loader, val_loader

