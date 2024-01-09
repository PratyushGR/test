import sqlite3
import torch
import zlib
import numpy as np

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, db_file, label_type='label'):
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
        self.cursor.execute("SELECT Image, Label, GWlabels, ANAlabels FROM mindboggle101")
        self.data = self.cursor.fetchall()
        self.len = len(self.data)
        self.label_type = label_type

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = zlib.decompress(sample[0])

        if self.label_type == 'Label':
            label = zlib.decompress(sample[1])
            label_tensor = torch.from_numpy(np.frombuffer(label, dtype=np.float32).reshape((256, 256, 256)))
            return torch.from_numpy(np.frombuffer(image, dtype=np.float32).reshape((256, 256, 256))), label_tensor
        elif self.label_type == 'GWlabels':
            gw_labels = zlib.decompress(sample[2])
            gw_labels_tensor = torch.from_numpy(np.frombuffer(gw_labels, dtype=np.float32).reshape((256, 256, 256)))
            return torch.from_numpy(np.frombuffer(image, dtype=np.float32).reshape((256, 256, 256))), gw_labels_tensor
        elif self.label_type == 'ANAlabels':
            ana_labels = zlib.decompress(sample[3])
            ana_labels_tensor = torch.from_numpy(np.frombuffer(ana_labels, dtype=np.float32).reshape((256, 256, 256)))
            return torch.from_numpy(np.frombuffer(image, dtype=np.float32).reshape((256, 256, 256))), ana_labels_tensor

    def split_dataset(self):
        train_size = int(0.7 * self.len)
        valid_size = int(0.2 * self.len)
        train_data, valid_data, infer_data = torch.utils.data.random_split(self, [train_size, valid_size, self.len - train_size - valid_size])
        return train_data, valid_data, infer_data

    def get_loaders(self, batch_size=1, shuffle=True):
        train_data, valid_data, infer_data = self.split_dataset()
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        infer_loader = torch.utils.data.DataLoader(infer_data, batch_size=1, shuffle=False)
        return train_loader, valid_loader, infer_loader
