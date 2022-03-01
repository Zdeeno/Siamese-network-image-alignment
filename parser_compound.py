from torch.utils.data import Dataset
from parser_nordland import RectifiedNordland
from parser_eulongterm import RectifiedEULongterm


class CompoundDataset(Dataset):

    def __init__(self, crop_width, fraction, smoothness, nordland_path, eu_path):
        print("Creating Nordland dataset ...")
        self.nordland = RectifiedNordland(crop_width, fraction, smoothness, path=nordland_path)
        print("Nordland parsed \nCreating EU dataset ...")
        self.eu = RectifiedEULongterm(crop_width, fraction, smoothness, path=eu_path)
        print("EU dataset parsed!")

    def __len__(self):
        return len(self.nordland) + len(self.eu)

    def __getitem__(self, idx):
        if idx >= len(self.nordland):
            return self.eu[idx - len(self.nordland)]
        else:
            return self.nordland[idx]

