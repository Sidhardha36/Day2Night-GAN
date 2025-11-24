from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms

class DayNightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        self.day_dir = os.path.join(root_dir, "train_A")
        self.night_dir = os.path.join(root_dir, "train_B")

        day_files = {f for f in os.listdir(self.day_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
        night_files = {f for f in os.listdir(self.night_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))}

        # Only keep matching paired filenames
        self.files = sorted(list(day_files.intersection(night_files)))

        print("Found", len(self.files), "valid paired images")

        # Default transform (optimized for CPU)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((150, 150)),      # faster than 256
                transforms.RandomCrop(128),         # final image size
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        day_img = Image.open(os.path.join(self.day_dir, fname)).convert("RGB")
        night_img = Image.open(os.path.join(self.night_dir, fname)).convert("RGB")

        day_img = self.transform(day_img)
        night_img = self.transform(night_img)

        return {"day": day_img, "night": night_img}
