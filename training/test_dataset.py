print("TEST SCRIPT STARTED")

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from training.dataset import DayNightDataset

print("Loading dataset...")
ds = DayNightDataset('.', transform=None)

print("Total images:", len(ds))

sample = ds[0]
print("Day shape:", sample['day'].shape)
print("Night shape:", sample['night'].shape)
