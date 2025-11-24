import os

def rename_images(folderA, folderB):
    filesA = sorted(os.listdir(folderA))
    filesB = sorted(os.listdir(folderB))

    min_len = min(len(filesA), len(filesB))

    print(f"Found {len(filesA)} images in A")
    print(f"Found {len(filesB)} images in B")
    print(f"Renaming {min_len} paired images...")

    for idx in range(min_len):
        new_name = f"{idx:05d}.jpg"

        oldA = os.path.join(folderA, filesA[idx])
        oldB = os.path.join(folderB, filesB[idx])

        newA = os.path.join(folderA, new_name)
        newB = os.path.join(folderB, new_name)

        os.rename(oldA, newA)
        os.rename(oldB, newB)

    print("Renaming complete!")

if __name__ == "__main__":
    rename_images("train_A", "train_B")
