import matplotlib.pyplot as plt
from pathlib import Path
from imutils import paths

def plot_image_counts(path: Path):
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    image_count = {}

    for subdir in subdirs:
        subdir_images = list(sorted(paths.list_images(subdir)))
        image_count[subdir.name] = len(subdir_images)

    plt.bar(image_count.keys(), image_count.values())
    for i, (subdir, count) in enumerate(image_count.items()):
        plt.text(i, count + 3, str(count), ha='center')
    plt.title("Number of Images in Each Subdirectory")
    plt.xlabel("Subdirectories")
    plt.ylabel("Counts")
    plt.show()
