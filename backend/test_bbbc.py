from backend.app.datasets.bbbc_loader import load_bbbc001_images

data = load_bbbc001_images()

print("Loaded images:", len(data))

print("First image shape:", data[0]["image"].shape)
print("First filename:", data[0]["filename"])