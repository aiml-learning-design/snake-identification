import os
import pandas as pd


def normalize(name):
    """Normalize names for matching"""
    return (name.lower()
            .replace("'", "")
            .replace("-", " ")
            .replace(",", "")
            .replace(".", "")
            .replace("  ", " ")
            .strip())


# Load metadata
metadata_df = pd.read_csv("data/metadata.csv")

# Base directories - corrected path joining
base_dir = "data/raw_images/"
venomous_dir = os.path.join(base_dir, "Venomous")
non_venomous_dir = os.path.join(base_dir, "Non-Venomous")

image_metadata = []

for venom_type, type_dir in [("Venomous", venomous_dir),
                             ("Non-Venomous", non_venomous_dir)]:

    if not os.path.exists(type_dir):
        print(f"[WARN] Directory not found: {type_dir}")
        continue

    for snake_name in os.listdir(type_dir):
        snake_dir = os.path.join(type_dir, snake_name)
        if not os.path.isdir(snake_dir):
            continue

        # Find matching species
        normalized_folder = normalize(snake_name)
        matched_rows = metadata_df[
            metadata_df["common_name"].apply(normalize) == normalized_folder
            ]

        if matched_rows.empty:
            print(f"[WARN] No match for: {snake_name}")
            continue

        # Process each image
        for img_file in os.listdir(snake_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                row = matched_rows.iloc[0].copy()
                row["image_path"] = os.path.join(type_dir, snake_name, img_file)
                row["venomous"] = venom_type == "Venomous"
                image_metadata.append(row)

# Save results
if image_metadata:
    output_df = pd.DataFrame(image_metadata)
    output_df.to_csv("data/image_metadata.csv", index=False)
    print(f"Saved {len(output_df)} image records")
else:
    print("No images processed - check your paths")
