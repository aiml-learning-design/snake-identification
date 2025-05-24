import pandas as pd


class MetadataHandler:
    def __init__(self, metadata_path: str):
        self.metadata = pd.read_csv(metadata_path) # Path of image_metadata.csv

    def get_species_info(self, species_name: str) -> dict:
        row = self.metadata[self.metadata['species'].str.lower() == species_name.lower()]
        return row.to_dict(orient='records')[0] if not row.empty else {}

    def get_all_species(self):
        return self.metadata.groupby('species')['image_path'].apply(list).to_dict()

    def get_classes(self):
        return self.metadata['species'].unique()

  #  def get_label_encoder(self):
       # class_to_idx = {name: idx for idx, name in enumerate(self.get_classes())}
       # for tuple use list(enumerate(classes)) #output = [
        #   (0, 'king cobra'),
        #   (1, 'krait'),
        #   (2, 'russells viper'),
        #   (3, 'saw scaled viper'),
        #   (4, 'non-venomous')
        # ]
       # return class_to_idx


