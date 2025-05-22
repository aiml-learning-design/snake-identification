import pandas as pd


class MetadataHandler:
    def __init__(self, metadata_path: str):
        self.metadata = pd.read_csv(metadata_path) # Path of image_metadata.csv

    def get_species_info(self, species_name: str) -> dict:
        row = self.metadata[self.metadata['species'].str.lower() == species_name.lower()]
        return row.to_dict(orient='records')[0] if not row.empty else {}

    def get_all_species(self):
        return self.metadata.groupby('species')['image_path'].apply(list).to_dict()

    def get_venom_type(self, species_name: str):
        info = self.get_species_info(species_name)
        return info.get('venom_type', 'Unknown')

    def get_toxicity_level(self, species_name: str):
        info = self.get_species_info(species_name)
        return info.get('toxicity_level', 'Unknown')
