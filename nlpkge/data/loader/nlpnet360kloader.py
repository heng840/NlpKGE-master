import os

from .baseloader import BaseLoader
from ..lut import LookUpTable


class NLPNET360KLoader(BaseLoader):
    def __init__(self, dataset_path, download=False):
        super().__init__(dataset_path, download,
                         raw_data_path="NLPNET360K/raw_data",
                         processed_data_path="NLPNET360K/processed_data",
                         train_name="train.txt",
                         valid_name="valid.txt",
                         test_name="test.txt",
                         data_name="NLPNET360K")
        self.node_lut_name = "node_lut.json"

    def download_action(self):
        self.downloader.NLPNET360K()

    def load_node_lut(self):
        preprocessed_file = os.path.join(self.processed_data_path, "node_lut.pkl")
        if os.path.exists(preprocessed_file):
            node_lut = LookUpTable()
            node_lut.read_from_pickle(preprocessed_file)
            # node_lut = pd.read_pickle(preprocessed_file)
        else:
            node_lut = LookUpTable()
            node_lut.add_vocab(self.node_vocab)
            node_lut.add_processed_path(self.processed_data_path)
            node_lut.save_to_pickle(preprocessed_file)
        return node_lut

    def load_all_lut(self):
        node_lut = self.load_node_lut()

        relation_lut = LookUpTable()
        relation_lut.add_vocab(self.relation_vocab)
        relation_lut.add_processed_path(self.processed_data_path)

        return node_lut, relation_lut
