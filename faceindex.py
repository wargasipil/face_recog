import faiss
import os
import pickle
import time

class FaceIndex:
    
    index: faiss.IndexFlatL2
    filename: str
    label_map: dict

    def __init__(self, filename: str = "face_embedding.index", embedding_dim = 512) -> None:
        self.filename = filename
        filename_map = filename + ".map_label"
        

        if os.path.exists(filename):
            self.index = faiss.read_index(filename)
            if os.path.exists(filename_map):
                with open(filename_map, "rb") as out: 
                    self.label_map = pickle.load(out)
            else:
                self.label_map = {}
        else:
            index = faiss.IndexFlatL2(embedding_dim)
            self.index = faiss.IndexIDMap(index)
            self.label_map = {}

    def add(self, embedding, label):
        id = int(time.time())
        self.index.add_with_ids(embedding, id)
        self.label_map[id] = label
        self.save()

    def save(self):
        faiss.write_index(self.index, self.filename)
        filename_map = self.filename + ".map_label"
        with open(filename_map, "wb") as out: 
            pickle.dump(self.label_map, out)

    def search(self, query_embedding, k = 5):
        distances, indices = self.index.search(query_embedding, k)  # Search for k nearest neighbors
        labels = []
        for indice in indices[0]:
            labels.append(self.label_map.get(indice, "unknown"))

        return distances, indices, labels