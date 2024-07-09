import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load LLM Class
class LoadLLM:

    def __init__(self, model_name):

        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def main(self):

        return self.model
    
# Load file Class
class LoadFile:

    def __init__(self, file_path):
        
        self.file_path = file_path
        self.df_raw = pd.read_excel(self.file_path)
        
    def __problem_solution_creation(self):

        self.df_raw['DESCRIPTION'] = "The problem is identified as " + self.df_raw['PROBLEM'] + ". The possible solution is: " + self.df_raw['SOLUTION']

    def main(self):

        self.__problem_solution_creation()

        return self.df_raw
    
# Data Treatment Class
class DataTreatment:

    def __init__(self, df_raw):

        self.df_raw = df_raw
    
    def __index_creation(self):

        self.df_index = self.df_raw.copy()
        self.df_index['ID'] = np.arange(len(self.df_raw))
        self.df_index = self.df_index.set_index("ID", drop=False)

    def main(self):

        self.__index_creation()

        return self.df_index
    
# Vector Library Creation Class
class VectorLibraryCreation:

    def __init__(self, model, df_index, vector_library_path):

        self.model = model
        self.df_index = df_index
        self.vector_library_path = vector_library_path

    def __encoding_vectors(self):

        self.faiss_content_embedding = self.model.encode(self.df_index.DESCRIPTION.values.tolist())

    def __generate_embedding_vector(self):

        self.id_index = np.array(self.df_index.ID.values).flatten().astype("int")

        # Normalize the content
        self.content_encoded_normalized = self.faiss_content_embedding.copy()
        faiss.normalize_L2(self.content_encoded_normalized)

        # Creation of the Vector Library
        self.vector_library = faiss.IndexIDMap(faiss.IndexFlatIP(len(self.faiss_content_embedding[0])))
        self.vector_library.add_with_ids(self.content_encoded_normalized, self.id_index)

    def __export_faiss_vector_library(self):

        faiss.write_index(self.vector_library, f"{self.vector_library_path}.faiss")

    def main(self):

        self.__encoding_vectors()
        self.__generate_embedding_vector()
        self.__export_faiss_vector_library()

        return self.faiss_content_embedding, self.vector_library
    
class SearchContent:

    def __init__(self, df_index, model, vector_library_path):

        self.df_index = df_index
        self.model = model
        self.vector_library_path = vector_library_path

    def search(self, query, k=5):

        self.query_vector = self.model.encode([query])
        faiss.normalize_L2(self.query_vector)

        self.top_k = vector_library.search(self.query_vector, k)
        self.similarities = self.top_k[0][0].tolist()
        self.ids = self.top_k[1][0].tolist()
        self.results = self.df_index.loc[self.ids]
        self.results['SIMILARITY'] = self.similarities

        return self.results
    
# Run application
if __name__ == "__main__":
    
    model = LoadLLM("all-MiniLM-L6-v2").main()
    df_raw = LoadFile('data.xlsx').main()
    df_index = DataTreatment(df_raw).main()
    faiss_content_embedding, vector_library = VectorLibraryCreation(model, df_index, "model3Directory").main()
    search_content = SearchContent(df_index, model, "model3Directory")
    result = search_content.search('Science')
    result_after_thershold=result[result["SIMILARITY"]>0.5]
    print(result_after_thershold)