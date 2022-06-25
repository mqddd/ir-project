import math, os, hazm
from typing import List,Tuple

class BM25:

    def __init__(self):
        self.scores = []
        self.k = 2.0
        self.b = 0.75

    def read_train_files(self,dir):

        '''
            Description:
                This function reads train files

        '''
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_abs_dir = os.path.join(base_dir, dir)
        self.docs = []
        for path, subdirs, files in os.walk(dataset_abs_dir):
            for file in files:
                if file.endswith('.txt'):
                    with open(os.path.join(path, file), 'r', encoding='utf8') as f :
                        self.docs.append((file, f.read()))

        self.N = len(self.docs)
        return self

    def get_term_idf(self, term):
        '''
            Description:
                This function return idf value of a term

        '''
        df = 0
        for doc in self.docs :
            if term in hazm.word_tokenize(doc[1]) :
                df += 1

        idf = math.log10((self.N - df + 0.5)/(df + 0.5))
        return idf

    def get_tf(self, term, doc) :
        tf = 0
        for word in hazm.word_tokenize(doc) :
            if word == term :
                tf += 1
        return tf

    def calculate_doc_length(self):
        self.doc_length = []
        for doc in self.docs :
            doc_words = hazm.word_tokenize(doc[1])
            self.doc_length.append(len(doc_words))
        self.avg_doc_length = 0
        for length in self.doc_length :
            self.avg_doc_length += length
        self.avg_doc_length /= len(self.doc_length)

    def calculate_score(self, query):
        '''
            Description:
                This function calculates score of each doc according to
                the query 'query'

        '''
        self.scores = []
        query_terms = hazm.word_tokenize(query)
        idf_list = [self.get_term_idf(term) for term in query_terms]
        for i in range(len(self.docs)) :
            score = 0
            for t in range(len(query_terms)) :
                tf = self.get_tf(query_terms[t], self.docs[i][1])
                score += idf_list[t]*tf*(self.k + 1)/(tf + self.k*(1 - self.b + self.b * self.doc_length[i] / self.avg_doc_length))
            self.scores.append(score)


    def get_similar_docs(self,query)-> List[Tuple[str,int]]:

        '''
            Description:
                This function gets a query and ranks the dataset based on BM25 score sort by score
            output: a list of dicts [{"text": "document 1", "bm25_score": 1},
                                     {"text": "document 2", "bm25_score": 0.8}]
        '''
        output = []
        self.calculate_score(query)
        for i in range(len(self.scores)) :
            output.append({"text" : self.docs[i][0], "bm25_score" : self.scores[i]})

        sorted_output = sorted(output, key=lambda d:d["bm25_score"], reverse=True)
        return sorted_output

def read_query():
    return input('query >> ')

if __name__ == "__main__":
    dataset_dir = "./Dataset_IR/Train"

    bm25 = BM25().read_train_files(dataset_dir)
    bm25.calculate_doc_length()
    while True:
        query = read_query()
        print(bm25.get_similar_docs(query))


