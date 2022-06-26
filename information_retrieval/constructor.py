import contextlib
from .inverted_index import InvertedIndexIterator, InvertedIndexWriter, InvertedIndexMapper
import pickle as pkl
import os
from .helper import IdMap
from typing import *
from hazm import word_tokenize, Normalizer

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map IdMap: For mapping terms to termIDs
    doc_id_map(IdMap): For mapping relative paths of documents (eg path/to/docs/in/a/dir/) to docIDs
    data_dir(str): Path to data
    output_dir(str): Path to output index files
    index_name(str): Name assigned to index
    postings_encoding: Encoding used for storing the postings.
        The default (None) implies UncompressedPostings
    """

    def __init__(self, data_dir, output_dir, index_name="BSBI",
                 postings_encoding=None):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Stores names of intermediate indices
        self.intermediate_indices = []

    def save(self):
        """Dumps doc_id_map and term_id_map into output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pkl.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pkl.dump(self.doc_id_map, f)

    def load(self):
        """Loads doc_id_map and term_id_map from output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pkl.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pkl.load(f)

    def index(self):
        """Base indexing code

        This function loops through the data directories,
        calls parse_block to parse the documents
        calls invert_write, which inverts each block and writes to a new index
        then saves the id maps and calls merge on the intermediate indices
        """
        for block_dir_relative in sorted(next(os.walk(self.data_dir))[1]):
            print(f'Process {block_dir_relative}...')
            td_pairs = self.parse_block(block_dir_relative)
            print('Block parsed! Writing invert...')
            index_id = 'index_' + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, directory=self.output_dir,
                                     postings_encoding=
                                     self.postings_encoding) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
        print('Processing block finished! Saving...')
        self.save()
        print('Blocks saved. Merging...')
        with InvertedIndexWriter(self.index_name, directory=self.output_dir,
                                 postings_encoding=
                                 self.postings_encoding) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexIterator(index_id,
                                          directory=self.output_dir,
                                          postings_encoding=
                                          self.postings_encoding))
                    for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)
        print('Merged!')

    def parse_block(self, block_dir_relative):
        """Parses a tokenized text file into termID-docID pairs

        Parameters
        ----------
        block_dir_relative : str
            Relative Path to the directory that contains the files for the block

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block

        Should use self.term_id_map and self.doc_id_map to get termIDs and docIDs.
        These persist across calls to parse_block
        """
        ### Begin your code
        all_td_pairs = []
        for block_file in os.listdir(os.path.join(self.data_dir, block_dir_relative)):
            doc_full_path = os.path.join(self.data_dir, block_dir_relative, block_file)
            with open(doc_full_path, 'r', encoding='utf-8') as f:
                doc_text = f.read()
            doc_text = Normalizer().normalize(doc_text)
            td_pairs = []
            for token in word_tokenize(doc_text):
                td_pairs.append((self.term_id_map[token], self.doc_id_map[doc_full_path]))
            all_td_pairs.extend(td_pairs)
        return all_td_pairs
        ### End your code

    def invert_write(self, td_pairs, index: InvertedIndexWriter):
        """Inverts td_pairs into postings_lists and writes them to the given index

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index on disk corresponding to the block
        """
        ### Begin your code
        sorted_td_pairs = sorted(td_pairs, key=lambda x: (self.term_id_map[x[0]], x[1]))
        last_term_id = None
        last_doc_id = None
        cur_postings_list = []
        counter = 0
        for term_id, doc_id in sorted_td_pairs:
            if counter % 100 == 0:
                print(f'Create post list {counter}/{len(sorted_td_pairs)}')
            counter += 1
            if term_id != last_term_id and cur_postings_list != []:
                index.append(last_term_id, cur_postings_list)
                cur_postings_list = []

            if (term_id, doc_id) != (last_term_id, last_doc_id):
                cur_postings_list.append(doc_id)
            
            last_term_id, last_doc_id = term_id, doc_id
        if cur_postings_list != []:
            index.append(last_term_id, cur_postings_list)
        ### End your code

    def merge(self, indices: List[InvertedIndexIterator], merged_index: InvertedIndexWriter):
        """Merges multiple inverted indices into a single index

        Parameters
        ----------
        indices: List[InvertedIndexIterator]
            A list of InvertedIndexIterator objects, each representing an
            iterable inverted index for a block
        merged_index: InvertedIndexWriter
            An instance of InvertedIndexWriter object into which each merged
            postings list is written out one at a time
        """
        ### Begin your code
        indices_head = [next(index) for index in indices]
        while any([head is not None for head in indices_head]):
            min_term = min([self.term_id_map[head[0]] for head in indices_head if head is not None])
            min_term_id = self.term_id_map[min_term]
            postings_set = set()
            for i in range(len(indices_head)):
                while indices_head[i] is not None and min_term_id == indices_head[i][0]:
                    postings_set.update(indices_head[i][1])
                    try:
                        indices_head[i] = next(indices[i])
                    except StopIteration:
                        indices_head[i] = None
            merged_index.append(min_term_id, sorted(list(postings_set)))
        ### End your code

    def retrieve(self, query: AnyStr):
        """
        use InvertedIndexMapper here!
        Retrieves the documents corresponding to the conjunctive query

        Parameters
        ----------
        query: str
            Space separated list of query tokens

        Result
        ------
        List[str]
            Sorted list of documents which contains each of the query tokens.
            Should be empty if no documents are found.

        Should NOT throw errors for terms not in corpus
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        ### Begin your code
        query = Normalizer().normalize(query)
        token_ids= [self.term_id_map[token] for token in word_tokenize(query)]
        with InvertedIndexMapper(self.index_name, self.postings_encoding, self.output_dir) as index:
            res_posting_list = index[token_ids[0]]
            for i in range(1, len(token_ids)):
                token_posting_list = index[token_ids[i]]
                res_posting_list = sorted_intersect(res_posting_list, token_posting_list)
            return [self.doc_id_map[doc_id] for doc_id in res_posting_list]
        ### End your code


def sorted_intersect(list1: List[Any], list2: List[Any]):
    """Intersects two (ascending) sorted lists and returns the sorted result

    Parameters
    ----------
    list1: List[Comparable]
    list2: List[Comparable]
        Sorted lists to be intersected

    Returns
    -------
    List[Comparable]
        Sorted intersection
    """
    ### Begin your code
    counter1 = 0
    counter2 = 0
    intersect = []
    while counter1 < len(list1) and counter2 < len(list2):
        if list1[counter1] == list2[counter2]:
            intersect.append(list1[counter1])
            counter1 += 1
            counter2 += 1
        elif list1[counter1] > list2[counter2]:
            counter2 += 1
        else:
            counter1 += 1
    return intersect
    ### End your code
