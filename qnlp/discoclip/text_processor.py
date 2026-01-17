"""
Convert a sentence into an einsum expression and a list of tensor ids. 
The pipeline involves the following steps:
1. Tokenize the sentence.
2. Lemmatize the tokens.
3. Parse the sentence with BobcatParser or TreeParser.
4. Apply the MPSAnsatz to obtain a tensor network diagram.
5. Convert the diagram to an einsum expression and a list of symbols with their shapes.

Before training, we need to gather the symbols and their shapes from the training data.
The symbols will be used to initialize the tensors in the model.
"""
from itertools import count
from typing import Optional

import opt_einsum as oe
from lambeq.backend.tensor import Diagram, Swap, Cup, Cap, Spider
from lambeq import CCGTree, TreeReader
from lambeq import BobcatParser, TensorAnsatz, TreeReaderMode, Rewriter

from nltk import pos_tag
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Tokenizer:
    """
    A simple tokenizer and lemmatizer using NLTK's Treebank
    tokenizer and WordNet lemmatizer.
    """
    def __init__(self):
        self.word_tokenizer = TreebankWordTokenizer()
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, sentence):
        tokens = self.word_tokenizer.tokenize(sentence)
        return tokens
    
    def lemmatize(self, tokens):
        pos_tags = pos_tag(tokens)
        lemmas = [
            self.lemmatizer.lemmatize(token.lower(), pos=self._to_wordnet_pos(tag))
            for token, tag in pos_tags
        ]
        return lemmas

    def _to_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def __call__(self, sentence):
        """
        Tokenize and lemmatize a sentence.
        """
        tokens = self.tokenize(sentence)
        lemmas = self.lemmatize(tokens)
        return lemmas


def union_find(merges: list[tuple[int, int]]):
    """
    Given a list of merges, return a dictionary which maps
    an edge to its representative.
    """
    parent = {i: i for merge in merges for i in merge}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for x, y in merges:
        union(x, y)

    return {i: find(i) for i in parent}


def tn_to_einsum(diag: Diagram, interleaved: bool = False):
    """Convert a diagram to an einsum string expression,
    and a list of tensors to contract.
    arguments:
        interleaved: if True, return the data for the
        interleaved mode of einsum, i.e. a list of interleaved
        tensors and their indices, and finally the dangling
        indices. E.g. for the einsum string 'abc,cd->abd', the
        interleaved data would be:
        [tensor_1, [0, 1, 2], tensor_2, [2, 3], [0, 1, 3]]
    """
    idx_gen = count(0)

    merges = []
    tensors = []
    tensor_edges = [] 
    size_dict = {}

    def get_new_index(size):
        """Generate a new index and record its size"""
        new_index = next(idx_gen)
        size_dict[new_index] = size
        return new_index

    inputs = [get_new_index(size) for size in diag.dom.dim]
    scan = inputs[:]

    for layer in diag.layers:
        l, box, _ = layer.unpack()

        if isinstance(box, Swap):
            scan[len(l)], scan[len(l) + 1] = scan[len(l) + 1], scan[len(l)]

        elif isinstance(box, Cup):
            merges.append((scan[len(l)], scan[len(l) + 1]))
            scan = scan[:len(l)] + scan[len(l) + 2:]

        elif isinstance(box, Cap):
            new_edge = get_new_index(box.left.dim[0])
            scan = scan[:len(l)] + [new_edge, new_edge] + scan[len(l):]

        elif isinstance(box, Spider):
            new_edge = get_new_index(box.type.dim[0])
            merges.extend((scan[len(l) + i], new_edge) for i in range(len(box.dom)))
            output_edges = [new_edge for _ in range(len(box.cod))]
            scan = scan[:len(l)] + output_edges + scan[len(l) + len(box.dom):]
        else:
            input_edges = scan[len(l):len(l) + len(box.dom)]
            output_edges = [get_new_index(size) for size in box.cod.dim] # type: ignore
            tensors.append((box.data, box.dom.dim + box.cod.dim)) # type: ignore
            tensor_edges.append(input_edges + output_edges)
            scan = scan[:len(l)] + output_edges + scan[len(l) + len(box.dom):]
    outputs = scan

    # Merge edges 
    repr = union_find(merges) 
    tensor_edges = [[repr.get(edge, edge) for edge in edges] for edges in tensor_edges]
    inputs = [repr.get(edge, edge) for edge in inputs]
    outputs = [repr.get(edge, edge) for edge in outputs]
    size_dict = {repr.get(edge, edge): size for edge, size in size_dict.items()}

    dangling = inputs + outputs
    if len(set(dangling)) != len(dangling):
        raise ValueError("Duplicate dangling indices found in the diagram. "
                         "This is not supported by the current implementation.")
  
    if not interleaved:
        subs = [''.join(oe.get_symbol(i) for i in indices) for indices in tensor_edges] # type: ignore
        output_subs = ''.join(oe.get_symbol(i) for i in dangling)
        einsum_string = ','.join(subs) + '->' + output_subs
        return einsum_string, tensors
    else:
        data = []
        for tensor, indices in zip(tensors, tensor_edges):
            data.append(tensor)
            data.append(indices)
        data.append(dangling)
        return data
        

class BobcatTextProcessor():

    def __call__(self, sentences: list[str], suppress_exceptions: bool = False,
            return_details: bool = False):
        return self.parse(sentences, suppress_exceptions=suppress_exceptions,
                            return_details=return_details)

    def __init__(self, ccg_parser: BobcatParser, ansatz: TensorAnsatz,
                 rewriter: Optional[Rewriter] = None,
                 tree_reader_mode: Optional[TreeReaderMode] = None):
        """
        Initialize the BobcatProcessor with a CCG parser and an ansatz.
        Args:
            ccg_parser: An instance of a CCG parser (e.g., BobcatParser).
            ansatz: An instance of a tensor ansatz (e.g., MPSAnsatz).
            tree_reader_mode: Optional mode for the tree reader. If
            provided, the diagram will be in the compact form.
        """
        self.ccg_parser = ccg_parser
        self.ansatz = ansatz
        self.tokenizer = Tokenizer()
        self.rewriter = rewriter
        self.tree_reader_mode = tree_reader_mode
    
    def sentences2trees(self, sentences: list[str], 
                        suppress_exceptions: bool,
                        return_details: bool = False):
        """
        Convert a list of sentences into a list of CCG trees.
        """
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        lemmas = [self.tokenizer.lemmatize(tokens) for tokens in tokens]

        trees = self.ccg_parser.sentences2trees(tokens, tokenised=True, 
                                                suppress_exceptions=suppress_exceptions, verbose="suppress")
        lemma_trees = [
            self.lemmatize_tree(tree, lemma) if tree else None 
            for tree, lemma in zip(trees, lemmas) # type: ignore
        ]

        if return_details:
            return {'tokens': tokens,
                    'lemmas': lemmas,
                    'trees': trees,
                    'lemma_trees': lemma_trees}
        else:
            del tokens, lemmas, trees  # Free memory
            return {'lemma_trees': lemma_trees}

    def parse(self, sentences: list[str], suppress_exceptions: bool = False,
              return_details: bool = False):
        """
        Parse a list of sentences into a tensor network diagram.
        """
        results = self.sentences2trees(sentences, suppress_exceptions=suppress_exceptions,
                                       return_details=return_details)

        lemma_trees = results['lemma_trees']
        if self.tree_reader_mode is not None:
            diagrams = [TreeReader.tree2diagram(tree, mode=self.tree_reader_mode) 
                        if tree else None for tree in lemma_trees]
        else:
            diagrams = [tree.to_diagram() if tree else None for tree in lemma_trees]

        if self.rewriter is not None:
            rewritten_diagrams = [self.rewriter(diagram).remove_snakes() if diagram else None for diagram in diagrams]
        else:
            rewritten_diagrams = diagrams  # Use diagrams directly if no rewriter

        circuits = [self.ansatz(diagram) if diagram else None for diagram in rewritten_diagrams]

        einsum_inputs = [tn_to_einsum(circuit) if circuit else None for circuit in circuits]

        # Create a new results dictionary with only what we need
        filtered_results = {}
        filtered_results['einsum_inputs'] = einsum_inputs
        filtered_results['sentences'] = sentences
        
        if return_details:
            filtered_results['lemma_trees'] = lemma_trees
            filtered_results['diagrams'] = diagrams
            filtered_results['rewritten_diagrams'] = rewritten_diagrams
            filtered_results['circuits'] = circuits
            # Copy any other fields from the original results dictionary
            for key in results:
                if key != 'lemma_trees':  # Already added this above
                    filtered_results[key] = results[key]
        
        # Explicitly delete all large objects if not returning them
        del results
        del einsum_inputs
        del diagrams
        del rewritten_diagrams
        del circuits
        del lemma_trees  # Free memory
        
        return filtered_results

    def lemmatize_tree(self, tree: CCGTree, lemmas: list):
        """
        Replace the text of the leaves in a CCG tree with the provided lemmas.
        Args:
            tree: A CCGTree instance.
            lemmas: A list of lemmas to replace the leaves' text.
        Returns:
            A new CCGTree with the leaves' text replaced by the lemmas.
        """
        tree = CCGTree.from_json(tree.to_json())
        def traverse(node: CCGTree, lemma_iter):
            if node.is_leaf:
                node._text = next(lemma_iter)
            else:
                for child in node.children:
                    traverse(child, lemma_iter)

        lemma_iter = iter(lemmas)
        traverse(tree, lemma_iter)
        return tree 