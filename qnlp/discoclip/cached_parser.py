import os

import diskcache

from lambeq import BobcatParser
from lambeq.core.utils import SentenceBatchType


class CachedBobcatParser(BobcatParser):
    def __init__(self, *args,
                 cache_path: str = "~/.cache/lambeq/bobcat/diskcache",
                 load_parser: bool = False,
                 **kwargs):
        """
        A cached version of the BobcatParser that uses diskcache to store
        previously parsed sentences. This avoids re-parsing the same sentences
        multiple times, which can be time-consuming.
        Args:
            cache_path (str): Path to the diskcache directory.
            load_parser (bool): Whether to load the parser immediately.
        """
        super().__init__(verbose="suppress")
        self._cache_path = os.path.expanduser(cache_path)
        self._cache = diskcache.Cache(self._cache_path)

        self.args = args
        self.kwargs = kwargs
        if load_parser:
            self._load_parser()

    def _load_parser(self):
        if not hasattr(self, 'tagger'):
            super().__init__(*self.args, **self.kwargs)

    def sentences2trees(self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = 'progress'):

        results = [None] * len(sentences)
        uncached_sentences = []
        
        for i, sent in enumerate(sentences):
            key = str((sent, tokenised, suppress_exceptions))
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_sentences.append((i, sent))
        
        if uncached_sentences:
            self._load_parser()
            uncached_results = super().sentences2trees(
                [sent for _, sent in uncached_sentences],
                tokenised=tokenised,
                suppress_exceptions=suppress_exceptions, 
                verbose=verbose
            )
            
            for (i, sent), result in zip(uncached_sentences, uncached_results):
                key = str((sent, tokenised, suppress_exceptions))
                self._cache[key] = result
                results[i] = result

        return results
