from lambeq import BobcatParser
from lambeq.core.utils import SentenceBatchType


class CachedBobcatParser(BobcatParser):
    def __init__(self, *args,
                 **kwargs):
        """
        A cached version of the BobcatParser that uses diskcache to store
        previously parsed sentences. This avoids re-parsing the same sentences
        multiple times, which can be time-consuming.
        Args:
            cache_path (str): Path to the diskcache directory.
            load_parser (bool): Whether to load the parser immediately.
        """
        super().__init__(verbose="suppress", *args, **kwargs)

        self.args = args
        self.kwargs = kwargs

    def _load_parser(self):
        if not hasattr(self, 'tagger'):
            super().__init__(*self.args, **self.kwargs)

    def sentences2trees(self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
        verbose: str | None = 'progress'):
        
        self._load_parser()
        results = super().sentences2trees(
            sentences,
            tokenised=tokenised,
            suppress_exceptions=suppress_exceptions, 
            verbose=verbose
        )

        return results
