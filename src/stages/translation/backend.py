from abc import ABC, abstractmethod


class TranslationBackend(ABC):
    """
    Abstract interface for translation inference engines.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load the underlying translation model.
        """
        raise NotImplementedError

    @abstractmethod
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        """
        Translate a single text segment.
        """
        raise NotImplementedError

    def translate_candidates(
        self,
        text: str,
        source_language: str,
        target_language: str,
        num_candidates: int = 1,
    ) -> list[str]:
        """
        Return several distinct translations of one segment, best first.

        Length control needs options to choose between; a backend that cannot
        produce them still works, it just leaves nothing to choose from. The
        first element is always what `translate` would have returned, so
        callers can measure what choosing differently cost them.
        """
        return [self.translate(text, source_language, target_language)]
