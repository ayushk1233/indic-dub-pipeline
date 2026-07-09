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
