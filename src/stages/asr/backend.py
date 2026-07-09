from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    """
    Abstract interface for speech recognition inference engines.
    """

    @abstractmethod
    def load(self) -> None:
        """
        Load the underlying ASR model.
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe_audio(
        self,
        audio_path: str,
    ):
        """
        Run inference on a single audio file.
        """
        raise NotImplementedError
