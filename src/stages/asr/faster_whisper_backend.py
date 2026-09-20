from faster_whisper import WhisperModel

from src.stages.asr.backend import InferenceBackend


class FasterWhisperBackend(InferenceBackend):
    """
    Concrete Faster-Whisper inference backend.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        language: str,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model = None

    def load(self) -> None:
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe_audio(
        self,
        audio_path: str,
    ):
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        if self.language == "auto":
            return self.model.transcribe(audio_path)

        return self.model.transcribe(
            audio_path,
            language=self.language,
        )
