import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.stages.translation.backend import TranslationBackend


INDIC_LANG_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "or": "ory_Orya",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "as": "asm_Beng",
}


# How hard diverse beam search is pushed away from repeating token choices
# across beam groups. Too low and the candidates are near-duplicates, which
# defeats the point; too high and later groups drift into bad translations.
DIVERSITY_PENALTY = 0.8


class IndicTrans2Backend(TranslationBackend):
    """
    Concrete IndicTrans2 inference backend.
    """

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        src_lang = INDIC_LANG_TAGS.get(source_language, source_language)
        tgt_lang = INDIC_LANG_TAGS.get(target_language, target_language)

        try:
            self.tokenizer.src_lang = src_lang
        except AttributeError:
            pass

        # The tokenizer requires the text to be prefixed with the language tags.
        formatted_text = f"{src_lang} {tgt_lang} {text}"

        inputs = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.model.device)

        gen_kwargs = {
            "max_length": 256,
            "num_beams": 5,
        }
        
        if hasattr(self.tokenizer, "lang_code_to_id") and tgt_lang in self.tokenizer.lang_code_to_id:
            gen_kwargs["forced_bos_token_id"] = self.tokenizer.lang_code_to_id[tgt_lang]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )

        translated_text = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0]

        return translated_text

    def translate_candidates(
        self,
        text: str,
        source_language: str,
        target_language: str,
        num_candidates: int = 1,
    ) -> list[str]:
        """
        Produce several genuinely different translations of one segment.

        Plain beam search returns near-duplicates, which is useless when the
        point is to find a shorter phrasing. Diverse beam search partitions the
        beams into groups and penalises groups for repeating tokens that
        earlier groups already chose, so the returned set actually spans
        different ways of saying the same thing.

        Results come back longest-confidence first, matching what `translate`
        would return, and duplicates are removed while preserving that order.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded.")

        if num_candidates <= 1:
            return [self.translate(text, source_language, target_language)]

        src_lang = INDIC_LANG_TAGS.get(source_language, source_language)
        tgt_lang = INDIC_LANG_TAGS.get(target_language, target_language)

        inputs = self.tokenizer(
            f"{src_lang} {tgt_lang} {text}",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.model.device)

        # Diverse beam search requires the beam count to divide evenly into
        # groups, and one group per returned sequence gives maximum spread.
        num_beams = max(num_candidates, 2)
        num_groups = num_beams

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=num_beams,
                num_beam_groups=num_groups,
                diversity_penalty=DIVERSITY_PENALTY,
                num_return_sequences=num_candidates,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        unique: list[str] = []
        seen: set[str] = set()

        for candidate in decoded:
            stripped = candidate.strip()
            if stripped and stripped not in seen:
                seen.add(stripped)
                unique.append(stripped)

        return unique or [self.translate(text, source_language, target_language)]
