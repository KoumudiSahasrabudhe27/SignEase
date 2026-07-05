"""
IndicTrans2 En→Indic inference service.

Mirrors the official HuggingFace pipeline from:
https://github.com/AI4Bharat/IndicTrans2/blob/main/huggingface_interface/example.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set

# NLTK must resolve data from project-local dir before any nltk import.
NLTK_DATA_DIR = Path(__file__).resolve().parent / "nltk_data"
NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)

import torch


def _patch_transformers_for_indictrans() -> None:
    """IndicTransToolkit collator expects PreTrainedTokenizerBase on transformers 4.x path."""
    import transformers.tokenization_utils as tokenization_utils

    if not hasattr(tokenization_utils, "PreTrainedTokenizerBase"):
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase


_patch_transformers_for_indictrans()

from IndicTransToolkit.processor import IndicProcessor
from mosestokenizer import MosesSentenceSplitter
from nltk import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# FLORES language code mapping from official example.py
FLORES_CODES = {
    "asm_Beng": "as",
    "awa_Deva": "hi",
    "ben_Beng": "bn",
    "bho_Deva": "hi",
    "brx_Deva": "hi",
    "doi_Deva": "hi",
    "eng_Latn": "en",
    "gom_Deva": "kK",
    "guj_Gujr": "gu",
    "hin_Deva": "hi",
    "hne_Deva": "hi",
    "kan_Knda": "kn",
    "kas_Arab": "ur",
    "kas_Deva": "hi",
    "kha_Latn": "en",
    "lus_Latn": "en",
    "mag_Deva": "hi",
    "mai_Deva": "hi",
    "mal_Mlym": "ml",
    "mar_Deva": "mr",
    "mni_Beng": "bn",
    "mni_Mtei": "hi",
    "npi_Deva": "ne",
    "ory_Orya": "or",
    "pan_Guru": "pa",
    "san_Deva": "hi",
    "sat_Olck": "or",
    "snd_Arab": "ur",
    "snd_Deva": "hi",
    "tam_Taml": "ta",
    "tel_Telu": "te",
    "urd_Arab": "ur",
}

SUPPORTED_TARGET_LANGS: Set[str] = {
    "asm_Beng",
    "ben_Beng",
    "brx_Deva",
    "doi_Deva",
    "guj_Gujr",
    "hin_Deva",
    "kan_Knda",
    "kas_Arab",
    "kas_Deva",
    "gom_Deva",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "mni_Beng",
    "mni_Mtei",
    "npi_Deva",
    "ory_Orya",
    "pan_Guru",
    "san_Deva",
    "sat_Olck",
    "snd_Arab",
    "snd_Deva",
    "tam_Taml",
    "tel_Telu",
    "urd_Arab",
}

SOURCE_LANG = "eng_Latn"
DEFAULT_MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
HF_MODEL_PAGE = f"https://huggingface.co/{DEFAULT_MODEL_ID}"
BATCH_SIZE = 4

from load_env import load_backend_env

load_backend_env()


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return token.strip() if token else None


def _configure_hf_auth() -> str | None:
    token = _hf_token()
    if not token:
        return None
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    except Exception:
        pass
    return token


def _resolve_model_path() -> str:
    """
    Resolve HuggingFace model id or local checkpoint directory.

    Priority:
    1) INDICTRANS_MODEL_DIR (local HF export folder)
    2) INDICTRANS_MODEL_ID (HuggingFace repo id)
    3) DEFAULT_MODEL_ID
    """
    local = os.environ.get("INDICTRANS_MODEL_DIR", "").strip()
    if local:
        path = Path(local).expanduser().resolve()
        if path.is_dir():
            return str(path)
        raise FileNotFoundError(f"INDICTRANS_MODEL_DIR not found: {path}")

    return os.environ.get("INDICTRANS_MODEL_ID", DEFAULT_MODEL_ID)


def _pick_device() -> str:
    """Match official example.py: cuda if available, otherwise cpu (no MPS)."""
    override = os.environ.get("INDICTRANS_DEVICE", "").strip().lower()
    if override in {"cpu", "cuda", "mps"}:
        return override
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _ensure_nltk_data() -> None:
    """Download punkt tokenizers into project-local nltk_data (official example dependency)."""
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass

    import nltk

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, download_dir=str(NLTK_DATA_DIR), quiet=True)
            nltk.data.find(f"tokenizers/{resource}")


def split_sentences(input_text: str, lang: str) -> List[str]:
    """Official IndicTrans2 sentence splitting (example.py)."""
    if lang == "eng_Latn":
        input_sentences = sent_tokenize(input_text)
        with MosesSentenceSplitter(FLORES_CODES[lang]) as splitter:
            sents_moses = splitter([input_text])
        sents_nltk = sent_tokenize(input_text)
        if len(sents_nltk) < len(sents_moses):
            input_sentences = sents_nltk
        else:
            input_sentences = sents_moses
        return [sent.replace("\xad", "") for sent in input_sentences]

    from indicnlp.tokenize.sentence_tokenize import DELIM_PAT_NO_DANDA, sentence_split

    return sentence_split(
        input_text, lang=FLORES_CODES[lang], delim_pat=DELIM_PAT_NO_DANDA
    )


def initialize_model_and_tokenizer(
    ckpt_dir: str,
    device: str,
) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """Official model/tokenizer loading (example.py, no quantization)."""
    token = _configure_hf_auth()
    model_kwargs: dict = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    if token:
        model_kwargs["token"] = token

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ckpt_dir, trust_remote_code=True, token=token
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, **model_kwargs)
    except OSError as exc:
        message = str(exc)
        if "gated repo" in message.lower() or "401" in message:
            raise RuntimeError(
                "Cannot download IndicTrans2 — the HuggingFace model is gated.\n\n"
                f"1) Accept access: {HF_MODEL_PAGE}\n"
                "2) From the backend2 folder run:\n"
                "     python3 hf_login.py\n"
                "3) Start the server:\n"
                "     python3 translation_api.py\n"
            ) from exc
        raise

    model = model.to(device)
    if device == "cuda":
        model = model.half()
    model.eval()
    return tokenizer, model


def batch_translate(
    input_sentences: List[str],
    src_lang: str,
    tgt_lang: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    ip: IndicProcessor,
    device: str,
) -> List[str]:
    """Official batched inference (example.py)."""
    translations: List[str] = []

    for i in range(0, len(input_sentences), BATCH_SIZE):
        batch = input_sentences[i : i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=(device == "cuda"),
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        generated_tokens = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        translations += ip.postprocess_batch(generated_tokens, lang=tgt_lang)

        del inputs
        if device == "cuda":
            torch.cuda.empty_cache()

    return translations


def translate_paragraph(
    input_text: str,
    src_lang: str,
    tgt_lang: str,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    ip: IndicProcessor,
    device: str,
) -> str:
    """Official paragraph translation (example.py)."""
    input_sentences = split_sentences(input_text, src_lang)
    translated = batch_translate(
        input_sentences, src_lang, tgt_lang, model, tokenizer, ip, device
    )
    return " ".join(translated)


class IndicTrans2Service:
    """En→Indic translator; model loaded once at startup via load()."""

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id = model_id or _resolve_model_path()
        self.device = _pick_device()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.processor: Optional[IndicProcessor] = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        _ensure_nltk_data()
        self.processor = IndicProcessor(inference=True)
        self.tokenizer, self.model = initialize_model_and_tokenizer(
            self.model_id, self.device
        )
        self._loaded = True

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not self._loaded or self.model is None or self.tokenizer is None or self.processor is None:
            raise RuntimeError("IndicTrans2 model is not loaded.")

        if src_lang != SOURCE_LANG:
            raise ValueError(f"Only English source is supported (expected {SOURCE_LANG}).")
        if tgt_lang not in SUPPORTED_TARGET_LANGS:
            raise ValueError(f"Unsupported target language: {tgt_lang}")

        stripped = text.strip()
        if not stripped:
            return ""

        return translate_paragraph(
            stripped,
            src_lang,
            tgt_lang,
            self.model,
            self.tokenizer,
            self.processor,
            self.device,
        )
