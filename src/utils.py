# src/utils.py
from __future__ import annotations
import uuid
import html
import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from decimal import Decimal
from typing import Any, List, Optional, Final
from uuid import UUID
from dateutil.parser import parse as dt_parse
from pydantic import BaseModel, ConfigDict
from milvus_haystack import MilvusDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder,SentenceTransformersTextEmbedder


logger = logging.getLogger(__name__)

# ======================================================================
#                         ──  MODELOS  ──
# ======================================================================

class Message(BaseModel):
    message_id:      UUID
    conversation_id: UUID
    conversation_ttl: str
    model_slug:      Optional[str]
    parent_id:       Optional[UUID]
    depth:           int
    order_in_conv:   int
    role:            Optional[str]
    text:            str
    code_md:         str
    media_tags:      str
    created_at:      Optional[datetime]
    updated_at:      Optional[datetime]
    plugin_ids:      Any
    children:        List[UUID]
    raw:             Optional[dict] = None

    model_config = ConfigDict(extra="forbid")


class Chunk(BaseModel):
    chunk_id:    int | None = None      # se completa en SQL
    message_id:  UUID
    chunk_index: int
    chunk_text:  str                    # siempre != ""
    role_prefix: Optional[str] = None
    model_config = ConfigDict(extra="forbid")

# ---------- Helpers de UUID ---------------------------------------
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", re.I)

def as_uuid(val: str) -> UUID:
    """Convierte cualquier id a UUID.
    * Si ya es un UUID válido ⇒ lo devuelve tal cual.
    * Si no lo es ⇒ genera un UUID determinista (v5) usando NAMESPACE_URL.
    """
    if _UUID_RE.match(val):
        return UUID(val)
    # raíz especial "client-created-root" u otros ids libres
    return uuid.uuid5(uuid.NAMESPACE_URL, val)


# ---------- limpieza ---------------------------------------------------
_CTRL = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]")
_URL  = re.compile(r"https?://\S+")

try:
    from haystack.components.preprocessors import TextCleaner
    _tc = TextCleaner(
        convert_to_lowercase=False,
        remove_punctuation=False,
        remove_numbers=False,
        remove_regexps=[r"https?://\S+"]
    )
    _HAS_TC = True
except ImportError:
    _HAS_TC = False

def clean_text(text: str) -> str:
    txt = html.unescape(str(text))
    txt = unicodedata.normalize("NFKC", txt)
    txt = _CTRL.sub("", txt)
    if _HAS_TC:
        return _tc.run(texts=[txt])["texts"][0]
    txt = _URL.sub("", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt



# ======================================================================
#                         ──  CHUNKING  ──
# ======================================================================

try:
    from haystack.nodes import PreProcessor
    _PRE = PreProcessor(split_by="token", split_length=500, split_overlap=50,
                        split_respect_sentence_boundary=True)
    _HAS_PRE = True
except ImportError:
    _HAS_PRE = False

def chunk_text(text: str) -> List[str]:
    if _HAS_PRE:
        docs = _PRE.process({"content": text})
        return [d.content for d in docs]
    # Chunking simple
    max_chars, overlap = 500, 80
    chunks, start = [], 0
    L = len(text)
    while start < L:
        end = min(start + max_chars, L)
        chunks.append(text[start:end])
        start += max_chars - overlap
    return chunks



# ======================================================================
#                      ──  PARSING HELPERS  ──
# ======================================================================

def _safe_dt(raw) -> Optional[datetime]:
    if raw in (None, ""):
        return None
    if isinstance(raw, (int, float, Decimal)):
        try:
            return datetime.fromtimestamp(float(raw))
        except (OverflowError, OSError):
            return None
    try:
        return dt_parse(str(raw))
    except (ValueError, TypeError):
        return None


def _extract_parts(msg: dict) -> list:
    content = msg.get("content")
    if content is None:
        return []
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        return [content]
    if isinstance(content, dict):
        if isinstance(content.get("parts"), list):
            return content["parts"]
        for key in ("text","value"):
            if key in content and isinstance(content[key], str):
                return [content[key]]
    return []


def _split_parts(parts) -> tuple[str,str,str]:
    txt, code, media = [], [], []
    def _disp(p):
        if isinstance(p,str):
            txt.append(p); return
        if not isinstance(p,dict):
            media.append(f"[UNKNOWN:{str(p)[:30]}]"); return
        if p.get("type")=="text" and "text" in p:
            txt.append(p["text"]); return
        if p.get("content_type")=="text":
            for sub in p.get("parts",[]): _disp(sub)
            return
        if p.get("content_type")=="code":
            lang = p.get("language","")
            val  = p.get("value") or "\n".join(p.get("parts",[])) or ""
            code.append(f"\n```{lang}\n{val}\n```")
            return
        ctype = p.get("content_type") or p.get("type") or "BLOB"
        hint  = (p.get("asset_pointer") or p.get("id") or "")[:12]
        media.append(f"[{ctype.upper()}:{hint}]")
    for part in parts:
        if isinstance(part,list):
            for sub in part: _disp(sub)
        else:
            _disp(part)
    return "".join(txt), "\n".join(code), " ".join(media)


# ======================================================================
#                          ──  I/O helpers  ──
# ======================================================================
def to_jsonl(objs: list[BaseModel], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o.model_dump(mode="json"), ensure_ascii=False) + "\n")






_META_COLS: Final[List[str]] = [
    # Identificadores jerárquicos
    "message_id",
    "parent_id",
    "conversation_id",

    # Orden lógico dentro de la conversación
    "depth",           # profundidad en el árbol
    "order_in_conv",   # orden global 

    # Contexto del mensaje
    "role",             # user / assistant / system / tool…
    "model_slug",       # e.g. gpt-4-1106-preview
    "conversation_ttl", # título de la conversación
    
    # Timestamps
    "created_at",
    "updated_at",
]

INDEX_NAME   = "openai_messages_chunks"
MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2" 

embed_instruction = "Represent this sentence for searching relevant passages:"
doc_embedder    = SentenceTransformersDocumentEmbedder(
    model=MODEL_NAME
)

text_embedder    = SentenceTransformersTextEmbedder(
    model=MODEL_NAME
)


# from sentence_transformers import SentenceTransformer

# st_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")



mivuls_doc_store = MilvusDocumentStore(
    connection_args={"uri": "/home/jd/Documentos/CODIGO-2025/Machine-Learning-2025/src/ML/tutorials/rag-openai-chats/sql/milvus_v0.db"},
    # drop_old=True,
)