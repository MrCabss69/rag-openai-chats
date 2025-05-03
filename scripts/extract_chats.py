"""
extract_chats.py · v0.5 · 2025‑05‑02
------------------------------------------------
• Lee el conversations.json exportado por la UI de OpenAI (2023‑05 → 2025‑04).
• Convierte cada nodo en una fila de pandas.DataFrame sin perder NINGÚN texto.
• Reconoce todos los formatos de `message.content` y de cada “part”.
"""
from __future__ import annotations
import os 
import json, pathlib, typing as t
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd
from dateutil.parser import parse as dt_parse

try:
    import ijson          # lectura streaming
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

try:
    from tqdm.auto import tqdm    # barra de progreso
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ----------------------------------------------------------------------
#                       helpers internos
# ----------------------------------------------------------------------
def _safe_dt(raw) -> t.Optional[datetime]:
    """Convierte None, timestamp (int|float) o string ISO a datetime."""
    if raw in (None, ""):
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(raw)
        except (OSError, OverflowError):
            return None
    try:
        return dt_parse(str(raw))
    except (ValueError, TypeError):
        return None


def _extract_parts(msg: dict) -> list:
    """
    Devuelve SIEMPRE una lista de “parts” homogénea cualquiera que sea
    el formato de msg["content"]:
        • list                       → tal cual
        • dict con .parts            → .parts
        • dict con .text | .value    → [texto]
        • string                     → [string]
        • None / caso raro           → []
    """
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
        for key in ("text", "value"):           # nuevo formato 2025
            if key in content and isinstance(content[key], str):
                return [content[key]]

    return []                                   # fallback seguro


def _split_parts(parts) -> tuple[str, str, str]:
    """
    Separa una lista de parts en (texto plano, bloques de código Markdown,
    etiquetas multimedia).  Abarca todos los content_type/type actuales.
    """
    txt, code_blks, media = [], [], []

    def _dispatch(p):
        # --- string simple ------------------------------
        if isinstance(p, str):
            txt.append(p);  return
        # --- dict textual -------------------------------
        if isinstance(p, dict):
            if p.get("type") == "text" and "text" in p:
                txt.append(p["text"]); return
            if p.get("content_type") == "text":
                for sub in p.get("parts", []): _dispatch(sub); return
            if p.get("content_type") == "code":
                lang = p.get("language","")
                value = p.get("value") or "\n".join(p.get("parts", [])) or ""
                code_blks.append(f"\n```{lang}\n{value}\n```"); return
            # multimedia
            tag  = p.get("content_type") or p.get("type") or "BLOB"
            hint = (p.get("asset_pointer") or p.get("id") or "")[:12]
            media.append(f"[{tag.upper()}:{hint}]"); return
        # --- lista anidada ------------------------------
        if isinstance(p, list):
            for sub in p: _dispatch(sub); return
        # --- fallback -----------------------------------
        media.append(f"[UNKNOWN:{str(p)[:30]}]")

    for part in parts:
        _dispatch(part)

    return "".join(txt), "\n".join(code_blks), " ".join(media)


@dataclass
class MessageRow:
    conversation_id: str
    conversation_ttl: str
    model_slug: t.Optional[str]
    node_id: str
    parent_id: t.Optional[str]
    depth: int
    order_in_conversation: int
    role: t.Optional[str]
    text: str
    code_md: str
    media_tags: str
    created: t.Optional[datetime]
    updated: t.Optional[datetime]
    plugin_ids: t.Any
    children: list[str]


# ----------------------------------------------------------------------
#                       función pública
# ----------------------------------------------------------------------
def load_chats(
    path: str | pathlib.Path = "conversations.json",
    *,
    keep_raw_json: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Lee `path` y devuelve DataFrame exhaustivo y jerárquico.
    Si `keep_raw_json=True` añade columna extra 'raw' con el dict original
    del mensaje.
    """
    pth = pathlib.Path(path)

    # 1 – Abrir archivo (streaming si hay ijson)
    conv_iter = (
        ijson.items(pth.open("rb"), "item") if HAS_IJSON else json.loads(pth.read_text())
    )
    bar = (
        tqdm(conv_iter, desc="Conversaciones", unit="conv")
        if progress and HAS_TQDM
        else conv_iter
    )

    rows: list[dict] = []

    for convo in bar:
        conv_id   = convo.get("id")
        conv_ttl  = convo.get("title")
        conv_dt   = _safe_dt(convo.get("create_time"))

        mapping = convo["mapping"]
        # DFS para profundidad y orden
        stack = [(nid, None, 0) for nid in mapping if mapping[nid]["parent"] is None]
        order = 0

        while stack:
            node_id, parent_id, depth = stack.pop()
            node = mapping[node_id]
            msg  = node.get("message") or {}

            parts = _extract_parts(msg)
            text, code, media = _split_parts(parts)

            record = MessageRow(
                conversation_id         = conv_id,
                conversation_ttl        = conv_ttl,
                model_slug              = convo.get("model_slug"),
                node_id                 = node_id,
                parent_id               = parent_id,
                depth                   = depth,
                order_in_conversation   = order,
                role                    = (msg.get("author") or {}).get("role"),
                text                    = text,
                code_md                 = code,
                media_tags              = media,
                created                 = _safe_dt(msg.get("create_time")) or conv_dt,
                updated                 = _safe_dt(msg.get("update_time")),
                plugin_ids              = convo.get("plugin_ids"),
                children                = node.get("children") or [],
            )
            row_dict = asdict(record)
            if keep_raw_json:
                row_dict["raw"] = msg
            rows.append(row_dict)

            order += 1
            # hijos (reverse → orden natural)
            for child_id in reversed(node.get("children") or []):
                stack.append((child_id, node_id, depth + 1))

    return pd.DataFrame(rows)



input_path  = "./data/raw/OPENAI/conversations.json"
output_dir  = "./data/processed"
base_name   = "messages"

# --- Carga e identificación de roles ---------------------------------
df = load_chats(input_path)
role_dfs = {
    "user":   df[df.role == "user"].reset_index(drop=True),
    "system": df[df.role == "system"].reset_index(drop=True),
    "tool":   df[df.role == "tool"].reset_index(drop=True),
}

# --- Exportar a JSONL por rol ----------------------------------------
for role, df_role in role_dfs.items():
    out_path = os.path.join(output_dir, f"{base_name}_{role}.jsonl")
    df_role.to_json(
        out_path,
        orient="records",
        lines=True,
        force_ascii=False,
        date_format="iso"       # si quieres ISO en los timestamps
    )
    print(f"👉 Guardado {len(df_role)} registros en {out_path}")