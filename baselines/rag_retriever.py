"""Retrieval-memory baseline.

Given a test context, we retrieve the top-K most similar client utterances
from the training corpus using BM25 over jieba-tokenized Chinese text, then
prepend the top-K matched (client_turn, counselor_reply) pairs as a
``memory:`` system block before the usual chat history.

This doesn't require training modifications — we can treat it as a
data-augmentation wrapper around the *same* SFT baseline.

Retrieval modes
---------------
* ``"global"`` (default): pure BM25 over all training client turns.  Used by
  the vanilla RAG baseline and by CPsyCoun-style memory.
* ``"emotion_typed"``: the KEMI (Deng et al., ACL 2023) emotion-typed
  retriever.  Every corpus entry is tagged with a coarse emotion label
  inferred from a lexicon (distress / worry / confusion / hopeless /
  calm); at query time we first predict the query's emotion bucket and
  restrict the BM25 search to in-bucket entries before back-filling from
  other buckets if we have fewer than ``k`` hits.  This is the key
  algorithmic idea the paper contributes over vanilla RAG.

Usage in training:
    Tell the ``PsyChatDataset`` to augment each row by
    ``rag.augment_messages(row['messages'])``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi


# -- Lightweight Chinese emotion buckets for KEMI-style emotion-typed retrieval --
# The lexicon is deliberately small and interpretable. It is applied to the
# client turn only (the counsellor's reply is not used for bucketing).
_EMO_BUCKETS: dict[str, list[str]] = {
    "distress":  ["痛苦", "难受", "绝望", "崩溃", "伤心", "难过", "心痛", "压抑", "郁闷"],
    "worry":     ["担心", "害怕", "紧张", "焦虑", "忧虑", "不安", "惶恐", "恐惧"],
    "anger":     ["生气", "愤怒", "厌恶", "恨", "烦躁", "讨厌", "委屈", "不公"],
    "hopeless":  ["没用", "没意义", "活不下去", "没希望", "不想活", "死了算了", "轻生"],
    "confusion": ["不知道", "困惑", "迷茫", "搞不懂", "不明白", "犹豫", "分不清"],
    "calm":      ["还好", "平静", "一般", "还行", "放松", "接受"],
}
_BUCKET_NAMES = list(_EMO_BUCKETS.keys())


def _infer_emotion_bucket(text: str) -> str:
    """Return the coarse emotion bucket for a piece of client text.

    Defaults to ``"confusion"`` when no lexicon term fires (a neutral bucket
    that overlaps with most counselling openings).
    """
    scores = {name: 0 for name in _BUCKET_NAMES}
    for name, words in _EMO_BUCKETS.items():
        for w in words:
            scores[name] += len(re.findall(w, text))
    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else "confusion"


class RAGMemory:
    """BM25 memory bank with optional emotion-typed retrieval (KEMI, 2023)."""

    def __init__(
        self,
        corpus_path: str | Path,
        k: int = 3,
        max_memory_chars: int = 600,
        mode: str = "global",
    ):
        self.k = k
        self.max_memory_chars = max_memory_chars
        self.mode = mode
        self.client_texts: list[str] = []
        self.counselor_replies: list[str] = []
        self.buckets: list[str] = []
        with Path(corpus_path).open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                msgs = rec.get("messages", [])
                if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
                    continue
                last_user = next(
                    (m["content"] for m in reversed(msgs[:-1]) if m.get("role") == "user"),
                    None,
                )
                if not last_user:
                    continue
                self.client_texts.append(last_user)
                self.counselor_replies.append(msgs[-1]["content"])
                self.buckets.append(_infer_emotion_bucket(last_user))
        tokenized = [list(jieba.cut(t)) for t in self.client_texts]
        self.bm25 = BM25Okapi(tokenized)
        # Per-bucket index cache: bucket -> list of memory indices.
        self._bucket_index: dict[str, list[int]] = {b: [] for b in _BUCKET_NAMES}
        for i, b in enumerate(self.buckets):
            self._bucket_index[b].append(i)

    # ------------------------------------------------------------------ utils
    def retrieve(self, query: str) -> list[tuple[str, str]]:
        toks = list(jieba.cut(query))
        scores = self.bm25.get_scores(toks)
        if self.mode == "emotion_typed":
            qb = _infer_emotion_bucket(query)
            candidates = self._bucket_index.get(qb, [])
            if len(candidates) < self.k:
                # Back-fill across all buckets, keeping in-bucket first.
                seen = set(candidates)
                tail = [i for i in range(len(scores)) if i not in seen]
                candidates = candidates + tail
            in_bucket_sorted = sorted(candidates, key=lambda i: -scores[i])
            top = in_bucket_sorted[: self.k]
        else:
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[: self.k]
        return [(self.client_texts[i], self.counselor_replies[i]) for i in top]

    def augment_messages(self, messages: list[dict]) -> list[dict]:
        """Prepend a memory block to the system message."""
        query = next(
            (m["content"] for m in reversed(messages[:-1]) if m.get("role") == "user"),
            "",
        )
        if not query:
            return messages
        hits = self.retrieve(query)
        header = "参考资料（不要直接复述，只作为启发）"
        if self.mode == "emotion_typed":
            qb = _infer_emotion_bucket(query)
            header = f"情绪类别={qb} 的参考案例（仅作启发，不要直接复述）"
        memory = []
        used = 0
        for c_text, r_text in hits:
            block = f"[相似案例] 用户: {c_text}\n咨询师: {r_text}"
            if used + len(block) > self.max_memory_chars:
                break
            memory.append(block)
            used += len(block)
        if not memory:
            return messages

        sys = next((m for m in messages if m.get("role") == "system"), None)
        mem_text = "\n\n".join(memory)
        if sys:
            sys["content"] = sys["content"] + f"\n\n{header}：\n" + mem_text
            return messages
        return [{"role": "system", "content": f"{header}：\n" + mem_text}] + messages
