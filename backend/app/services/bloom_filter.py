"""
Simple Bloom filter with deterministic double hashing.

Limitations
-----------
- False positives are possible: a normal frame may appear as a member.
- Deletions are not supported; removing a rule does not remove bits from the filter.
  The rule store remains authoritative; Bloom is a fast pre-filter for confirmed signatures.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


def _hashes(item: str, k: int) -> list[int]:
    digest = hashlib.sha256(item.encode("utf-8")).digest()
    h1 = int.from_bytes(digest[:8], "big")
    h2 = int.from_bytes(digest[8:16], "big") or 1
    return [(h1 + i * h2) % (2**63 - 1) for i in range(k)]


@dataclass
class BloomFilterConfig:
    expected_items: int = 10_000
    false_positive_rate: float = 0.01

    def __post_init__(self) -> None:
        if type(self.expected_items) is not int or self.expected_items < 1:
            raise ValueError("expected_items must be a positive integer")
        if not 0 < self.false_positive_rate <= 0.5:
            raise ValueError("false_positive_rate must be greater than 0 and at most 0.5")


class BloomFilter:
    def __init__(self, config: BloomFilterConfig | None = None) -> None:
        config = config or BloomFilterConfig()
        self.config = BloomFilterConfig(
            expected_items=config.expected_items,
            false_positive_rate=config.false_positive_rate,
        )
        n = self.config.expected_items
        p = self.config.false_positive_rate
        m = max(64, int(-(n * math.log(p)) / (math.log(2) ** 2)))
        k = max(1, int((m / n) * math.log(2)))
        self.size = m
        self.num_hashes = k
        self._bits = bytearray((m + 7) // 8)

    @property
    def estimated_false_positive_rate(self) -> float:
        n = self.count or 1
        return (1 - math.exp(-self.num_hashes * n / self.size)) ** self.num_hashes

    @property
    def count(self) -> int:
        return getattr(self, "_count", 0)

    def _set_bit(self, index: int) -> None:
        self._bits[index // 8] |= 1 << (index % 8)

    def _get_bit(self, index: int) -> bool:
        return bool(self._bits[index // 8] & (1 << (index % 8)))

    def _indices(self, item: str) -> list[int]:
        return [h % self.size for h in _hashes(item, self.num_hashes)]

    def add(self, item: str) -> None:
        for idx in self._indices(item):
            self._set_bit(idx)
        self._count = self.count + 1

    def __contains__(self, item: str) -> bool:
        return all(self._get_bit(idx) for idx in self._indices(item))

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "num_hashes": self.num_hashes,
            "bits_hex": self._bits.hex(),
            "insert_count": self.count,
            "config": {
                "expected_items": self.config.expected_items,
                "false_positive_rate": self.config.false_positive_rate,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BloomFilter":
        config_data = data.get("config") or {
            "expected_items": max(1, int(data.get("insert_count", 1))),
            "false_positive_rate": 0.01,
        }
        bf = cls.__new__(cls)
        bf.config = BloomFilterConfig(**config_data)
        bf.size = int(data["size"])
        bf.num_hashes = int(data["num_hashes"])
        bf._bits = bytearray.fromhex(data["bits_hex"])
        bf._count = int(data.get("insert_count", 0))
        if bf.size < 1 or bf.num_hashes < 1 or len(bf._bits) != (bf.size + 7) // 8:
            raise ValueError("Invalid serialized Bloom filter dimensions")
        return bf
