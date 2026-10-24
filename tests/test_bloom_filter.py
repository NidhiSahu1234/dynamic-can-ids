from backend.app.services.bloom_filter import BloomFilter, BloomFilterConfig


def test_add_and_contains():
    bf = BloomFilter(BloomFilterConfig(expected_items=100, false_positive_rate=0.01))
    sig = "018f|dlc=8|fe:5b:00:00:00:3c:00:00"
    assert sig not in bf
    bf.add(sig)
    assert sig in bf


def test_repeated_insertions_still_member():
    bf = BloomFilter(BloomFilterConfig(expected_items=50, false_positive_rate=0.05))
    sig = "0260|dlc=8|19:21:22:30:08:8e:6d:3a"
    for _ in range(5):
        bf.add(sig)
    assert sig in bf


def test_serialization_roundtrip():
    bf = BloomFilter(BloomFilterConfig(expected_items=20, false_positive_rate=0.1))
    bf.add("test|dlc=1|aa")
    restored = BloomFilter.from_dict(bf.to_dict())
    assert "test|dlc=1|aa" in restored
