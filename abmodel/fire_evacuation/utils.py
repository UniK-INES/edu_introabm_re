import uuid


def get_random_id(rng) -> uuid.UUID:
    return uuid.UUID(int = int.from_bytes(rng.bytes(16), 'big'), version=4)
