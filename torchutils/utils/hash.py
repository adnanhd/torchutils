from hashlib import md5, sha256, blake2b


class Hashable:
    @property
    def md5(self) -> str:
        return md5(self._digest()).hexdigest()

    @property
    def sha256(self) -> str:
        return sha256(self._digest()).hexdigest()

    @property
    def blake2b(self) -> str:
        return blake2b(self._digest()).hexdigest()

    def _digest(self) -> bytes:
        return bytes(self.__repr__(), encoding='utf-8')

    def __hash__(self) -> int:
        return int(self.md5, 16)


def digest_numpy(arr):
    return md5(arr.tobytes()).hexdigest()


def digest_torch(arr):
    return md5(arr.cpu().numpy().tobytes()).hexdigest()
