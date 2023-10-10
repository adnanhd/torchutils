from hashlib import md5, sha256, blake2b
from collections import OrderedDict


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


def digest_numpy(arr) -> str:
    return md5(arr.tobytes()).hexdigest()


def digest_torch(arr) -> str:
    return md5(arr.cpu().numpy().tobytes()).hexdigest()


def old_digest(state_dict: OrderedDict) -> str:
    def concat(arr):
        result = 0
        for a in arr:
            result ^= a
        return hex(result)[2:]

    return concat(int(digest_torch(state_array), 16)
                  for state_array in state_dict.values())


def digest(state_dict: OrderedDict) -> str:
    if state_dict is None:
        return 'None'
    else:
        state_dict = state_dict['model']
    return md5("#<@_@>#".join(
        map(digest_torch, state_dict.values())
    ).encode()).hexdigest()
