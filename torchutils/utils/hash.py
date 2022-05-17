from hashlib import md5, sha256, blake2b

class Hashable:
    def _digest(self) -> bytes:
        #randomizer = lambda s: "".join(chr(ord(c) + 32) for c in s)
        return bytes(self.__repr__(), encoding='utf-8')

    def _md5(self) -> str:
        return md5(self._digest()).hexdigest()
    
    def _sha256(self) -> str:
        return sha256(self._digest()).hexdigest()
    
    def _blake2b(self) -> str:
        return blake2b(self._digest()).hexdigest()
