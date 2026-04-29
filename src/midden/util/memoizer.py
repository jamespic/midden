from typing import Generic, TypeVar
from weakref import WeakValueDictionary

T = TypeVar("T")

class Memoizer(Generic[T]):
    def __init__(self):
        self.cache = WeakValueDictionary()
    
    def get(self, value: T) -> T:
        h = hash(value)
        if h in self.cache:
            if (result:= self.cache[h]) == value:
                return result
            else:
                return value
        else:
            self.cache[h] = value
            return value