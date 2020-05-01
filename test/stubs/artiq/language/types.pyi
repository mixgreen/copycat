import typing
import numpy as np

TNone = None
TBool = bool
TInt32 = np.int32
TInt64 = np.int64
TFloat = float
TStr = str
TBytes = bytes
TByteArray = bytearray
TTuple = typing.Tuple[type, ...]
TList = typing.List[type]
TRange32 = typing.Iterable[TInt32]
TRange64 = typing.Iterable[TInt64]
