import typing
import numpy as np

__all__ = ["TNone", "TTuple",
           "TBool", "TInt32", "TInt64", "TFloat",
           "TStr", "TBytes", "TByteArray",
           "TList", "TArray", "TRange32", "TRange64",
           "TVar"]

TNone = typing.Any  # Simplified typing
TBool = bool
TInt32 = np.int32
TInt64 = np.int64
TFloat = float
TStr = str
TBytes = bytes
TByteArray = bytearray
TList = lambda elt=...: typing.List  # Simplified typing
TArray = lambda elt=..., num_dims=...: np.ndarray  # Simplified typing
TRange32 = typing.Iterable[TInt32]
TRange64 = typing.Iterable[TInt64]
TVar = typing.Any  # Simplified typing
TTuple = lambda elts=...: typing.Tuple  # Simplified typing
