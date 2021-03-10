import typing
import numpy as np

__all__ = ["TNone", "TTuple",
           "TBool", "TInt32", "TInt64", "TFloat",
           "TStr", "TBytes", "TByteArray",
           "TList", "TArray", "TRange32", "TRange64",
           "TVar"]

TNone = type(None)
TBool = bool
TInt32 = np.int32
TInt64 = np.int64
TFloat = float
TStr = str
TBytes = bytes
TByteArray = bytearray
TList = lambda elt=...: typing.List  # Does not type check well because it is a function
TArray = lambda elt=..., num_dims=...: np.ndarray  # Does not type check well because it is a function
TRange32 = typing.Iterable[TInt32]
TRange64 = typing.Iterable[TInt64]
TVar = typing.Any  # Simplified typing
TTuple = lambda elts=...: typing.Tuple  # Does not type check well because it is a function
