from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PushTextRequest(_message.Message):
    __slots__ = ("text", "time_stamp")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    text: str
    time_stamp: str
    def __init__(self, text: _Optional[str] = ..., time_stamp: _Optional[str] = ...) -> None: ...

class PushTextResponse(_message.Message):
    __slots__ = ("request_status",)
    REQUEST_STATUS_FIELD_NUMBER: _ClassVar[int]
    request_status: str
    def __init__(self, request_status: _Optional[str] = ...) -> None: ...

class PopTextRequest(_message.Message):
    __slots__ = ("time_stamp",)
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    time_stamp: str
    def __init__(self, time_stamp: _Optional[str] = ...) -> None: ...

class PopTextResponse(_message.Message):
    __slots__ = ("text", "request_status")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    REQUEST_STATUS_FIELD_NUMBER: _ClassVar[int]
    text: str
    request_status: str
    def __init__(self, text: _Optional[str] = ..., request_status: _Optional[str] = ...) -> None: ...

class MatrixRow(_message.Message):
    __slots__ = ("elements",)
    ELEMENTS_FIELD_NUMBER: _ClassVar[int]
    elements: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, elements: _Optional[_Iterable[float]] = ...) -> None: ...

class Matrix(_message.Message):
    __slots__ = ("rows",)
    ROWS_FIELD_NUMBER: _ClassVar[int]
    rows: _containers.RepeatedCompositeFieldContainer[MatrixRow]
    def __init__(self, rows: _Optional[_Iterable[_Union[MatrixRow, _Mapping]]] = ...) -> None: ...

class PushFrameRequest(_message.Message):
    __slots__ = ("frame", "time_stamp")
    FRAME_FIELD_NUMBER: _ClassVar[int]
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    frame: Matrix
    time_stamp: str
    def __init__(self, frame: _Optional[_Union[Matrix, _Mapping]] = ..., time_stamp: _Optional[str] = ...) -> None: ...

class PushFrameResponse(_message.Message):
    __slots__ = ("request_status",)
    REQUEST_STATUS_FIELD_NUMBER: _ClassVar[int]
    request_status: str
    def __init__(self, request_status: _Optional[str] = ...) -> None: ...

class PopFrameRequest(_message.Message):
    __slots__ = ("time_stamp",)
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    time_stamp: str
    def __init__(self, time_stamp: _Optional[str] = ...) -> None: ...

class PopFrameResponse(_message.Message):
    __slots__ = ("frame", "request_status")
    FRAME_FIELD_NUMBER: _ClassVar[int]
    REQUEST_STATUS_FIELD_NUMBER: _ClassVar[int]
    frame: Matrix
    request_status: str
    def __init__(self, frame: _Optional[_Union[Matrix, _Mapping]] = ..., request_status: _Optional[str] = ...) -> None: ...

class PushImageRequest(_message.Message):
    __slots__ = ("text", "time_stamp")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    text: str
    time_stamp: str
    def __init__(self, text: _Optional[str] = ..., time_stamp: _Optional[str] = ...) -> None: ...

class PushImageResponse(_message.Message):
    __slots__ = ("request_status",)
    REQUEST_STATUS_FIELD_NUMBER: _ClassVar[int]
    request_status: str
    def __init__(self, request_status: _Optional[str] = ...) -> None: ...

class PopImageRequest(_message.Message):
    __slots__ = ("time_stamp",)
    TIME_STAMP_FIELD_NUMBER: _ClassVar[int]
    time_stamp: str
    def __init__(self, time_stamp: _Optional[str] = ...) -> None: ...

class PopImageResponse(_message.Message):
    __slots__ = ("text", "request_status")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    REQUEST_STATUS_FIELD_NUMBER: _ClassVar[int]
    text: str
    request_status: str
    def __init__(self, text: _Optional[str] = ..., request_status: _Optional[str] = ...) -> None: ...
