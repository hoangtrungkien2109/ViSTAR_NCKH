# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: streaming.proto
# Protobuf Python Version: 4.25.0
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0fstreaming.proto\x12\x12streaming_services\"3\n\x0fPushTextRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x12\n\ntime_stamp\x18\x02 \x01(\t\"*\n\x10PushTextResponse\x12\x16\n\x0erequest_status\x18\x01 \x01(\t\"$\n\x0ePopTextRequest\x12\x12\n\ntime_stamp\x18\x01 \x01(\t\"7\n\x0fPopTextResponse\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x16\n\x0erequest_status\x18\x02 \x01(\t\"\x1d\n\tMatrixRow\x12\x10\n\x08\x65lements\x18\x01 \x03(\x02\"5\n\x06Matrix\x12+\n\x04rows\x18\x01 \x03(\x0b\x32\x1d.streaming_services.MatrixRow\"Q\n\x10PushFrameRequest\x12)\n\x05\x66rame\x18\x01 \x01(\x0b\x32\x1a.streaming_services.Matrix\x12\x12\n\ntime_stamp\x18\x02 \x01(\t\"+\n\x11PushFrameResponse\x12\x16\n\x0erequest_status\x18\x01 \x01(\t\"%\n\x0fPopFrameRequest\x12\x12\n\ntime_stamp\x18\x01 \x01(\t\"U\n\x10PopFrameResponse\x12)\n\x05\x66rame\x18\x01 \x01(\x0b\x32\x1a.streaming_services.Matrix\x12\x16\n\x0erequest_status\x18\x02 \x01(\t\"4\n\x10PushImageRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x12\n\ntime_stamp\x18\x02 \x01(\t\"+\n\x11PushImageResponse\x12\x16\n\x0erequest_status\x18\x01 \x01(\t\"%\n\x0fPopImageRequest\x12\x12\n\ntime_stamp\x18\x01 \x01(\t\"8\n\x10PopImageResponse\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x16\n\x0erequest_status\x18\x02 \x01(\t2\xaa\x04\n\tStreaming\x12W\n\x08PushText\x12#.streaming_services.PushTextRequest\x1a$.streaming_services.PushTextResponse\"\x00\x12V\n\x07PopText\x12\".streaming_services.PopTextRequest\x1a#.streaming_services.PopTextResponse\"\x00\x30\x01\x12Z\n\tPushFrame\x12$.streaming_services.PushFrameRequest\x1a%.streaming_services.PushFrameResponse\"\x00\x12Y\n\x08PopFrame\x12#.streaming_services.PopFrameRequest\x1a$.streaming_services.PopFrameResponse\"\x00\x30\x01\x12Z\n\tPushImage\x12$.streaming_services.PushImageRequest\x1a%.streaming_services.PushImageResponse\"\x00\x12Y\n\x08PopImage\x12#.streaming_services.PopImageRequest\x1a$.streaming_services.PopImageResponse\"\x00\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'streaming_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_PUSHTEXTREQUEST']._serialized_start=39
  _globals['_PUSHTEXTREQUEST']._serialized_end=90
  _globals['_PUSHTEXTRESPONSE']._serialized_start=92
  _globals['_PUSHTEXTRESPONSE']._serialized_end=134
  _globals['_POPTEXTREQUEST']._serialized_start=136
  _globals['_POPTEXTREQUEST']._serialized_end=172
  _globals['_POPTEXTRESPONSE']._serialized_start=174
  _globals['_POPTEXTRESPONSE']._serialized_end=229
  _globals['_MATRIXROW']._serialized_start=231
  _globals['_MATRIXROW']._serialized_end=260
  _globals['_MATRIX']._serialized_start=262
  _globals['_MATRIX']._serialized_end=315
  _globals['_PUSHFRAMEREQUEST']._serialized_start=317
  _globals['_PUSHFRAMEREQUEST']._serialized_end=398
  _globals['_PUSHFRAMERESPONSE']._serialized_start=400
  _globals['_PUSHFRAMERESPONSE']._serialized_end=443
  _globals['_POPFRAMEREQUEST']._serialized_start=445
  _globals['_POPFRAMEREQUEST']._serialized_end=482
  _globals['_POPFRAMERESPONSE']._serialized_start=484
  _globals['_POPFRAMERESPONSE']._serialized_end=569
  _globals['_PUSHIMAGEREQUEST']._serialized_start=571
  _globals['_PUSHIMAGEREQUEST']._serialized_end=623
  _globals['_PUSHIMAGERESPONSE']._serialized_start=625
  _globals['_PUSHIMAGERESPONSE']._serialized_end=668
  _globals['_POPIMAGEREQUEST']._serialized_start=670
  _globals['_POPIMAGEREQUEST']._serialized_end=707
  _globals['_POPIMAGERESPONSE']._serialized_start=709
  _globals['_POPIMAGERESPONSE']._serialized_end=765
  _globals['_STREAMING']._serialized_start=768
  _globals['_STREAMING']._serialized_end=1322
# @@protoc_insertion_point(module_scope)
