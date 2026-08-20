# Qwen3.8 vision boundary

The public vision path contains the native vision tower interface, patch-grid
metadata, image preprocessing and the model-side embedding bridge. PNG, JPEG
and WebP decoding is implemented through the platform libraries.

Video is intentionally a separate capability. The in-memory FFmpeg decoder is
optional and is not part of the default build. When it is not compiled or a
container/codec is not verified, the runtime must return an explicit unsupported
media error. It must not reinterpret an unknown container as a still image or
silently route to a different runtime.
