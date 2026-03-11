import pathlib
p = pathlib.Path('/sgl-workspace/sglang/python/sglang/srt/layers/quantization/quark/schemes/__init__.py')
content = p.read_text()
content = content.replace(
    'from .quark_w4a4_mxfp4 import QuarkW4A4MXFP4',
    'try:\n    from .quark_w4a4_mxfp4 import QuarkW4A4MXFP4\nexcept Exception:\n    QuarkW4A4MXFP4 = None'
)
p.write_text(content)
print('Patched:', p)
