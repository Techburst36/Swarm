import re

with open("gguf_tensor_loader.py", "r") as f:
    code = f.read()

pattern = re.compile(r"    def read_tensor_bytes\(self, name: str\) -> bytes:.*?return self\._cache\[name\]", re.DOTALL)

new_func = """    def read_tensor_bytes(self, name: str) -> bytes:
        if name not in self._cache:
            raw = self._reader.read_tensor_bytes(name)
            info = self._reader.header.tensors[name]
            # Auto-upcast F16 (ggml_type 1) to F32 so our Python backbone can read it
            if info.ggml_type == 1:
                import numpy as np
                raw = np.frombuffer(raw, dtype=np.float16).astype(np.float32).tobytes()
            self._cache[name] = raw
        return self._cache[name]"""

if pattern.search(code):
    code = pattern.sub(new_func, code, count=1)
    with open("gguf_tensor_loader.py", "w") as f:
        f.write(code)
    print("F16 automatic upcasting patch applied successfully!")
else:
    print("Failed to find the function in gguf_tensor_loader.py")
