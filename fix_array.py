with open("gguf_stream_reader.py", "r") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("    def _read_value(self"):
        start_idx = i
    if start_idx != -1 and line.startswith("# ── GGUF writer"):
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    new_lines = lines[:start_idx] + [
        "    def _read_value(self, f, value_type: int):\n",
        "        if value_type == _GGUF_TYPE_STRING:\n",
        "            return self._read_string(f)\n",
        "        elif value_type == _GGUF_TYPE_ARRAY:\n",
        "            elem_type = struct.unpack('<I', f.read(4))[0]\n",
        "            length = struct.unpack('<Q', f.read(8))[0]\n",
        "            if elem_type == _GGUF_TYPE_STRING:\n",
        "                return [self._read_string(f) for _ in range(length)]\n",
        "            reader = _GGUF_VALUE_READERS.get(elem_type)\n",
        "            if reader is None:\n",
        "                return None\n",
        "            return [reader(f) for _ in range(length)]\n",
        "        else:\n",
        "            reader = _GGUF_VALUE_READERS.get(value_type)\n",
        "            if reader is None:\n",
        "                return None\n",
        "            return reader(f)\n\n\n"
    ] + lines[end_idx:]
    with open("gguf_stream_reader.py", "w") as f:
        f.writelines(new_lines)
    print("Array parser successfully patched!")
else:
    print("Could not find boundaries.")
