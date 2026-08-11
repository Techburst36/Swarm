with open("gguf_tensor_loader.py", "r") as f:
    code = f.read()

start_str = "    def _detect_config(self) -> ModelConfig:"
end_str = "    def __repr__(self) -> str:"

start_idx = code.find(start_str)
end_idx = code.find(end_str)

new_method = """    def _detect_config(self) -> ModelConfig:
        cfg = ModelConfig()

        meta_block_count = self._reader.get_metadata("olmo.block_count")
        if meta_block_count is None:
            meta_block_count = self._reader.get_metadata("llama.block_count")
        if meta_block_count is not None:
            cfg.num_layers = int(meta_block_count)
        else:
            max_layer = -1
            for tname in self._reader.list_tensors():
                if tname.startswith("blk."):
                    try:
                        max_layer = max(max_layer, int(tname.split(".")[1]))
                    except: pass
            if max_layer >= 0:
                cfg.num_layers = max_layer + 1

        t_header = self._reader.header.tensors

        if "token_embd.weight" in t_header:
            dims = t_header["token_embd.weight"].dims
            cfg.vocab_size = dims[0]
            if len(dims) > 1:
                cfg.hidden_dim = dims[1]

        if "blk.0.ffn_gate.0.weight" in t_header:
            cfg.intermediate_dim = t_header["blk.0.ffn_gate.0.weight"].dims[0]

        cfg.tied_embeddings = "output.weight" not in t_header

        if "blk.0.attn_q.weight" in t_header:
            q_rows = t_header["blk.0.attn_q.weight"].dims[0]
            if cfg.hidden_dim > 0:
                for possible_hd in (128, 96, 80, 64):
                    if q_rows % possible_hd == 0 and cfg.hidden_dim % possible_hd == 0:
                        cfg.head_dim = possible_hd
                        cfg.num_heads = q_rows // possible_hd
                        break

        if "blk.0.attn_k.weight" in t_header:
            k_rows = t_header["blk.0.attn_k.weight"].dims[0]
            if cfg.head_dim > 0:
                cfg.num_kv_heads = k_rows // cfg.head_dim

        return cfg

"""
if start_idx != -1 and end_idx != -1:
    code = code[:start_idx] + new_method + code[end_idx:]
    with open("gguf_tensor_loader.py", "w") as f:
        f.write(code)
    print("Successfully patched tensor dimension logic!")
else:
    print("Could not find the function boundaries.")
