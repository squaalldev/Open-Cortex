llama-server:
	llama-server \
	-hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M \
	-c 2048 -t 8 -tb 8 \
	--metrics \
	--host 127.0.0.1 --port 8080

llama-cli:
	llama-cli \
	-hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M \
	-c 2048 -t 8 \
	-n 128 -cnv