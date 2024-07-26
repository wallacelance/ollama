package convert

import (
	"strings"

	"github.com/ollama/ollama/llm"
)

type commandr struct {
	Parameters
	MaxPositionEmbeddings uint32  `json:"max_position_embeddings"`
	HiddenSize            uint32  `json:"hidden_size"`
	HiddenLayers          uint32  `json:"num_hidden_layers"`
	IntermediateSize      uint32  `json:"intermediate_size"`
	NumAttentionHeads     uint32  `json:"num_attention_heads"`
	NumKeyValueHeads      uint32  `json:"num_key_value_heads"`
	LayerNormEPS            float32 `json:"layer_norm_eps"`
	RopeTheta             float32 `json:"rope_theta"`
	UseQKNorm		   bool `json:"use_qk_norm"`
	ModelMaxLength 	  uint32  `json:"model_max_length"`
	LogitScale 		  float32 `json:"logit_scale"`
}

var _ Converter = (*commandr)(nil)

func (p *commandr) KV(t *Tokenizer) llm.KV {
	kv := p.Parameters.KV(t)
	kv["general.architecture"] = "command-r"
	kv["general.name"] = "command-r"
	kv["command-r.context_length"] = p.MaxPositionEmbeddings
	kv["command-r.embedding_length"] = p.HiddenSize
	kv["command-r.block_count"] = p.HiddenLayers
	kv["command-r.feed_forward_length"] = p.IntermediateSize
	kv["command-r.attention.head_count"] = p.NumAttentionHeads
	kv["command-r.attention.head_count_kv"] = p.NumKeyValueHeads
	kv["command-r.attention.layer_norm_eps"] = p.LayerNormEPS
	
	kv["tokenizer.ggml.eot_token_id"] = uint32(107)
	kv["tokenizer.ggml.middle_token_id"] = uint32(68)
	kv["tokenizer.ggml.prefix_token_id"] = uint32(67)
	kv["tokenizer.ggml.suffix_token_id"] = uint32(69)

	return kv
}

func (p *commandr) Tensors(ts []Tensor, nameFunc NameFunc) []*llm.Tensor {
	var out []*llm.Tensor
	for _, t := range ts {
		name := nameFunc(t.Name())
		out = append(out, &llm.Tensor{
			Name:     name,
			Kind:     t.Kind(),
			Shape:    t.Shape(),
			WriterTo: t,
		})
	}

	return out
}

func (p *commandr) tensorName(n string) string {
	return strings.NewReplacer(
		"lm_head", "output",
		"model.embed_tokens", "token_embd",
		"model.norm", "output_norm",
		"model.layers", "blk",
		"input_layernorm", "attn_norm",
		"self_attn.q_proj", "attn_q",
		"self_attn.k_proj", "attn_k",
		"self_attn.v_proj", "attn_v",
		"self_attn.o_proj", "attn_output",
		"mlp.gate_proj", "ffn_gate",
		"mlp.down_proj", "ffn_down",
		"mlp.up_proj", "ffn_up",
		"post_attention_layernorm", "ffn_norm",
		// mixtral
		"block_sparse_moe.gate", "ffn_gate_inp",
	).Replace(n)
}