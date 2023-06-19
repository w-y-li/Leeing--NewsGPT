import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
def prepare(pre_seq_len=64):
    tokenizer = AutoTokenizer.from_pretrained("E:\\westlake\\chatglm-6b-int4", trust_remote_code=True)
    config = AutoConfig.from_pretrained("E:\\westlake\\chatglm-6b-int4", trust_remote_code=True,
                                        pre_seq_len=pre_seq_len)
    model = AutoModel.from_pretrained("E:\\westlake\\chatglm-6b-int4", config=config, trust_remote_code=True)
    prefix_state_dict = torch.load(os.path.join('./chatglm_ptuning/output/checkpoint-3000', "pytorch_model.bin"))
    new_prefix_state_dict = {}

    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

    # Comment out the following line if you don't use quantization
    model = model.quantize(4)
    model = model.half().cuda()
    model.transformer.prefix_encoder.float()
    model = model.eval()
    return model,tokenizer

def chat(tokenizer,user_text,history,model):
    response, history = model.chat(tokenizer, user_text, history=history)

    return response,history

