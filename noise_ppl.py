from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

def compute_ppl(
        predictions, model, tokenizer, batch_size: int = 4, add_start_token: bool = True, max_length=None
    ):

    # model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    # tokenizer = AutoTokenizer.from_pretrained(model_id)

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    )

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0))
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

def add_noise_hooks(model, noise_ratio_dict):
    """
    :param noise_ratio_dict: 噪声配置字典，格式 {层号x(str): 噪声比例p(float)}，x到x+1的activation的p占比的元素置零
    """
    handles = []
    
    # 遍历所有需要添加噪声的层
    for layer_str, p in noise_ratio_dict.items():
        layer_idx = int(layer_str)
        target_layer = model.model.layers[layer_idx]

        # 定义噪声添加函数
        def add_noise(module, input, output, p=p):
            if isinstance(output, tuple):
                # 处理包含隐藏状态和其他输出的情况
                hidden_states = output[0]
                mask = torch.rand_like(hidden_states) > p
                mask = mask.to(hidden_states.dtype).to(hidden_states.device)
                noisy_hidden = hidden_states * mask
                return (noisy_hidden,) + output[1:]
            else:
                # 处理单个张量输出
                mask = torch.rand_like(output) > p
                return output * mask.to(output.dtype).to(output.device)

        handle = target_layer.register_forward_hook(add_noise)
        handles.append(handle)
    
    return handles

model_file = '/home/super/AIModel/Llama-2-7b-hf'
model = AutoModelForCausalLM.from_pretrained(model_file, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_file)

input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:50]
input_texts = [s for s in input_texts if s!='']


def inference_with_ppl(noise_ratio_dict):
    handles = add_noise_hooks(model, noise_ratio_dict)
    result = compute_ppl(
        predictions=input_texts,
        model=model,
        tokenizer=tokenizer,
        batch_size = 4,
        max_length=512
    )
    # print(result['mean_perplexity'])
    return result['mean_perplexity']




if __name__ == '__main__':
    model_file = '/home/super/AIModel/Llama-2-7b-hf'
    model = LlamaForCausalLM.from_pretrained(model_file, device_map='auto')
    tokenizer = LlamaTokenizer.from_pretrained(model_file)

    # 格式 {层号x(str): 噪声比例p(float)}，x到x+1的activation的p占比的元素随机置零
    noise_ratio_dict = {
        '0': 0.08,
        '3': 0.01,
    }
    handles = add_noise_hooks(model, noise_ratio_dict)

    input_texts = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"][:50]
    input_texts = [s for s in input_texts if s!='']

    result = compute_ppl(
        predictions=input_texts,
        model=model,
        tokenizer=tokenizer,
        batch_size = 4,
        max_length=512
    )
    print(result['mean_perplexity'])

