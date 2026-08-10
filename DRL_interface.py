from noise_ppl import inference_with_ppl

noise_ratio_dict = {
        '0': 0.08,
        '31': 0.01,
}

ppl = inference_with_ppl(noise_ratio_dict)
print(ppl)

ppl = inference_with_ppl(noise_ratio_dict)
print(ppl)
