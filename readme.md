# MultifacetEval

Code of the IJCAI 2024 paper "MultifacetEval: Multifaceted Evaluation to Probe LLMs in Mastering Medical Knowledge"

# Requirements
>google-ai-generativelanguage==0.4.0\
>google-api-core==2.15.0\
>google-generativeai==0.3.1\
>googleapis-common-protos==1.62.0\
>huggingface-hub==0.20.2\
>numpy==1.25.1\
>openai==0.27.8\
>scikit-learn==1.3.0\
>scipy==1.11.1\
>tokenizers==0.15.0\
>torch==2.0.1\
>tqdm==4.65.0\
>transformers==4.36.2
# Experiments
Please follow the instruction below to reimplement our experiments. Results can be found in the *results/medqa* directory.
## GPT-3.5-turbo
Please fill your API key (and API base) into the corresponding code first. Note that running all experiments of GPT-3.5-turbo under the CoT+SC setting will be very **expensive** (~2400$).
### Answer-only
```
python evaluate_gpt_medqa_ao.py
```
### CoT+SC
```
python evaluate_gpt_medqa_cotsc.py
```
## Gemini-pro
Please fill your API key into the corresponding code first. The usage of Gemini-pro can be found [here](https://ai.google.dev).
### Answer-only
```
python evaluate_gemini_medqa_ao.py
```
### CoT+SC
```
python evaluate_gemini_medqa_cotsc.py
```
## HuggingFace Models
You may first download the corresponding LLMs in [here](https://huggingface.co). Some LLMs (llama2, med42, etc.) may require certification. Please make sure that you have the corresponding certification.
### Answer-only
```
CUDA_VISIBLE_DEVICES=X python evaluate_hf_medqa_ao.py --model [model path] --model_name [model name]
```
### CoT+SC
```
CUDA_VISIBLE_DEVICES=X python evaluate_hf_medqa_cotsc.py --model [model path] --model_name [model name]
```
## Results Analysis

For Answer-only setting, run

```
bash eval_ao.sh
```

For CoT+SC, run the following command instead

```
bash eval_cot.sh
```



# Resources Requirements

>7B inference: a single RTX4090 (24GB)\
>13B inference: 2 RTX4090 (24GB) or 1 A800 (80GB)\
>70B inference: 2 A800 (80GB)
