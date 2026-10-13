# Language models

Local Lantern Talk runs language models (LLMs) in the GGUF format with
[llama.cpp](https://github.com/ggml-org/llama.cpp). Open the catalog with
**Language model → Change model…**.

> No model is redistributed by this project. Every model has its own license, shown
> in the catalog. Uncensored models answer without the usual refusals — **you are
> responsible for how you use them.**

## The catalog

The catalog focuses on **German**, roleplay and assistant use. It lives in
`config\model_catalog.yaml` and can be edited without touching code.

| Model | Variant | Purpose | Q4_K_M | Notes |
|---|---|---|---|---|
| Gemma 4 26B-A4B Heretic | uncensored | Roleplay & assistant | 16.8 GB | Best German in the catalog; mixture-of-experts, stays fast when partly on the CPU |
| Cydonia 24B v4.3 | uncensored | Roleplay | 14.3 GB | Community favorite, lots of dialogue |
| Magidonia 24B v4.3 | uncensored | Roleplay | 14.3 GB | Cydonia's sibling on the Magistral base |
| Skyfall 31B v4.2 | uncensored | Roleplay | 19.0 GB | Larger; for 32 GB GPUs |
| Gemma 4 12B Heretic | uncensored | Roleplay & assistant | 7.4 GB | Good German, for smaller GPUs |
| Rocinante 12B v1.1 | uncensored | Roleplay | 7.5 GB | Classic; clearly strongest in English |
| Gemma 4 26B-A4B | censored | Assistant | 17.0 GB | Very good German |
| Mistral Small 3.2 24B | censored | Assistant | 14.3 GB | Good German, French, Spanish |
| Gemma 4 12B | censored | Assistant | 7.1 GB | Fast; the installer's starter model |

How the list was chosen: community ratings on the
[UGI leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)
(writing quality, willingness to answer, dialogue share) and how widely the models
are used on Hugging Face, with a preference for good German.

Good to know:

- **Roleplay finetunes** (Cydonia, Magidonia, Skyfall, Rocinante) are strongest in
  English. Their German is good (Mistral Small based) or weaker (Rocinante). For
  German, Gemma 4 is the best choice.
- **Uncensored builds** ("abliterated", "heretic") are made by the community from
  the original models and can be slightly less precise than the originals.
- **Thinking is switched off.** Some models reason before answering; that would
  delay every spoken reply by seconds. The app disables it, and anything that still
  looks like reasoning (`<think>…`) is never spoken or shown.

## Quantization and GPU memory

Each model comes in several **quantizations**: Q6_K (best, largest), Q5_K_M, Q4_K_M
(smallest). The catalog picks one automatically:

1. Start with the quantization of your hardware profile (`high`: Q5_K_M,
   `mid` / `low`: Q4_K_M — see [installation.md](installation.md)).
2. Go down until the model fits completely into the free GPU memory.
3. A quantization you already downloaded is used if it fits at least partly.

You can choose another quantization per model in the catalog.

The free memory already accounts for what the speech components need: with voice
output on, about 3.5 GB are kept free for OmniVoice, which grows while speaking.
Each model shows one of:

| Badge | Meaning |
|---|---|
| **fits on the GPU** | Fully on the GPU — fastest. |
| **partly on the CPU (slower)** | Part of the model runs on the CPU. It works, but replies take longer; a smaller quantization would be faster. Mixture-of-experts models (Gemma 4 26B-A4B) handle this well because only the expert weights move to the CPU. |
| **too big for this GPU** | Cannot be downloaded from the catalog; choose a smaller quantization or model. |

Before loading, the app asks llama.cpp exactly how to split the model and applies
that split, so the notice under the current model always matches what runs.

### Measured on an RTX 5090 Laptop GPU (24 GB)

With Whisper, OmniVoice and Windows loaded, about **14.7 GB** remain for a language
model.

| Model | Placement | First audio after a typed message |
|---|---|---|
| Gemma 4 12B Q5_K_M | fully on the GPU (8.8 GB) | about 1.3–2.2 s |
| Gemma 4 26B-A4B Heretic Q4_K_M | experts of 5–6 of 30 blocks on the CPU | about 2.2 s |

On 32 GB GPUs (desktop RTX 5090) the 24–26B models fit completely. Values for other
GPUs are estimates until someone measures them.

## Downloads

- **Download & load** downloads only the one GGUF file of the chosen quantization
  into `models\llm` and loads it when finished.
- Downloads can be **cancelled** and continue where they stopped — also after
  closing the app (the partial file is kept as `<file>.part`).
- Only one download runs at a time. The app checks free disk space first and the
  file size at the end.
- **Delete** removes a downloaded file. The loaded model cannot be deleted; load
  another one first.

## Your own models

In **Add your own model** enter one of:

| Input | Example |
|---|---|
| Hugging Face repo id | `TheDrummer/Cydonia-24B-v4.3-GGUF` |
| Hugging Face page or file link | `https://huggingface.co/owner/name/blob/main/model-Q4_K_M.gguf` |
| Direct link to a `.gguf` file | `https://example.org/model.Q4_K_M.gguf` |

For a repository, the app picks one GGUF file: the quantization you chose, or with
**Auto** the profile's quantization (or the next smaller one). Only that file is
downloaded — repositories are never cloned. Split GGUFs (`-00001-of-00003.gguf`)
are not supported; pick a quantization that comes as a single file.

Give the model a **label** (censored / uncensored / unknown); you can change it later
in the catalog. Your models are stored in `user_data\custom_models.json`.

**Copying files by hand:** GGUF files you put into `models\llm` appear in the catalog
automatically with the label "unlabeled", which you can change.

**Gated models:** some repositories require accepting a license on Hugging Face.
Accept it on the model page, create an access token, and set it as the environment
variable `HF_TOKEN` before running `start.bat`.

## Context length

The default context is 8192 tokens (about 6,000 words of conversation). Longer
conversations are summarized by the memory feature in a later version.
