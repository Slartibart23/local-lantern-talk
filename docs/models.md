# Language models

Local Lantern Talk runs language models (LLMs) in the GGUF format with
[llama.cpp](https://github.com/ggml-org/llama.cpp). Open the catalog with
**Language model → Change model…**.

> No model is redistributed by this project. Every model has its own license, shown
> in the catalog. Uncensored and explicit models answer without the usual refusals —
> **you are responsible for how you use them.**

## The catalog

The catalog focuses on **German**, roleplay and assistant use. It lives in
`config\model_catalog.yaml` and can be edited without touching code.

| Model | Variant | Purpose | Q4_K_M | Notes |
|---|---|---|---|---|
| Gemma 4 26B-A4B Heretic | uncensored | Roleplay & assistant | 16.8 GB | Best German in the catalog; mixture-of-experts, stays fast when partly on the CPU |
| Cydonia 24B v4.3 | uncensored | Roleplay | 14.3 GB | Community favorite, lots of dialogue |
| Magidonia 24B v4.3 | uncensored | Roleplay | 14.3 GB | Cydonia's sibling on the Magistral base |
| Magidonia 24B v4.3 Heretic | uncensored | Roleplay | 14.3 GB | Magidonia with refusals removed |
| Skyfall 31B v4.2 | uncensored | Roleplay | 19.0 GB | Larger; for 32 GB GPUs |
| Qwen 3.8 27B Abliterated | uncensored | Assistant & roleplay | 16.6 GB | Strong all-rounder; dense model, best on 32 GB GPUs |
| Gemma 4 12B Heretic | uncensored | Roleplay & assistant | 7.4 GB | Good German, for smaller GPUs |
| Rocinante 12B v1.1 | uncensored | Roleplay | 7.5 GB | Classic; clearly strongest in English |
| Qwen 3.6 27B Uncensored Aggressive | explicit | Adult roleplay & assistant | 17.5 GB (Q4_K_P) | HauhauCS build: answers directly, without preamble; own K_P quantizations |
| Qwen 3.6 27B Abliterated | explicit | Adult roleplay & assistant | 16.5 GB | Strong all-rounder; small context cache, so long contexts are cheap |
| Mistral Small 3.2 24B Abliterated v2 | explicit | Adult roleplay & assistant | 14.3 GB | Mistral Small 3.2 with refusals removed |
| Mistral Small 24B (2501) Abliterated | explicit | Adult roleplay & assistant | 14.3 GB | The earlier Mistral Small, refusals removed |
| Dolphin 3.0 Mistral 24B | explicit | Adult roleplay & assistant | 14.3 GB | Follows your instructions without refusals; mostly trained in English |
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
- **Labels:** *censored* = the original releases, careful and precise; *uncensored* =
  community builds and roleplay finetunes without the usual refusals; *explicit* =
  models chosen for adult content. **Family mode** (Settings) hides uncensored and
  explicit models.
- **Thinking is switched off.** Some models reason before answering; that would
  delay every spoken reply by seconds. The app disables it, and anything that still
  looks like reasoning (`<think>…`) is never spoken or shown.

## Quantization and GPU memory

Each model comes in several **quantizations**: Q6_K (best, largest), Q5_K_M, Q4_K_M
(smallest). HauhauCS' models use their own *K_P* quantizations (Q4_K_P, Q5_K_P,
Q6_K_P), which are a little larger and better than the K_M of the same number. The
catalog picks one automatically:

1. Start with the quantization of your hardware profile (`ultra`: Q6_K, `high`: Q5_K_M,
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

### Measured on a desktop RTX 5090 (32 GB)

With Whisper, OmniVoice, Smart Turn and Windows loaded (2.3 GB in use before the start),
about **21.4 GB** remain for a language model.

| Model | Context (Automatic) | Placement | First audio after a typed message |
|---|---|---|---|
| Gemma 4 12B Q5_K_M | 32 768 tokens | fully on the GPU (9.7 GB) | about 1.3 s |

The 24–27B models fit completely in Q5 or Q6. Values for other GPUs are estimates until
someone measures them.

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

Give the model a **label** (censored / uncensored / explicit / unknown); you can change
it later in the catalog. Your models are stored in `user_data\custom_models.json`.

**Copying files by hand:** GGUF files you put into `models\llm` appear in the catalog
automatically with the label "unlabeled", which you can change.

**Gated models:** some repositories require accepting a license on Hugging Face.
Accept it on the model page, create an access token, and set it as the environment
variable `HF_TOKEN` before running `start.bat`.

## Context length

The context is the model's short-term memory: everything it sees at once — the
character's description, the rules, its memory and the recent conversation. About half
of it is kept for the conversation. Set it in **Settings → Language model**:

- **Automatic** (the default) loads the largest context that fits completely on the GPU,
  up to the limit of your hardware profile (`ultra` / `high`: 32 768 tokens, `mid`:
  16 384, `low`: 8 192, `unsupported`: 4 096). On a 16 GB GPU it therefore picks a
  shorter context than on a 32 GB GPU, without any setting.
- Switch Automatic off to choose a value yourself with the **slider** (2 048 tokens up to
  the model's maximum, in steps of 1 024). The bar under it shows for the loaded model:
  **green** fits completely on the GPU (fast), **yellow** part of the model moves to the
  CPU (noticeably slower), **red** too big.
- How much memory a context needs depends strongly on the model: Gemma 4 and Qwen 3.6
  need little, Mistral-based 24B models several GB for 32 k tokens. The app measures it
  for each model with llama.cpp (three quick checks the first time a model is loaded),
  so the zones are exact for your GPU, the chosen quantization and whether voice output
  is on.
- Changing the value reloads the model (a few seconds). Longer conversations are
  summarized into the character's memory (see [memory.md](memory.md)); a longer context
  keeps more of the chat word for word and makes the summary longer.
