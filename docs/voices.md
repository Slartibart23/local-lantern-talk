# Voices

Local Lantern Talk speaks with [OmniVoice](https://github.com/k2-fsa/OmniVoice). You can
use one of the 8 built-in voices or clone a voice from a short recording.

> **Consent:** only clone a voice with the explicit consent of the person it belongs to.
> Never use a cloned voice for impersonation, fraud or deception.

## Your own voices (voice cloning)

### Adding a voice

- In the app: **Voice → Add voice…** and pick an audio file. It is copied into
  `user_data\voices\` and selected right away.
- Or copy the file into `user_data\voices\` yourself and reload the page.

Supported formats: **MP3, WAV, FLAC, OGG, OPUS, M4A, AAC, WEBM** (up to 50 MB) — so
messenger voice notes (`.opus`) and phone recordings (`.m4a`) work directly. The file
name becomes the voice name. Your voices stay in `user_data\` and are never part of
the repository or of updates.

When a voice is used for the first time, the app prepares the clip with ffmpeg:
silence at the start and end is removed, the volume is evened out (loudness
normalization), and the audio is converted to the format OmniVoice expects. The
result is cached in `user_data\voices\.cache` and redone automatically when you
replace the file.

### Managing voices

Select a voice in the side panel, then:

- **▶ Preview** speaks a short sample sentence in the voice's language (German,
  English, Spanish or French; for your own voices the language you chose or the language
  of its transcript).
- **Rename…** changes the name (and the file name). The prepared voice is kept, so
  renaming is instant.
- **Delete** removes the audio file and its transcript. If it was the selected voice,
  a built-in voice takes over.
- **Language of the clip** (below the reference text): the language spoken in the clip.
  *Detect automatically* guesses it from the transcript (German, English, Spanish or
  French). Choosing it yourself helps Whisper transcribe the clip correctly — the clip
  is transcribed again with that language — and is used for the preview and for replies
  whose language is not known. It is saved as `clip.lang` next to `clip.mp3`.

### Quality hints

After adding a clip, and under the selected voice, the app shows hints when the clip
is too short or too long, has **noticeable background noise or reverb**, or is
**distorted** (recorded too loud). These are best-effort checks; your ears decide.

### How long should the clip be?

| Speech in the clip | Result |
|---|---|
| **under 3 s** | Too little of the voice to copy: the clone sounds generic. The app shows a warning. |
| 3 – 10 s | Works, but the clone catches less of the voice. The app shows a hint. |
| **10 – 13 s** | **Recommended.** Best similarity; the clip still fits completely into the 15 s that are used without your own transcript. |
| 13 – 20 s | Works; rarely sounds better and makes every reply slightly slower. The app shows a hint. |
| **over 20 s** | Without your own transcript, only the **first 15 s** are used. With your own transcript the whole clip is used, but OmniVoice becomes slower, needs more memory and usually clones *worse*. Cut the clip instead. |

Only speech counts: pauses longer than about 0.2 s and silence at the start and end
are removed automatically.

### What makes a good reference clip

The clone copies everything it hears — including the room, the microphone and the
compression of the file. A clean clip matters more than any quality setting.

- **One speaker only**, no other voices in the background.
- **Quiet room**, no music, TV, traffic or keyboard noise.
- **Little echo**: a furnished room or a closet is better than a bare room or bathroom.
- **Microphone close** to the mouth (about 20–30 cm), normal speaking volume,
  nothing distorted or clipped.
- **Good audio quality**: WAV or a high-bitrate MP3. Phone calls and messenger voice
  notes (e.g. WhatsApp) are heavily compressed, and the clone inherits that muffled,
  "phone-like" sound.
- **Natural, continuous speech** in the **style you want to hear**: a calm clip gives a
  calm voice, an excited clip an excited one.
- **Same language** as the replies. A German clip speaking English replies keeps a
  German accent (and vice versa).
- **Complete sentences**, not cut off in the middle of a word; avoid laughing,
  coughing or long "ähm"s.

### The reference text (transcript)

OmniVoice needs to know **exactly what is said in the clip**. By default the app
transcribes the clip with Whisper the first time the voice is used. Whisper can mishear
words — especially names, dialect or unclear speech — and a wrong transcript makes the
clone noticeably worse.

**Checking and correcting it in the app:** select the voice; the **Reference text**
field shows the transcript. *from Whisper — please check* means it was generated
automatically. Correct it and click **Save text**; the voice is rebuilt with your text.
**Use Whisper** discards your text again.

**Providing it without Whisper:** put a text file with the **same name** next to the
clip *before* you use the voice, e.g.

```
user_data\voices\Anna.mp3
user_data\voices\Anna.txt    ← exactly what is said in Anna.mp3
```

If this file exists, Whisper is not used for that clip at all. Tips for the text:

- Write down **every word** as spoken, including filler words and repetitions.
- Write numbers the way they are spoken ("twenty-five", not "25") if unsure.
- Normal punctuation helps; one line or several lines both work.
- Save it as UTF-8 (the default in Windows Notepad).
- With your own transcript the clip is **never shortened**, so the text must cover the
  whole recording — keep such clips around 10–13 s.

## Built-in voices

Eight voices — German, English, Spanish and French, each male and female — come with
the app as short reference clips in `config\builtin_voices\`. They were created with
OmniVoice *voice design* (synthetic voices, not recordings of real people), so every
installation sounds the same. Their descriptions are in `config\builtin_voices.yaml`;
the prepared voice files are in `models\builtin_voices\`.

**Settings → Voice output → Regenerate built-in voices** designs all eight anew on your
GPU. Voice design produces a slightly different voice each time, so only do this if you
do not like the current ones; it takes about a minute and needs voice output to be on.
To go back to the shipped voices, close the app and delete `models\builtin_voices\`.

With **Match built-in voice to spoken language** (on by default), a built-in voice
switches to the built-in voice of the same gender in the conversation language, so a
German reply is never spoken with an English accent. Your own voices are never
switched.

The **conversation language** (right panel, above the voice) is English by default.
Choose another language, or *Automatic* to follow the language you speak or type. A
fixed language is also passed to speech recognition, so a short word or background
noise is never mistaken for another language. In *Automatic*, an unsure first guess
falls back to English.

## Speech quality

**Settings → Voice output** sets OmniVoice's synthesis steps (16–64, default
32) and precision (fp16 / bf16 / fp32, default fp16). More steps and fp32 can sound
slightly cleaner but delay every reply — on an RTX 5090 Laptop GPU, 64 steps add about
1 s before the reply starts. There is only one official OmniVoice model; these settings
are its quality controls.
