# Session recording

Local Lantern Talk can record each chat as **one MP3 of the whole dialogue** (your
voice and the character's voice, alternating) plus a **Markdown transcript** with
timestamps. Everything stays on your computer in `user_data\sessions\`.

## What is a session?

A session is one chat with one character: it starts with your first message and ends
when you click **End chat**. If you come back to the character later (after switching
characters or restarting the app) and continue the chat, the recording continues in
the same files. **Start new chat** or **End chat** starts a new session.

Files are named after the start time and the character:

```
user_data\sessions\2026-10-13_1538_lena.mp3
user_data\sessions\2026-10-13_1538_lena.md
```

## Turning it on and off

- **Globally**: right panel → *Session* → *Record sessions* (on by default).
- **Per character**: character editor → *Record sessions*. A character with this switched
  off is never recorded, even when recording is on globally.

The *Session* section shows the files of the session being recorded, and a red **REC**
badge in the header shows that recording is active. *Open session folder* opens
`user_data\sessions\` in Explorer, where you can copy, share or delete recordings.

## The MP3

- One file for the whole dialogue, in order. Your speech comes from the full-quality
  microphone recording, not from the reduced copy used for speech recognition.
- Waiting time (while the model thinks) is not recorded. Turns are separated by a short
  pause of 0.5 s.
- User and AI speech are brought to a common loudness.
- Mono, 64 kbit/s, 24 kHz: about 29 MB per hour.
- Each utterance is appended as soon as it is complete. If the app or the computer
  crashes, the file is intact up to the last complete utterance.
- Encoding runs in the background on the CPU and never delays the conversation.

The pause and the bitrate can be changed in `user_data\settings.json`
(`session_gap_ms`, `session_bitrate_kbps`).

## The transcript

A Markdown file you can open in any text editor. Each utterance starts with its
**position in the MP3**, the speaker and some details:

```markdown
### [00:12.4] You · spoken · de · 3.1 s

Erzähl mir von deiner Taverne.

### [00:16.0] Elara · German female · 6.8 s

_[not spoken: wischt den Tresen ab]_ Ach, die Goldene Laterne …
```

- Typed messages and replies without voice output are marked `typed` / `text only` /
  `no audio`. Their timestamp is the position in the MP3 at that moment.
- Actions appear according to the character's *Show actions as text* setting, marked
  `_[not spoken: …]_`. Model reasoning is never written.

## Replay

- Every recorded message in the chat has a **▶ 0:12** button. It plays that message
  from the session MP3 and stops at its end. Click it again (or press Esc) to stop.
- The microphone stays muted while a recording plays, so the app never hears itself.
- **Recent sessions** in the left panel lists past sessions with their length. Clicking
  one opens it **read-only**, with a replay button for each message. *Back to chat*
  returns to the conversation.
