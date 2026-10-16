# Session recording

Local Lantern Talk can record each chat as **one MP3 of the whole dialogue** (your
voice and the character's voice, alternating) plus a **Markdown transcript** that reads
like a screenplay. Everything stays on your computer in `user_data\sessions\`.

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

A Markdown file you can open in any text editor. It reads like a screenplay: only
who said what, in order.

```markdown
# Elara — 13 Oct 2026, 15:38

Elara: Willkommen in der Goldenen Laterne!

You: Erzähl mir von deiner Taverne.

Elara: Ach, die Goldene Laterne … die gibt es schon seit dreißig Jahren.
```

- Only the spoken words are written. Actions (`*wipes the counter*`), out-of-character
  remarks and model reasoning are left out.
- Typed messages and replies without voice output appear like spoken ones.
- With a name in **Settings → Your profile**, your lines start with your name instead of
  "You". Director's notes are never part of the transcript.
- **Deleting messages** in the chat (🗑, see [characters.md](characters.md#deleting-writing-again-and-editing-messages))
  cuts them off the end of the MP3 and the transcript too. *Write again* replaces the
  last reply in the recording.
- The position of every utterance in the MP3 (for the replay buttons) is kept in a
  hidden file, `user_data\sessions\.index\<session>.jsonl`. If you delete a session's
  `.md` and `.mp3`, this file is removed automatically.

## Replay

- Every recorded message in the chat has a **▶ 0:12** button. It plays that message
  from the session MP3 and stops at its end. Click it again (or press Esc) to stop.
- The microphone stays muted while a recording plays, so the app never hears itself.
- **Recent sessions** in the left panel lists past sessions with their length. Clicking
  one opens it **read-only**, with a replay button for each message. *Back to chat*
  returns to the conversation.
