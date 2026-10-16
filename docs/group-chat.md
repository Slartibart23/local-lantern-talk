# Group chats and story mode

Up to **four characters** can be in one chat. They are equal partners: each keeps its own
personality, voice, look and way of speaking, and all of them use the **one language model
that is loaded** (there is never a second model in memory, so switching speakers costs
nothing).

## Bringing characters together

1. Pick a character on the left as usual.
2. Click **+ Add character** under the header and choose who joins. Repeat for a third or
   fourth character. **✕** on a name removes that character from the chat.
3. The chat now belongs to this group: it has its **own memory** (history, summary and
   facts of the group, separate from the chats with each character alone) and its own
   session recording.

The last groups you used are listed on the left under **Groups**; one click opens the
group chat again where it stopped. Picking a single character on the left goes back to a
chat with that character alone.

The look of the chat (scene, background) follows the character picked first. In the chat,
every reply shows who is speaking, in that character's text colour; their portraits are
on the left, and the one who is speaking glows.

## Two modes

| Mode | What happens |
| --- | --- |
| **Chat** | You lead the conversation. The character you address by name answers ("Max, what do you think?"); otherwise whoever has been quiet the longest. If that reply addresses another character, they may answer too. Click a name in the bar to let that character speak now. |
| **Story** | The characters talk with each other by themselves. **Play** starts, **Pause** stops after the line that is being spoken, **Next line** adds one line and waits. Click a name to choose who speaks next. |

In both modes you can join in at any time by speaking or typing; the character you
address answers, and with *Play* the story goes on afterwards. Story mode pauses by itself
after 50 lines without you.

## Story setup

**Story setup** (story mode) holds what every character knows about this story: where and
when it takes place, what is at stake, secrets or goals. Example:

> A storm traps everyone in the Gilded Lantern overnight. Captain Vargas is secretly looking
> for a stolen star chart; Elara knows who has it, but she wants a price.

It is saved for the group and kept after *End chat*.

## Director's notes in a group

The director's field works as with one character (see
[characters.md](characters.md#directors-notes)), with one addition: the list next to it
chooses **who** the note is for.

- **Everyone**: *Next reply* goes to whoever speaks next; *Pin* applies to all characters.
- **One character**: *Next reply* waits for that character's next line ("Max confesses a
  secret"); *Pin* applies to them only. *Continue ▶* lets exactly this character speak.

Pinned notes of a group belong to the group, not to the characters.

## How each character is told what to do

Every line is written with a system prompt made for the speaker:

1. **Personality and instructions** of the character.
2. **How they speak** — tone, words, habits (character editor).
3. **In a group** — how they behave with others (character editor), for example "plays
   the host and stirs the pot" for Elara or "takes command when things get tense" for
   Captain Vargas.
4. **With you in the scene** — a short introduction of every other character (from their
   profile).
5. In story mode the **story rules** (**Settings → Story mode**, editable; *Built-in rules*
   restores them) and the story setup; in chat mode short group rules.
6. Memory of the group, director's notes, your profile and the speech rules.

The history is shown to every character from its own point of view: its own lines as its
own, everything the others said with their name in front.

### Against repetition

- The story rules ask for something new in every line and forbid repeating or summarizing.
- A sampler against repeated phrases (DRY) is used for every group line.
- In story mode, a character gets a quiet nudge every five lines without a director's note
  ("Bring something unexpected into the scene", "Raise the stakes" …), so the story never
  runs in circles.
- Nobody speaks twice in a row unless you ask for it.

Bigger models (24B and more) keep several characters apart best; with 12B models the
characters sound more alike.

## Timing

Each line takes as long as a normal reply: the model writes, the first sentence is spoken
while the rest is still being written. Measured on an RTX 5090 with Qwen 3.6 27B (Q4): about
1.5 s until the text starts and 6 s until the first audio of a long first sentence.
