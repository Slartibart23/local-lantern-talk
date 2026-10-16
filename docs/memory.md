# Memory

Characters remember previous conversations, even after the app has been closed. The
memory is deliberately simple and transparent: a history, a short summary and a list
of facts per character, all visible and editable in the right panel under **Memory**.
Everything is stored locally in `user_data\memory\memory.db` (SQLite).

## How it works

- **History**: every message is saved. When you switch back to a character or restart
  the app, the conversation continues where it stopped.
- **Rolling summary**: when the history gets too long for the language model's context
  (about half of the context is reserved for it), the oldest exchanges are summarized
  by the model. The summary replaces them in the prompt; the most recent messages are
  always kept word for word. The summary grows with the context: about 200 words at
  8 192 tokens, 400 at 16 384, 800 at 32 768 and at most 1000.
- **Facts**: every 4 replies the model is asked, in a separate short request, which
  durable facts the conversation revealed: names, preferences, relationships and
  important events. Facts that turned out to be wrong are removed. Up to 30 facts are
  kept per character.
- **End chat** (top right of the conversation) starts over with the character's
  greeting. The previous chat is first folded into the summary and the facts, so the
  character still knows what happened.
- **Continue or start new**: after a restart, or when you switch back to a character,
  its last chat is loaded and a bar offers *Continue* or *Start new chat*. Simply
  talking on also continues it.

Facts and the summary are added to the character's instructions in every request. The
character therefore "knows" them without reading the whole old conversation.

### How much of a chat is remembered word for word?

That depends on the context length (**Settings → Language model**). About half of it
holds the conversation:

| Context | Word for word | Roughly |
| --- | --- | --- |
| 8 192 tokens | about 2 000 words | 10–20 exchanges |
| 16 384 tokens | about 4 000 words | 25–40 exchanges |
| 32 768 tokens | about 8 000 words | 50–80 exchanges |

A chat can go on as long as you like; older parts live on in the summary and the facts.
With *Automatic* the app picks the longest context that fits on your GPU. The slider
shows in green, yellow and red what fits for the loaded model.

Memory upkeep runs between replies and takes a second or two. If you speak during that
time, your words are handled right after it.

## Editing the memory

In the right panel under **Memory**:

- **Add a fact**: type it and press *Add*, for example "My name is Tom" or "I'm
  vegetarian". Facts you add yourself are never removed by the model.
- **✎** edits a fact, and **✕** deletes it.
- **Summary of earlier conversations** shows the current summary.
- **Reset this character's memory…** deletes the history, the summary and all facts
  of the active character. Other characters are not affected.
- **Deleting messages** in the chat (🗑) removes them from the history too. Facts the
  model already learned from them stay in the list until you delete them there.

## Turning memory off

Switch off *Remember conversations* in the character editor. The character then
forgets everything when you end the chat or restart the app, and no new
messages are stored. Memory that was stored earlier is kept, not used, and comes back
when you switch memory on again. To remove it, use *Reset this character's memory…*.

## Privacy

The memory never leaves your computer. `user_data\` is excluded from the repository,
and chat text, characters and facts are never written to the log files.
