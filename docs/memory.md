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
  always kept word for word.
- **Facts**: every 4 replies the model is asked, in a separate short request, which
  durable facts the conversation revealed: names, preferences, relationships and
  important events. Facts that turned out to be wrong are removed. Up to 30 facts are
  kept per character.
- **New conversation** starts over with the character's greeting. The previous
  conversation is first folded into the summary and the facts, so the character still
  knows what happened.

Facts and the summary are added to the character's instructions in every request. The
character therefore "knows" them without reading the whole old conversation.

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

## Turning memory off

Switch off *Remember conversations* in the character editor. The character then
forgets everything when you start a new conversation or restart the app, and no new
messages are stored. Memory that was stored earlier is kept, not used, and comes back
when you switch memory on again. To remove it, use *Reset this character's memory…*.

## Privacy

The memory never leaves your computer. `user_data\` is excluded from the repository,
and chat text, characters and facts are never written to the log files.
