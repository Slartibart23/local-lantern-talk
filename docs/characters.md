# Characters

Every conversation in Local Lantern Talk is held with a **character**. A character
bundles who the AI is (personality and instructions), how it sounds (voice), which
language model it uses and what it remembers. Switching characters in the left panel
switches all of that in one click.

## Example characters

Four example characters are shipped in `config\default_characters\` and copied to
`user_data\characters\` on the first start:

| Character | Type | Voice |
| --- | --- | --- |
| **Lena** | Assistant: warm and upbeat | built-in English female |
| **Max** | Assistant: calm and precise | built-in English male |
| **Elara** | Roleplay: keeper of the Gilded Lantern tavern (fantasy) | built-in English female |
| **Captain Vargas** | Roleplay: captain of the survey ship Meridian (sci-fi) | built-in English male |

They greet you in English. As soon as you speak or type German, Spanish or French, they
answer in your language, and their built-in voice switches to the same language.

They are copied only once. Changing or deleting them later never brings them back on
their own, and updates never overwrite your edited versions. To get the current shipped
version of an example character back: delete it in the app, close the app, delete the
file `user_data\characters\.defaults-copied` and start the app again. Missing example
characters are then copied once more; your other characters are not touched.

## Creating and editing characters

- **+ New character** (left panel) opens the editor with an empty character.
- The **pencil** next to a character (or next to the name in the header) edits it.
- **Duplicate** creates a copy named "… (copy)" to experiment with.
- **Export…** saves the character as one file (`<name>.llt-character.json` in your
  downloads folder) to keep or share: description, greeting, look, pictures and pinned
  director's notes. **Its memory and a cloned voice are never included.**
- **Import…** (left panel, next to *+ New character*) adds a character from such a file.
- **Delete** removes the character file *and* everything the character remembers.
  The last remaining character cannot be deleted.

### Fields

| Field | Meaning |
| --- | --- |
| **Name** | Shown in the list and the header (up to 60 characters). |
| **Initials** | Shown when there is no avatar picture. Left empty, they are made from the name ("Großmutter Hilde" → GH). |
| **Type** | *Assistant* or *Roleplay*. Roleplay characters additionally get rules to stay in character and never speak for you. |
| **Short description** | One line shown as a tooltip in the list. |
| **Personality and instructions** | The most important field: who the character is. See below. |
| **Greeting** | The first message of every new conversation. It is shown as text, not spoken. |
| **Scene** | The look of the chat: *Tavern*, *Space*, *Red lantern*, *Study* or *Plain* — background, accent colour, fairy-light colours and cliparts in the side panels. Elara and Captain Vargas come with a scene. Switch decorations off in **Settings → Appearance**. |
| **Avatar** | A picture of the character, shown in the list, in the header and as a portrait at the top left of the chat. Choose any JPG, PNG or WebP photo; you move and zoom it to fit the circle, and the app shrinks it to 768 × 768 pixels. The portrait size is set in **Settings → Appearance → Avatar size**: *Small* (header only), *Medium* or *Large*. Portraits grow with the window height (Medium about 245 px, Large about 390 px on a 2560 × 1440 screen), get smaller in narrow windows and are hidden on very small ones. |
| **Chat background** | A picture behind the chat, shown washed out so the text stays readable (*Washed out*, 40–95 %). It replaces the scene's background; the app shrinks it to at most 1920 pixels. |
| **Text colour** | One of 16 clearly different colours for the character's messages, all readable on the dark chat. Your own colour is set in **Settings → Your profile**. |
| **Voice** | A built-in voice or one of your own voices (see [voices.md](voices.md)). |
| **Language model** | *Use the selected model*, or a specific downloaded model that is loaded whenever you switch to this character. |
| **Voice output** | Off: this character always answers as text only. |
| **Show actions as text** | Whether `*actions*` are shown (dimmed, never spoken) or hidden. |
| **Remember conversations** | Memory on or off (see [memory.md](memory.md)). |
| **Record sessions** | Used by session recording. |
| **For adults only** | Hidden while family mode is on (**Settings → Family mode**). |

The voice and *Show actions as text* can also be changed in the right panel. The
change is saved to the active character.

### Writing a good personality

The text in *Personality and instructions* is sent to the language model at the start
of every request, so the character behaves the same in every conversation. Write it
in the second person ("You are …") and describe:

- **Who**: name, age, job, background, a few defining experiences.
- **How they talk**: warm or dry, formal or casual, typical phrases ("calls the user *Kind*").
- **What they know**, and what they honestly don't know (for example, today's news).
- **The setting**: where the conversation takes place and who the user is in it
  ("The user is a guest who just walked into your tavern").

English works best with most models, and the character still replies in the
language you speak. German instructions work too. You don't need to add rules about
speech: "no markdown", "no stage directions read aloud" and "reply in the user's
language" are added automatically.

Example:

> You are Hilde, a 72-year-old baker from a village in the Black Forest. You are warm,
> a little stubborn, call the user "Kind" and love telling stories about your bakery.
> The user is your grandson who visits you on Sunday afternoons.

**How long?** Up to 8000 characters. The whole text is sent with every message, so
nothing is ever cut or forgotten, but it shares the context with the conversation:
the longer the description, the less of the chat fits next to it. 1500–3000 characters
(about a page) work best; very long descriptions rarely make a character better,
because the model overlooks details.

## Your profile

**Settings → Your profile** holds your name, a few sentences about you, your own avatar
(shown at the top right of the chat) and the text colour of your messages. Your name
and the text about you are sent to every character, so they can address you personally;
your name also replaces "You" in session transcripts. Leave the fields empty if you
prefer.

## Director's notes

Under the message box is the **Director's note** field (🎬). What you write there is
not said to the character: it is a stage direction that the character follows without
mentioning or answering it.

| Button | Effect |
| --- | --- |
| **Next reply** (Enter) | The note goes with whatever you say or type next, for that one reply. Example: "Elara grows suspicious of the stranger." |
| **Pin** | The note applies to every reply until you remove it (✕ on its card above the field). Pinned notes belong to the character and stay after *End chat* and restarts. Example: "It is a stormy night." Up to 8 per character. |
| **Continue ▶** | The character goes on by itself, without you saying anything — with the note, if you typed one. Example: "A stranger in a grey cloak walks in." |

Notes appear dimmed in the chat (🎬 …) so you can see where you steered. They are never
spoken and never appear in the session transcript. Bigger models (24B and more) follow
them most reliably.

## Deleting, writing again and editing messages

Point at a message to see its buttons:

- **🗑 Delete** removes this message and everything after it — from the chat, the
  character's memory and the session recording. You can delete several messages at
  once this way, always from the end. Messages that are already folded into the memory
  summary can no longer be deleted.
- **↻ Write again** (last reply) lets the character answer your last message again.
- **✎ Edit** (your last message) puts it into the message box; send it to get a new
  answer.
- **■ Stop** (next to *Pause listening*, or Esc) stops a reply that is being written or
  spoken. What was written so far stays.

## Files

Each character is one JSON file in `user_data\characters\` (for example `lena.json`);
its pictures are in `user_data\images\`.
You can back them up, copy them to another installation or edit them with a text
editor while the app is closed. `user_data\` is never part of the repository, so your
own characters stay private.
