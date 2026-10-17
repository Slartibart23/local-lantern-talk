# Characters

Every conversation in Local Lantern Talk is held with a **character**. A character
bundles who the AI is (personality and instructions), how it sounds (voice), which
language model it uses and what it remembers. Switching characters in the left panel
switches all of that in one click.

## Example characters

Four example characters are shipped in `config\default_characters\` and copied to
`user_data\characters\` on the first start:

| Character | Use | Scene | Recommended models (24–32 GB · 10–16 GB) |
| --- | --- | --- | --- |
| **Lena** | Everyday assistant: questions, writing, planning | Plain | Gemma 4 26B-A4B · Gemma 4 12B |
| **Max** | Small talk and good company | Plain | Mistral Small 3.2 24B · Mistral Nemo 12B |
| **Professor Edmund Hale** | In-depth conversations: explaining and debating | Study | Qwen 3.8 27B Abliterated · Qwen3 14B |
| **Elara** | Fantasy roleplay: keeper of the Gilded Lantern tavern | Tavern | Cydonia 24B · Rocinante 12B |
| **Captain Vargas** | Science-fiction roleplay: commander of the survey ship Meridian | Space | Magidonia 24B · MN-12B Mag-Mell R1 |
| **Vivienne** | Adult evening roleplay: owner of The Red Lantern bar (*for adults only*) | Red lantern | Qwen 3.6 27B Abliterated · Gemma 4 12B Heretic |

Each comes with a portrait (AI-generated, no real people), a **profile** and two
recommended models: one for GPUs with 24–32 GB and one for 10–16 GB. Click the character's
picture in the chat to see the profile card; *Show in catalog* opens the model in the
catalog. A recommendation never loads a model by itself.

They greet you in English. As soon as you speak or type German, Spanish or French, they
answer in your language, and their built-in voice switches to the same language.

Each example is copied once. New examples in an update appear at the next start, and an
example without a picture gets its portrait once. Changing or deleting them later never
brings them back on their own, and updates never overwrite your edited versions. To get
the current shipped version of an example character back: delete it in the app, close
the app, delete the file `user_data\characters\.defaults.json` (and `.defaults-copied`, if
it is there) and start the app again.
Missing example characters are then copied once more; your other characters are not
touched.

## Creating and editing characters

- **+ New character** (left panel) opens the editor with a **template** for the chosen type
  (Assistant or Roleplay): replace the parts in [brackets] with your own words. *Use
  template* fills it in again; switching the type of a new character switches the template.
- **Write from profile ✨**: write only the name and the **profile** in your own words (any
  language, e.g. "Hilde, 72, Bäckerin aus dem Schwarzwald, warmherzig, etwas stur"). The
  loaded language model then drafts the instructions, how the character speaks, how it
  behaves in a group, a short description and a greeting. You see the draft in the editor
  and change it before you save; nothing is saved by itself.
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
| **Short description** | One line shown as a tooltip in the list and on the profile card. |
| **Profile** | The profile card for you: age, background, personality. Shown when you click the character's picture; the model does not see it. |
| **Personality and instructions** | The most important field: who the character is. See below. |
| **How they speak** | Tone, word choice, sentence length, habits ("short, dry sentences; calls everyone *stranger*"). Added to the instructions, alone and in groups. |
| **In a group** | How the character behaves with other characters ("takes the lead", "teases whoever spoke last"). Only used in [group chats and story mode](group-chat.md). |
| **Greeting** | The first message of every new conversation. It is shown as text, not spoken. |
| **Scene** | The look of the chat: *Tavern*, *Space*, *Red lantern*, *Study* or *Plain* — background, accent colour, fairy-light colours and cliparts in the side panels. Every example character comes with a scene. Switch decorations off in **Settings → Appearance**. |
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

| How | Effect |
| --- | --- |
| **Send note ▶** (Enter) | The note alone: the character reacts at once, without you saying anything. Example: "A stranger in a grey cloak walks in", or an action like "Elara hands him a glass of rum". With an empty field the button says **Continue ▶** and the character simply goes on. |
| **Just speak or type** | A note that is still in the field goes with your next spoken or typed message, for that one reply. Example: write "Elara grows suspicious of the stranger", then say "I'm only passing through." |
| **Pin** | The note applies to every reply until you remove it (✕ on its card above the field). Pinned notes belong to the character and stay after *End chat* and restarts. Example: "It is a stormy night." Up to 8 per character. |

In a [group chat](group-chat.md#directors-notes-in-a-group) a list next to the field
chooses who the note is for: everyone, or one character.

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
- **✎ Edit** changes the text of any message that can still be deleted; the messages after
  it stay, and the model sees the new text from now on. A **character's reply** is spoken
  again with its voice and replaces the old audio in the session recording. **Your own
  message** keeps your recorded voice; only its text changes. For your last message,
  *Save and answer again* lets the character answer the new text (the old answer goes).
- **■ Stop** (next to *Pause listening*, or Esc) stops a reply that is being written or
  spoken. What was written so far stays.

## Files

Each character is one JSON file in `user_data\characters\` (for example `lena.json`);
its pictures are in `user_data\images\`.
You can back them up, copy them to another installation or edit them with a text
editor while the app is closed. `user_data\` is never part of the repository, so your
own characters stay private.
