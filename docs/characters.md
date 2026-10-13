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
| **Lena** | Assistant: warm and upbeat | built-in German female |
| **Max** | Assistant: calm and precise | built-in German male |
| **Elara** | Roleplay: keeper of the Gilded Lantern tavern (fantasy) | built-in German female |
| **Captain Vargas** | Roleplay: captain of the survey ship Meridian (sci-fi) | built-in German male |

They are copied only once. Changing or deleting them later never brings them back on
their own, and updates never overwrite your edited versions.

## Creating and editing characters

- **+ New character** (left panel) opens the editor with an empty character.
- The **pencil** next to a character (or next to the name in the header) edits it.
- **Duplicate** creates a copy named "… (copy)" to experiment with.
- **Delete** removes the character file *and* everything the character remembers.
  The last remaining character cannot be deleted.

### Fields

| Field | Meaning |
| --- | --- |
| **Name** | Shown in the list and the header (up to 60 characters). |
| **Initials** | The avatar text. Left empty, it is made from the name ("Großmutter Hilde" → GH). |
| **Type** | *Assistant* or *Roleplay*. Roleplay characters additionally get rules to stay in character and never speak for you. |
| **Short description** | One line shown as a tooltip in the list. |
| **Personality and instructions** | The most important field: who the character is. See below. |
| **Greeting** | The first message of every new conversation. It is shown as text, not spoken. |
| **Voice** | A built-in voice or one of your own voices (see [voices.md](voices.md)). |
| **Language model** | *Use the selected model*, or a specific downloaded model that is loaded whenever you switch to this character. |
| **Voice output** | Off: this character always answers as text only. |
| **Show actions as text** | Whether `*actions*` are shown (dimmed, never spoken) or hidden. |
| **Remember conversations** | Memory on or off (see [memory.md](memory.md)). |
| **Record sessions** | Used by session recording. |

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

## Files

Each character is one JSON file in `user_data\characters\` (for example `lena.json`).
You can back them up, copy them to another installation or edit them with a text
editor while the app is closed. `user_data\` is never part of the repository, so your
own characters stay private.
