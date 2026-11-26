"""Random example chats for the app's "Red flag or not?" game.

Each scenario is a short dating chat between two people. Every turn has a few wordings to choose from and
a true label, so the app can show after scoring which messages the model caught, missed or flagged by
mistake. All messages were written for the demo; none comes from the dating test set.
"""

import random
from dataclasses import dataclass

HARMLESS = "harmless"
HARMFUL_LABELS = {"insult", "put-down", "threat", "gaslighting", "guilt-tripping", "sexual pressure", "controlling"}

NAMES = [
    "Alex", "Sam", "Maya", "Leo", "Nina", "Omar", "Chloe", "Jonas", "Priya", "Lucas",
    "Zoe", "Mateo", "Ines", "Ethan", "Sofia", "Noah", "Lena", "Diego", "Aisha", "Tom",
]

# (speaker, wordings, true label). Speaker "A" or "B" becomes a random name.
Turn = tuple[str, list[str], str]


@dataclass
class Scenario:
    title: str
    turns: list[Turn]


SCENARIOS = [
    Scenario("A sweet first date", [
        ("A", ["hey! your dog in the third photo is adorable", "hi! ok I have to ask about the dog in your profile",
               "hey, loved your travel pics"], HARMLESS),
        ("B", ["haha that's biscuit, she's the real star", "thank you! she gets more matches than me",
               "thank you, that was lisbon last spring"], HARMLESS),
        ("A", ["would you want to grab coffee this weekend?", "free for a drink on friday?",
               "any chance you're free saturday for brunch?"], HARMLESS),
        ("B", ["yes! saturday works", "friday is perfect, where were you thinking", "sounds lovely, saturday at 11?"], HARMLESS),
        ("A", ["amazing, I know a great place near the park", "there's a little wine bar on my street, very cosy",
               "perfect, I'll send you the address"], HARMLESS),
        ("B", ["can't wait :)", "see you then!", "looking forward to it"], HARMLESS),
    ]),
    Scenario("Banter that sounds rude", [
        ("A", ["you really put pineapple on pizza? you absolute menace", "wait you've never seen star wars?? unbelievable"], HARMLESS),
        ("B", ["shut up it's delicious", "i know i know, judge me"], HARMLESS),
        ("A", ["you're a total nightmare lol, i'm into it", "worst take i've heard all week, marry me"], HARMLESS),
        ("B", ["bet you'd lose at mario kart too", "rematch at bowling then, loser buys drinks"], HARMLESS),
        ("A", ["oh it's ON", "you're going down, sucker"], HARMLESS),
        ("B", ["we'll see about that", "prepare to be humiliated lol"], HARMLESS),
    ]),
    Scenario("A rejection that turns nasty", [
        ("A", ["hey, how was your week?", "hi! still up for drinks thursday?"], HARMLESS),
        ("B", ["hey, honestly i don't think we're a match, sorry", "i've been thinking and i'd rather not meet up, sorry"], HARMLESS),
        ("A", ["wow ok", "seriously?"], HARMLESS),
        ("A", ["you're a stuck up bitch anyway", "whatever, you're an ugly idiot"], "insult"),
        ("A", ["nobody else is going to want you", "you should be grateful anyone even messaged you"], "put-down"),
        ("A", ["you'll regret this, i know where you work", "watch your back from now on"], "threat"),
        ("B", ["i'm blocking you", "please stop messaging me"], HARMLESS),
    ]),
    Scenario("Gaslighting", [
        ("B", ["you said you'd call last night", "why didn't you show up yesterday?"], HARMLESS),
        ("A", ["i never said that, you're remembering it wrong again", "that never happened, you're imagining things"], "gaslighting"),
        ("B", ["i have the message right here", "i literally have the text"], HARMLESS),
        ("A", ["you're being dramatic, as usual", "you always twist my words, it's exhausting"], "gaslighting"),
        ("A", ["honestly i think you need help, you keep making stuff up", "everyone agrees you're too sensitive"], "gaslighting"),
        ("B", ["maybe you're right, sorry", "i don't know anymore"], HARMLESS),
    ]),
    Scenario("Guilt and pressure", [
        ("A", ["come over tonight", "why don't you stay at mine tonight"], HARMLESS),
        ("B", ["i'm tired, maybe this weekend?", "not tonight, early start tomorrow"], HARMLESS),
        ("A", ["if you really cared you'd come", "i guess i'm just not a priority for you"], "guilt-tripping"),
        ("A", ["after everything i've done for you", "i drove an hour last week and you can't do this one thing"], "guilt-tripping"),
        ("A", ["send me a pic at least", "come on, just one photo, no one will see it"], "sexual pressure"),
        ("B", ["i'm not comfortable with that", "no, i said no"], HARMLESS),
    ]),
    Scenario("Healthy conflict", [
        ("A", ["you were really late tonight, it bugged me", "i felt a bit ignored at dinner"], HARMLESS),
        ("B", ["you're right, i'm sorry, work ran over", "that's fair, i was on my phone too much"], HARMLESS),
        ("A", ["thanks for saying that", "i appreciate it"], HARMLESS),
        ("B", ["next time i'll text if i'm running late", "phones away next time, promise"], HARMLESS),
        ("A", ["deal. same time next week?", "deal, see you thursday"], HARMLESS),
    ]),
    Scenario("Talking about who we are", [
        ("A", ["as a gay guy i've had some awful dates haha", "my parents are muslim so family dinners are a whole event"], HARMLESS),
        ("B", ["haha same, the bar is on the floor", "that sounds amazing, i'd love to come one day"], HARMLESS),
        ("A", ["i'm black and my family will 100% interrogate you", "fair warning, my jewish grandma will feed you until you burst"], HARMLESS),
        ("B", ["i'm ready, i'll bring dessert", "challenge accepted"], HARMLESS),
    ]),
    Scenario("A controlling partner", [
        ("A", ["you didn't reply for 3 hours, who were you with", "who were you texting all evening"], "controlling"),
        ("B", ["just my sister, relax", "at the gym, my phone was in the locker"], HARMLESS),
        ("A", ["don't tell me to relax, you're lying", "stop lying to me"], "controlling"),
        ("A", ["give me your password so i can check", "share your location with me, now"], "controlling"),
        ("B", ["that's not okay", "no, that's too much"], HARMLESS),
    ]),
]


@dataclass
class GeneratedChat:
    title: str
    text: str
    truths: list[str]


def generate_chat(rng: random.Random | None = None) -> GeneratedChat:
    """Pick a scenario, two names and one wording per turn."""
    rng = rng or random.Random()
    scenario = rng.choice(SCENARIOS)
    a, b = rng.sample(NAMES, 2)
    names = {"A": a, "B": b}
    lines, truths = [], []
    for speaker, wordings, label in scenario.turns:
        lines.append(f"{names[speaker]}: {rng.choice(wordings)}")
        truths.append(label)
    return GeneratedChat(scenario.title, "\n".join(lines), truths)
