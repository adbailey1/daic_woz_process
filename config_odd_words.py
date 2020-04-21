checker = [
    'xxx', 'xxxx', 'zz', 'b', 'd', 'oop', 'ayn', 'ad', 't', '[laughter]',
    '<se>', 'analytical<anal>', 'c', 'oaxaca', 'p', 'ca', 'ci', 'comme', 'mani',
    'pedi', 'x', 'ba', '<calm>calms', '<s>still', 'y', 'z', 'de', 'g', 'ws',
    '<deep', 'breath>'
]

# 'xxx', 'xxxx',
# Ayn Rand is an American-Russian Writer and Philosopher
# Ad hominem - latin for fallacious argumentative strategy whereby genuine discussion of the topic at hand is avoided by instead attacking the character
# T. Colin Campbell American Biochemist
# Oaxaca City in Mexico
# p was the patient trying to say PTSD
# mani-pedi is slang for getting a manicure and pedicure at the same time
# lots of the other words are explained in context. Words with <> shall be
# removed due to interruption/difficulty transcribing the audio,


def find_odd_words(data):
    """
    Finds any odd words from the list variable 'checker' in the textual data.

    Input
        data: list - List of lists for every file in the dataset and every
              sentence in the file
    """
    weird_words = []
    for i, strings in enumerate(data):
        for p, v in enumerate(strings):
            if isinstance(v, str):
                if v in checker:
                    weird_words.append(f"{v}:{i, p}")
            else:
                inter_v = v.split()
                for k in inter_v:
                    if k in checker:
                        weird_words.append(f"{v}:{i, p}")

