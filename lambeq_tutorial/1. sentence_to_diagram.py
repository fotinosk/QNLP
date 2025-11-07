"""
Convert a sentence to a diagram
"""

from lambeq import SpacyTokeniser, BobcatParser


if __name__ == "__main__":
    sentence = """You're a wizzard Harry"""
    
    tokeniser = SpacyTokeniser()
    tokens = tokeniser.tokenise_sentence(sentence)
    parser = BobcatParser(verbose='suppress')
    print(tokens)
    diagram = parser.sentence2diagram(tokens, tokenised=True)

    diagram.draw(figsize=(23,4), fontsize=12)
    