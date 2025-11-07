from lambeq import BobcatParser, AtomicType, IQPAnsatz, SpacyTokeniser


if __name__ == "__main__":
    
    sentence = "You are a wizzard Harry"
    
    tokeniser = SpacyTokeniser()
    tokens = tokeniser.tokenise_sentence(sentence)
    parser = BobcatParser(verbose='suppress')
    diagram = parser.sentence2diagram(tokens, tokenised=True)

    # Define atomic types
    N = AtomicType.NOUN
    S = AtomicType.SENTENCE

    # Convert string diagram to quantum circuit
    ansatz = IQPAnsatz({N: 1, S: 1}, n_layers=2)
    circuit = ansatz(diagram)
    circuit.draw(figsize=(15,10))