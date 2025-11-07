from lambeq.experimental.discocirc import DisCoCircReader
from lambeq import AtomicType, IQPAnsatz

if __name__ == "__main__":
    paragraph = """The child is playing with balls. They are noisy"""
    
    reader = DisCoCircReader()
    diagram = reader.text2circuit(paragraph, sandwich=True)
    ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1}, n_layers=1)
    circuit = ansatz(diagram)
    circuit.draw(figsize=(10,10))