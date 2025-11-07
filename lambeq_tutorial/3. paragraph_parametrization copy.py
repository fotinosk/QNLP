from lambeq.experimental.discocirc import DisCoCircReader


if __name__ == "__main__":
    paragraph = """The child is playing with balls. They are noisy"""
    
    reader = DisCoCircReader()
    diagram = reader.text2circuit(paragraph)
    diagram.draw()