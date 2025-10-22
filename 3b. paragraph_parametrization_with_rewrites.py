from lambeq.experimental.discocirc import DisCoCircReader


if __name__ == "__main__":
    paragraph = """Mr. and Mrs. Dursley, of number four Privet Drive, 
    were proud to say that they were perfectly normal, thank you very much. 
    They were the last people you'd expect to be involved in anything strange or mysterious,
    because they just didn't hold with such nonsense"""
    
    reader = DisCoCircReader()
    diagram = reader.text2circuit(
        paragraph, 
        rewrite_rules=[
            'sentence_modification', 
            'noun_modification', 
            'determiner'
        ]
    )
    diagram.draw()