import tiktoken

def show_subtokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    subtokens = [enc.decode_single_token_bytes(t).decode("utf-8", errors="replace") for t in tokens]
    return list(zip(tokens, subtokens))

if __name__ == "__main__":
    print("Evaluating processor", show_subtokens('optimizer'), end='\n\n')
    print("Evaluating processing", show_subtokens('optimizing'), end='\n\n')
