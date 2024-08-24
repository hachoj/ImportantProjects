from sympy import O
from Tokenizer import BasicTokenizer, RegexTokenizer

with open('taylorswift.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tok = BasicTokenizer()
retok = RegexTokenizer()

tok.train(text, 275)
# tok.get_vocab()

retok.train(text, 275)
# retok.get_vocab()

print(tok.decode(tok.encode("hello world!!!? (안녕하세요!) lol123 😉"))=="hello world!!!? (안녕하세요!) lol123 😉")
print(retok.decode(retok.encode("hello world!!!? (안녕하세요!) lol123 😉"))=="hello world!!!? (안녕하세요!) lol123 😉")
