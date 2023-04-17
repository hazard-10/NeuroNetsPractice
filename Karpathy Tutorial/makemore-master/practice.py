import torch

bigram_set = {}
words = open('names.txt', 'r').read().splitlines()
for w in words:
    characters = ['<S>']+list(w)+['<E>']
    for char1, char2 in zip(characters, characters[1:]):
        b_index = (char1, char2)
        bigram_set[b_index] = bigram_set.get(b_index, 0)+1
        
bigram_set = sorted(bigram_set.items(), key=lambda k:-k[1])
# for k in bigram_set:
    # print(k)
#     pass

N = torch.zeros((28, 28), dtype=torch.int32)
all_single_chars = sorted(list(set(''.join(words))))
char_to_index = {s:i for i,s in enumerate(all_single_chars)}
print(char_to_index)