import random
X = [random.randint(1, 10) for i in range(8)]
S = X
print("x before", X)
print("S before change",S)


def change(X):
    X = [X[i]+1 for i in range(8)]
    print("X function",X)
change(X)
print("S after change",S)
hi = sum(S)
print(hi)

