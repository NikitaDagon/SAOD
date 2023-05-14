import timeit
import random

arr = sorted([random.randint(1, 10000) for i in range(1000)])
board = [[0 for _ in range(8)] for _ in range(8)]
arr2 = sorted(random.sample(range(10000),10000))


def binary_search(arr, x):  # Бинарный поиск сложность O(log n)
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def test_binary_search():
    binary_search(arr, 256)


class Node:  # Бинарное дерево
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, node):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert(value, node.left)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert(value, node.right)

    def search(self, value):
        if self.root is None:
            return False
        else:
            return self._search(value, self.root)

    def _search(self, value, node):
        if node is None:
            return False
        elif node.value == value:
            return True
        elif value < node.value:
            return self._search(value, node.left)
        else:
            return self._search(value, node.right)

    def delete(self, value):
        if self.root is None:
            return False
        else:
            self.root = self._delete(value, self.root)
            return True

    def _delete(self, value, node):
        if node is None:
            return node
        elif value < node.value:
            node.left = self._delete(value, node.left)
        elif value > node.value:
            node.right = self._delete(value, node.right)
        else:
            if node.left is None and node.right is None:
                node = None
            elif node.left is None:
                node = node.right
            elif node.right is None:
                node = node.left
            else:
                temp = self._find_min(node.right)
                node.value = temp.value
                node.right = self._delete(temp.value, node.right)
        return node

    def _find_min(self, node):
        while node.left is not None:
            node = node.left
        return node


def fibonacci_search(arr, x):
    n = len(arr)
    fib2 = 0
    fib1 = 1
    fib = fib1 + fib2
    while fib < n:
        fib2 = fib1
        fib1 = fib
        fib = fib1 + fib2
    offset = -1
    while fib > 1:
        i = min(offset + fib2, n - 1)
        if arr[i] < x:
            fib = fib1
            fib1 = fib2
            fib2 = fib - fib1
            offset = i
        elif arr[i] > x:
            fib = fib2
            fib1 = fib1 - fib2
            fib2 = fib - fib1
        else:
            return i
    if fib1 and arr[offset + 1] == x:
        return offset + 1
    return -1


def test_Fibonacci():
    fibonacci_search(arr, 15)


def interpolation_search(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high and arr[low] <= x <= arr[high]:
        pos = low + ((x - arr[low]) * (high - low)) // (arr[high] - arr[low])
        if arr[pos] == x:
            return pos
        elif arr[pos] < x:
            low = pos + 1
        else:
            high = pos - 1
    return -1


def test_interpolation():
    interpolation_search(arr, 15)


class HashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size
        self.const = 1

    def hash(self, key):
        return key % self.size

    def rehash(self, key):
        return (key + self.const) % self.size

    def insert(self, key, value):
        index = self.hash(key)
        while self.table[index] is not None and self.table[index][0] != key:
            index = self.rehash(index)
        self.table[index] = (key, value)

    def search(self, key):
        index = self.hash(key)
        while self.table[index] is not None and self.table[index][0] != key:
            index = self.rehash(index)
        if self.table[index] is None:
            return None
        else:
            return self.table[index][1]

    def delete(self, key):
        index = self.hash(key)
        while self.table[index] is not None and self.table[index][0] != key:
            index = self.rehash(index)
        if self.table[index] is None:
            return False
        else:
            self.table[index] = None
            return True


class ReHashTable:
    def __init__(self, size):
        self.size = size
        self.table = [None] * size
        self.secret_key = random.randint(1, 10000)

    def hash(self, key):
        return key % self.size

    def rehash(self, key, i):
        return (self.hash(key) + self.secret_key * i) % self.size

    def insert(self, key, value):
        i = 0
        index = self.rehash(key, i)
        while self.table[index] is not None and self.table[index][0] != key:
            i += 1
            index = self.rehash(key, i)
        self.table[index] = (key, value)

    def search(self, key):
        i = 0
        index = self.rehash(key, i)
        while self.table[index] is not None and self.table[index][0] != key:
            i += 1
            index = self.rehash(key, i)
        if self.table[index] is None:
            return None
        else:
            return self.table[index][1]

    def delete(self, key):
        i = 0
        index = self.rehash(key, i)
        while self.table[index] is not None and self.table[index][0] != key:
            i += 1
            index = self.rehash(key, i)
        if self.table[index] is None:
            return False
        else:
            self.table[index] = None
            return True


class ChainsHashTable:
    def __init__(self, size):
        self.size = size
        self.table = [[] for _ in range(size)]

    def hash(self, key):
        return key % self.size

    def insert(self, key, value):
        index = self.hash(key)
        for item in self.table[index]:
            if item[0] == key:
                item[1] = value
                return
        self.table[index].append([key, value])

    def search(self, key):
        index = self.hash(key)
        for item in self.table[index]:
            if item[0] == key:
                return item[1]
        return None

    def delete(self, key):
        index = self.hash(key)
        for i, item in enumerate(self.table[index]):
            if item[0] == key:
                del self.table[index][i]
                return True
        return False


def is_safe(board, row, col):
    # Проверяем, что нет ферзей на левой диагонали
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    # Проверяем, что нет ферзей на правой диагонали
    for i, j in zip(range(row, -1, -1), range(col, len(board))):
        if board[i][j] == 1:
            return False
    # Проверяем, что нет ферзей на вертикали и горизонтали
    for i in range(len(board)):
        if board[i][col] == 1:
            return False
        if board[row][i] == 1:
            return False
    return True


def print_board(board):
    for row in board:
        print(row)


def solve(board, row):
    # Если все ферзи размещены
    if row == len(board):
        return True
    # Перебираем все клетки в текущей строке
    for col in range(len(board)):
        # Если можем разместить ферзя в текущей клетке, делаем это
        if is_safe(board, row, col):
            board[row][col] = 1
            # Рекурсивно решаем задачу для следующей строки
            if solve(board, row + 1):
                return True
            # Если не получилось разместить ферзя в следующей строке,
            # откатываемся к предыдущему шагу и пробуем другой вариант
            board[row][col] = 0
    # Если не получилось разместить ферзя в текущей строке,
    # возвращаем False
    return False


def print_board(board):
    for row in board:
        print(row)


tree = BinaryTree()
for i in range(1000):
    tree.insert(random.randint(0, 10000))

table = HashTable(10000)
for i in range(1000):
    table.insert(random.randint(0, 10000), i)

retable = ReHashTable(1000)
for i in range(1000):
    retable.insert(random.randint(0, 10000), i)

chatable = ChainsHashTable(1000)
for i in range(1000):
    chatable.insert(random.randint(0, 10000), i)

retime_search = timeit.timeit(lambda: retable.search(random.randint(0, 10000)), number=1000)
time_search = timeit.timeit(lambda: tree.search(random.randint(0, 10000)), number=1000)
time_hash = timeit.timeit(lambda: table.search(random.randint(0, 10000)), number=1000)
chatable_search = timeit.timeit(lambda: table.search(random.randint(0, 10000)), number=1000)
print("Оценка времени работы Бинарного поиска:", timeit.timeit(test_binary_search, number=1000))
print("Оценка времени работы в Бинарном дереве:", time_search)
print("Оценка времени работы Фибоначчиева поиска:", timeit.timeit(test_Fibonacci, number=124))
print("Оценка времени работы Интерполяционного поиска:", timeit.timeit(test_interpolation, number=1000))
print("Оценка времени работы в Хеш-таблице:", time_search)
print("Оценка времени работы в Рехэшированием(Псевдослучайных):", retime_search)
print("Время выполнения поиска в Хеш-таблице(Цепочки):", chatable_search)
print(("Стандартный поиск на Python: ", timeit.timeit(lambda: arr2, number=1251) * 1000))
# Задача Ферзи
solve(board, 0)
print_board(board)
