def max_triangle_perimeter(A):
    A.sort()
    max_perimeter = 0
    n = len(A)
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                if A[i] + A[j] > A[k]:
                    perimeter = A[i] + A[j] + A[k]
                    if perimeter > max_perimeter:
                        max_perimeter = perimeter
    return max_perimeter if max_perimeter > 0 else 0


print(max_triangle_perimeter([1, 1, 1]))


def largest_number(nums):
    nums = list(map(str, nums))
    nums.sort(key=lambda x: x * 10, reverse=True)
    return str(int(''.join(nums)))


print(largest_number([3, 30, 34, 5, 9]))


def diagonal_sort(mat):
    m, n = len(mat), len(mat[0])
    diagonals = {}
    for i in range(m):
        for j in range(n):
            if i - j not in diagonals:
                diagonals[i - j] = []
            diagonals[i - j].append(mat[i][j])
    for diagonal in diagonals.values():
        diagonal.sort()
    res = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            res[i][j] = diagonals[i - j].pop(0)
    return res


print(diagonal_sort([[3, 3, 1, 1], [2, 2, 1, 2], [1, 1, 1, 2]]))


def is_palindrome(s):
    return s == s[::-1]


def longest_palindrome(s):
    n = len(s)
    for i in range(n, 0, -1):
        for j in range(n - i + 1):
            if is_palindrome(s[j:j + i]):
                return s[j:j + i]
    return ""


print(longest_palindrome("babad"))


def can_strings_win(s1, s2):
    s1_sorted = sorted(s1)
    s2_sorted = sorted(s2)
    n = len(s1)
    for i in range(n):
        if s1_sorted[i] < s2_sorted[i]:
            break
    else:
        return True
    for i in range(n):
        if s2_sorted[i] < s1_sorted[i]:
            break
    else:
        return True
    return False


print(can_strings_win("abe", "acd"))


def is_concatenated_substring(s, start, end):
    n = end - start + 1
    if n % 2 == 0 and s[start:start + n // 2] == s[start + n // 2:end + 1]:
        return True
    return False


def count_concatenated_substrings(s):
    n = len(s)
    res = set()
    for i in range(n):
        for j in range(i + 1, n):
            if is_concatenated_substring(s, i, j):
                res.add(s[i:j + 1])
    return len(res)


print(count_concatenated_substrings("abcabcabc"))
