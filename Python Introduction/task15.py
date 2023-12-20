from typing import List


def hello(name: str = None) -> str:
    return "Hello" + (", " + name if name else '') + '!'

def int_to_roman(num: int) -> str:
    ONES = ['I', 'X', 'C', 'M']
    FIVES = ['V', 'L', 'D']

    res = ""
    pos = 0
    while num > 0:
        a = num % 10
        num //= 10
        if a in {1, 2, 3, 6, 7, 8}: res += a % 5 * ONES[pos]
        if a in {4, 5, 6, 7, 8}: res += FIVES[pos]
        if a in {4, 9}: res += int(a == 9) * ONES[pos + 1] + ONES[pos]
        pos += 1
    return res[::-1]


def longest_common_prefix(strs_input: List[str]) -> str:
    if not len(strs_input):
        return ""
    if (len(strs_input) == 1):
        return x[0]

    pref = ""
    m = [a.lstrip() for a in strs_input]

    for i in range(len(min(m, key=len))):
        if (all([a[i] == m[0][i] for a in m])):
            pref += m[0][i]
        else:
            break
    return pref


def primes() -> int:
    n = 2
    while True:
        for i in range(2, round(n ** 0.5) + 1):
            if not n % i:
                break
        else:
            yield n
        n += 1


class BankCard:
    def __init__(self, total_sum, balance_limit=-1):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __str__(self):
        return "To learn the balance call balance."

    def __add__(self, other):
        return BankCard(self.total_sum + other.total_sum,
                        self.balance_limit if self.balance_limit < 0 |
                                        self.balance_limit > other.balance_limit & other.balance_limit >= 0
                        else other.balance_limit)

    def __call__(self, sum_spent):
        if sum_spent <= self.total_sum:
            self.total_sum -= sum_spent
            print(f"You spent %d dollars." % sum_spent)
        else:
            print(f"Not enough money to spend %s dollars." % sum_spent)
            raise ValueError

    @property
    def balance(self):
        if self.balance_limit:
            self.balance_limit -= 1
            return self.total_sum
        else:
            print("Balance check limits exceeded.")
            raise ValueError

    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put %s dollars." % sum_put)
