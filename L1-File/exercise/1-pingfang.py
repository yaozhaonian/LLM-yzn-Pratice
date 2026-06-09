# 判断正整数是不是完全平方数

def is_square(n):
    return n ** 0.5 % 1 == 0

def is_square_2(n):
    j = 1
    while n > 0:
        n -= j
        j += 2
        if n == 0:
            print("该正整数是完全平方数")
            return True
        elif n < 0:
            print("该正整数不是完全平方数")
            return False

print("第一种",is_square(16))
print("第二种",is_square_2(16))