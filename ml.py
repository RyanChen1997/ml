# 写一个与门
import random


def AND(x1, x2):
    if x1 == 1 and x2 == 1:
        return 1
    else:
        return 0


# 写一个异或门
def XOR(x1, x2):
    if x1 == 1 and x2 == 0:
        return 1
    elif x1 == 0 and x2 == 1:
        return 1
    else:
        return 0


# 写一个或门
def OR(x1, x2):
    if x1 == 1 or x2 == 1:
        return 1
    else:
        return 0


# 帮我写一个飞机大战的游戏
def fight():
    while True:
        print("1. 我要打架")
        print("2. 我要打死你")
        print("3. 我要打赢你")

        choice = input("请输入你的选择：")
        if choice == "1":
            print("我要打架")
            if random.randint(0, 1) == 0:
                print("我赢了")
            else:
                print("我输了")
                break
            if random.randint(0, 1) == 0:
                print("你赢了")
            else:
                print("你输了")
                break
            if random.randint(0, 1) == 0:
                print("我赢了")
            else:
                print("我输了")
                break


if __name__ == "__main__":
    fight()
