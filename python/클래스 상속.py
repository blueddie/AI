# class Animal:
#     def speak(self):
#         print("동물이 소리를 낸다.")

# class Cat(Animal):
#     pass

# my_cat = Cat()
# my_cat.speak()  # 출력: "동물이 소리를 낸다."
class Animal:
    def speak(self):
        print("동물이 소리를 낸다.")

class Cat(Animal):
    def speak(self):
        super().speak()
        print("야옹")

my_cat = Cat()
my_cat.speak()  # 출력 : 동물이 소리를 낸다. 야옹

# 클