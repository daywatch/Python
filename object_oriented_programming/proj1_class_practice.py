import random

class Student:

    education = "u"
    def __init__(self,name,age=14):
        self.name = name
        self.age = age

    def greeting(self):
        self.num = random.randint(1, 3)
        if self.num == 1:
            return f"Hi I am {self.name}"
        if self.num == 2:
            return f"Hey there my name is {self.name}"
        else:
            return f"hi oh my name is {self.name}"


m1 = Student(name="bob",age=10)
m2 = Student(name="laura",age=40)
m3 = Student(name="monday",age=36)

# or use a loop for m1,2,3

print(m1.greeting())
print(m1.age)
print(m1.education)
print(m2.greeting())
print(m3.greeting())
