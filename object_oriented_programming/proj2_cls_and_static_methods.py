import random
import string

class Password:
    """
    This function generates password of customizing strength and length.
    There are three settings of strengths

    :param strength: mid, low, or high
    :param length: default is 10 characters
    """
    def __init__(self,strength="mid",length=10):
        """constructor method"""
        self.strength = strength
        self.length = length

    def _password_generator(self): #underscore means a method within the class!
        l = self.length or 10
        if self.strength == "low":
            return ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(l))
        elif self.strength == "mid":
            return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(l))
        elif self.strength == "high":
            return ''.join(random.choice(string.ascii_letters + string.digits + string.punctuation) for _ in range(l))

    @classmethod
    def show_input_universe(cls):

        character_pools = {
            "letters": list(string.ascii_letters),
            "numbers": list(string.digits),
            "punctuation": list(string.punctuation)
        }

        return character_pools

if __name__ == "__main__": #guards of execution
    m1 = Password("mid", 20)
    print(m1.__init__)
    print(m1._password_generator())
    print(m1.show_input_universe())

    m2 = Password("high", 200)
    print(m2._password_generator())

    m3 = Password()
    print(m3._password_generator())
