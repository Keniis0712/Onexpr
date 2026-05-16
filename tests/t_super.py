class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f'{self.name} makes a sound'

    def kind(self):
        return 'animal'


class Dog(Animal):
    def speak(self):
        base = super().speak()
        return base + ' (a bark)'

    def kind(self):
        return super().kind() + '/dog'


class Puppy(Dog):
    def speak(self):
        return super().speak() + ' [puppy]'

    def kind(self):
        return super().kind() + '/puppy'


for cls in (Animal, Dog, Puppy):
    obj = cls('rex')
    print(obj.speak())
    print(obj.kind())
