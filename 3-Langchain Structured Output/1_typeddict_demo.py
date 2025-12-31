from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name':'nitish', 'age':'35'}
print(new_person)

from typing import TypedDict

class Animal(TypedDict):
    name:str
    weight:float

new_animal: Animal = {'name':'Tiger', 'weight' : '70.5'}
print(new_animal)
