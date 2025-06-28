from typing import TypedDict


class Person(TypedDict):


    name :str
    age :int

new_person = {"name":"Bajarang","age": 21}
p1 = Person(new_person)
print(p1)
print(type(p1))