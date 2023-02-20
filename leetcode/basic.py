# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2023/2/11 10:38 PM
==================================="""

class Person:
    name = []

person1 = Person()
person2 = Person()
person1.name.append("a")
person2.name.append("b")
print(person1.name)
print(person2.name)


# 装饰器
def my_shiny_new_decorator(a_function_to_decorate):
    def the_wrapper_around_the_original_function():
        print("Before the function runs")
        a_function_to_decorate()
        print("After the function runs")
    return the_wrapper_around_the_original_function

def a_stand_alone_function():
    print("I am a stand alone function, don't you dare modify me")


a_stand_alone_function()

a_stand_alone_function_decorated = my_shiny_new_decorator(a_stand_alone_function)
a_stand_alone_function_decorated()

@my_shiny_new_decorator
def another_stand_alone_function():
    print("Leave me alone")

another_stand_alone_function()

# 装饰器高级用法

def a_decorator_passing_arguments(function_to_decorate):
    def a_wrapper_accepting_arguments(arg1, arg2):
        print("I got args! Look:", arg1, arg2)
        function_to_decorate(arg1, arg2)
    return a_wrapper_accepting_arguments

@a_decorator_passing_arguments
def print_full_name(first_name, last_name):
    print("My name is", first_name, last_name)

print_full_name("Peter", "Venkman")