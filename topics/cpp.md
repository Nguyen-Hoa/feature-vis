# The C++ Programming Language by Bjorn Stroustrup

source code (.cpp) -> compiler (gcc, mingw, etc.) -> object file (.o) -> linker -> executeable (.exe, .out)

C++ is statically typed, that means the type of every entity (variables, objects, etc.) must be known at compile time.

* *Declaration* introduces a name into the program, along with its type.

* *Type* determines the set of possible values and operations a variable can have.

* *Object* is some memory that holds a value of some type.

* *Value* is set of bits interpreted according to its type

* *Variable* is a named object.

---

## assignment operator '=' vs initializer list '{}'

Initializer list throws an error when converting type loses information.

```c++
int a = 7.2     // no error, the value of a is 7
int a{7.2}      // error: float to int conversion
```

---

## auto

used in place of a type declaration when the type can be deduced by the initializer.

```c++
auto b = true;      // boolean
auto c = 'c';       // char
auto f = fib();     // float if fib() returns a float type
```

---

## Immutability

* *const* do not change value, must be initialized with value.
* *constexpr* constant value available at compile time, must be initialized with a constant.

Constants are sometimes referred to as *literal*.

---

## Exceptions

```c++
double Vector::insert(double value, int index) {
    // throw out of range exception
    if (index < 0 || this.size() < index) throw out_of_range()
}

try {

    // attempt to insert at invalid index
    myVector.insert(4.2f, -1);
}

catch (out_of_range){
    // handle out of range exception
}
```

* *out_of_range* exception type is part of the standard library. Read more about exception types
* *class invariants* are pre-conditions that the class constructor enforces (throws exception)
* *assertions* check values at compile time, and can be used on anything expressed with *constant expressions*

---
