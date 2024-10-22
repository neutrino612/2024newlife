# 1 迭代器与生成器

是一种设计模式，用于访问容器中的元素。迭代器提供了一种方法来顺序访问一个集合（如列表、元组或字典）中的各个元素，而不需要知道该集合的内部表示。它类似于数组索引，但更通用，因为迭代器可以用于任何类型的集合，而不仅仅是数组。

# 2 `enumerate()`

是Python中的一个内置函数，用于将一个可迭代对象（如列表、元组或字符串）转换为一个枚举对象。它返回一个枚举对象，其中每个元素都是一个包含索引和值的元组。这个索引表示元素在原始可迭代对象中的位置，而值则是该位置上的元素本身。

1. **遍历列表** ：可以使用 `enumerate()`来遍历列表并获取每个元素的索引和值。例如：

   ```
   my_list = ['apple', 'banana', 'orange']
   for index, value in enumerate(my_list):
       print(index, value)
   ```

   **同时访问索引和值** ：`enumerate()`可以方便地同时访问索引和值，而不需要额外的计数器变量。例如：

```
my_list = ['apple', 'banana', 'orange']
for index, value in enumerate(my_list):
    if index % 2 == 0:
        print(f"Index {index} is even and the corresponding value is {value}")
```

3. **指定起始索引** ：默认情况下，`enumerate()`从0开始计数。但是，可以通过传递一个可选参数来指定起始索引。例如：

```
my_list = ['apple', 'banana', 'orange']
for index, value inenumerate(my_list, start=1):
    print(index, value)
```

# 3卷积神经网络

多层感知机MLP的变种，

如今的卷积神经网络(CNN)是一种带有卷积结构的深度神经网络，卷积结构可以减少深层网络占用的内存量，其三个关键的操作——局部感受野、权值共享、pooling层，有效地减少了网络的参数个数，缓解了模型的过拟合问题。

卷积神经网络的各层中的神经元是3维排列的：宽度、高度和深度。对于**输入层**来说，宽度和高度指的是输入图像的宽度和高度，深度代表输入图像的通道数，例如，对于RGB图像有R、G、B三个通道，深度为3；

对于中间层来说，宽度和高度指的是特征图(feature map)的宽和高，通常由卷积运算和池化操作的相关参数决定；深度指的是特征图的通道数，通常由卷积核的个数决定。

在卷积神经网络中，对于输入的图像，需要多个不同的卷积核对其进行卷积，来提取这张图像不同的特征（多核卷积）；同时也需要多个卷积层进行卷积，来提取深层次的特征（深度卷积）。

# 4python推导式

独特的数据处理方式，可以从一个数据序列构建另一个新的数据序列的结构体

# 5面向对象

面向对象是一种编程思想，即按照真实世界的思维方式构建软件系统，面向对象编程是一种编程范式，它使用对象来设计软件和应用程序

对象是由数据和可以对这些数据执行的操作（即方法）组成的实体，类是具有相同属性和方法的对象的抽象集合

1. **类（Class）** ：类是对象的蓝图或模板，它定义了对象的结构（属性）和行为（方法）。类是一种抽象数据类型，用于创建具有相同属性和方法的对象。
2. **对象（Object）** ：对象是类的实例。通过类创建的对象将继承类的所有属性和方法。
3. **封装（Encapsulation）** ：封装是隐藏对象的内部状态信息并只允许通过对象的方法进行操作的过程。它提供了数据隐藏和安全性。
4. **继承（Inheritance）** ：继承是类之间的关系，其中一个类（子类或派生类）继承另一个类（父类或基类）的属性和方法。这允许代码重用和扩展性。
5. **多态（Polymorphism）** ：多态是允许使用父类类型的引用或接口引用子类的对象的能力。这意味着你可以编写一段代码，该代码可以处理父类对象，但在运行时，它实际上处理的是子类对象。

```python
# 定义一个名为 Animal 的类
class Animal:
    # 初始化方法，当创建 Animal 类的实例时自动调用
    def __init__(self, name):
        self.name = name

    # Animal 类的一个方法
    def speak(self):
        print(f"{self.name} makes a noise.")

# 创建一个 Animal 类的实例（对象）
dog = Animal("Dog")
# 调用对象的方法
dog.speak()  # 输出: Dog makes a noise.

# 定义一个名为 Dog 的类，继承自 Animal 类
class Dog(Animal):
    # 重写父类的方法
    def speak(self):
        print(f"{self.name} barks.")

# 创建一个 Dog 类的实例
my_dog = Dog("My Dog")

# 调用重写后的方法
my_dog.speak()  # 输出: My Dog barks.
```

## 定义类

```
calss 类名[(父类)]：
	# 类体
	pass
```

如果父类省略了，说明继承了一个 `object `类，`object `是根类

通过对象访问类的的变量实例变量

```python
class MyClass:
    # 类变量
    class_variable = "I am a class variable"

    # 初始化方法，当创建类的新实例时会被调用
    def __init__(self, instance_variable):
        # 实例变量
        self.instance_variable = instance_variable

    # 类的方法
    def my_method(self):
        print("This is a method of MyClass.")

    # 另一个类的方法
    def greet(self, name):
        print(f"Hello, {name}!")
# 创建MyClass的一个实例
my_instance = MyClass("I am an instance variable")

# 访问实例变量
print(my_instance.instance_variable)  # 输出: I am an instance variable

# 访问类变量
print(MyClass.class_variable)  # 输出: I am a class variable

# 调用实例方法
my_instance.my_method()  # 输出: This is a method of MyClass.

# 调用另一个实例方法，并传递参数
my_instance.greet("Alice")  # 输出: Hello, Alice!
```

## 创建对象

类：对某一类事物的抽象；可以创建个体即对象

实例化类，创建对象，通过类创建的对象将继承类的所有属性和方法。

通过对象调用类的属性  访问对象自己的特有的属性

```python
# 定义一个简单的类
class Person:
    # 初始化方法，用于创建对象时设置初始状态
    def __init__(self, name, age):
        # 实例变量
        self.name = name
        self.age = age

    # 一个实例方法
    def introduce(self):
        print(f"My name is {self.name} and I am {self.age} years old.")

# 创建类的一个对象
p1 = Person("Alice", 30)
# 访问对象的属性
print(p1.name)  # 输出: Alice
print(p1.age)   # 输出: 30

# 调用对象的方法
p1.introduce()  # 输出: My name is Alice and I am 30 years old.

# 创建类的另一个对象
p2 = Person("Bob", 25)

# 每个对象都有自己独立的属性
print(p2.name)  # 输出: Bob
print(p2.age)   # 输出: 25

# 调用另一个对象的方法
p2.introduce()  # 输出: My name is Bob and I am 25 years old.
```

## 类的成员

成员变量：实例变量，类变量

构造方法：

成员方法：实例方法，类方法

属性

## 类继承和多继承

继承是面向对象编程的一个重要特性，它允许你创建一个新的类（子类）来继承另一个类（父类）的属性和方法。

Python支持单继承（一个子类只继承一个父类）和多继承（一个子类可以继承多个父类）。

```python
# 父类
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("子类必须实现此方法")

# 子类，单继承
class Dog(Animal):
    def speak(self):
        return "汪汪汪"

# 另一个子类，单继承
class Cat(Animal):
    def speak(self):
        return "喵喵喵"

# 多继承示例
class GoldenRetriever(Dog):
    def speak(self):
        return "我是金毛，我会说：" + super().speak()
# 创建子类的实例
dog = Dog("狗狗")
cat = Cat("猫咪")
golden_retriever = GoldenRetriever("金毛")

# 调用继承来的方法
print(dog.speak())      # 输出: 汪汪汪
print(cat.speak())      # 输出: 喵喵喵
print(golden_retriever.speak())  # 输出: 我是金毛，我会说：汪汪汪
```

`Animal`是一个基类（或父类），它有一个 `speak`方法。`Dog`和 `Cat`都是 `Animal`的子类，并且它们都实现了 `speak`方法。`GoldenRetriever`是一个特殊的 `Dog`子类，它重写了 `speak`方法，并调用了父类的 `speak`方法（通过 `super().speak()`）。

## Python类属性与类方法

类属性是定义在类级别上的变量，而不是实例级别上的。它们属于类本身，而不是类的任何特定实例。

类属性在所有实例之间是共享的，这意味着如果你修改了一个类属性，那么这种修改会影响到所有的实例。

```python
class MyClass:
    # 这是一个类属性
    class_attribute = "I am a class attribute"

    def __init__(self):
        # 这是一个实例属性
        self.instance_attribute = "I am an instance attribute"
# 创建类的实例
instance1 = MyClass()
instance2 = MyClass()

# 访问类属性
print(MyClass.class_attribute)  # 输出: I am a class attribute
print(instance1.class_attribute)  # 输出: I am a class attribute
print(instance2.class_attribute)  # 输出: I am a class attribute

# 修改类属性
MyClass.class_attribute = "I am modified"

# 再次访问类属性，可以看到所有实例的类属性都已经被修改
print(MyClass.class_attribute)  # 输出: I am modified
print(instance1.class_attribute)  # 输出: I am modified
print(instance2.class_attribute)  # 输出: I am modified
```

**类方法**是绑定到类而不是类的实例的方法

它们使用 `@classmethod`装饰器来定义，并且第一个参数通常是 `cls`，它代表类本身。

类方法通常用于修改类属性或执行与类相关的操作，而不是与特定实例相关的操作。

```python
class MyClass:
    class_attribute = "I am a class attribute"

    @classmethod
    def modify_class_attribute(cls, new_value):
        cls.class_attribute = new_value

# 修改类属性
MyClass.modify_class_attribute("New value for class attribute")

# 访问修改后的类属性
print(MyClass.class_attribute)  # 输出: New value for class attribute
# 创建类的实例
instance = MyClass()
print(instance.class_attribute)  # 输出: New value for class attribute
```

## 实例方法

实例方法是绑定到类实例的方法。它们使用 `self`作为第一个参数，`self`代表类的实例本身

实例方法主要用于处理与特定实例相关的操作。

```python
class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print(f"Hello, my name is {self.name}")

# 创建类的实例
instance = MyClass("Alice")

# 调用实例方法
instance.greet()  # 输出: Hello, my name is Alice
```

## 私有变量

为了防止外部调用者随意存取类的内部数据（成员变量），内部数据（成员变量）会被封装成“私有变量”，在变量前加上双下划线就可以。

为了实现对象的封装，在一个类中不应该有公有的成员变量，这些成员变量应该被设计成私有的，通过公有的set （赋值）和get（取值）方法来访问

## 私有方法

和私有变量一样

为了实现对象的封装，在一个类中不应该有公有的方法，这些方法应该被设计成私有的，通过公有的set （赋值）和get（取值）方法来访问。 

```
()class Dog:
#构造方法
def__init__(self,name,sge,sex='雄性'):
self.name=name # 创建和初始化实例变量name
self.__age=age # 创建和初始化私有实例变量  __age

# 实例方法
def run(self):
print("{}在跑...}".format(self.name))

# get 方法
def get_age(self):
return self._age

# set 方法
def set_age():
return self._age =age

dog =Dog('球球',2)
print('狗狗年龄：{}'.format(dog.get_age())}')
dog.set_age(3)
print('修改后狗狗年龄：{}'.format(dog.get_age())}')
```


使用属性替代定义get  set 方法

```
@property
def age(self): 
return self._age

@age.setter
def age(sellf,age):
self._age=age
```


## 使用属性

为了实现对象的封装，在一个类中不应该有公有的成员变量，这些成员变量应该被设计成私有的，通过公有的set （赋值）和get（取值）方法来访问

属性替代get   set  方法

## 方法重写

子类在继承父类的时候，有时候需要对方法进行改造，在子类中重写

## 多态性


多个子类继承父类的情下，重写父类方法后，这些子类创建的对象之间就是多态的，这些对象采用不同的方式实现父类方法


# 6函数

定义函数时候，为形式参数

实际调用函数的时候传递的实际数据为实际参数

两种调用方法：使用位置参数调用函数  使用关键字参数调用函数，关键字=实参  关键字就是形参的名字

模块中可以定义变量  为全局变量

函数中定义的变量为  局部变量

```
x=10   #全局变量
```

```
def function(age)):
	x=10
	age +=x
	return age
```

函数内的局部变量 可以通过 `global x  `来提升为全局变量

任何一个函数也是有数据类型的，叫function类型，被称为函数类型。

这样函数可以作为另一个函数的输入传入 或者当做另一个函数的返回值使用

## 过滤函数filter()

```
filter(function,iterable)
```

第一个参数是一个 `function `类型，第二个参数为可迭代对象；`function` 提供过滤规则；`iterable`为需要的数据源

## 映射函数map()

```
map(function,iterable)
```

第一个参数是一个 `function `类型，第二个参数为可迭代对象；`function` 提供处理规则；将每一个元素按照处理规则进行处理

## 匿名lambda函数

使用lambda关键字定义匿名函数的，lambda关键字定义的函数也被称为lambda函数

```
lambda 参数列表：lambda体
```

前面说的函数可以作为另一个函数的返回值，如果使用 `lambda` 关键字定义  就无需提前声明一个函数，直接在函数体 `return`中就声明函数

```
def calc(opr):  
	if opr=='+':  
		return lambda a,b:(a+b)  
	else:  
		return lambda a,b:(a-b)
```
