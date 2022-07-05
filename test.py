class Foo:
    def __init__(self, x):
        self.x = x
        
        self._y = 10


foo = Foo(5)
print(foo._y)