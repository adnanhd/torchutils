import valids

class Foo:
    pass

class Bar(Foo):
    pass


valids._BaseModelType2.register_subclasses(Foo)