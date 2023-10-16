import blpapi


def underscore_to_camelcase(text):
    """Converts underscore_delimited_text to camelCase"""
    return ''.join(word.title() if i else word.lower() for i, word in enumerate(text.split('_')))


class NameType(type):
    """Blpapi Name class wrapper"""

    def __getattribute__(cls, name):
        _name = underscore_to_camelcase(name)
        return blpapi.Name.findName(_name) or blpapi.Name(_name)


class Name(metaclass=NameType):
    pass
