from inspect import signature, _empty


def to_lower(string: str):
    return string.lower().replace(' ', '_')


def to_capital(string: str):
    return " ".join(word.capitalize() for word in to_lower(string).split("_"))


def has_allowed_arguments(fn):
    has_preds = False
    has_target = False
    for key, value in signature(fn).parameters.items():
        if key == 'preds':
            has_preds = True
        elif key == 'target':
            has_target = True
        elif value.default == _empty:
            return False
        continue
    return has_target and has_preds
