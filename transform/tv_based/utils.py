from functools import wraps


def transformation_forward_wrapper(func):
    @wraps(func)
    def wrapper(data):
        if not isinstance(data, (list, tuple)):
            data = [data]
        data = func(data)
        if len(data) == 1:
            data = data[0]
        return data

    return wrapper
