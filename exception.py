class BaseException(Exception):
    pass

class ToolNotExistException(BaseException):
    pass


class ToolResponseFormatException(BaseException):
    pass


class ToolArgumentsValidationException(BaseException):
    pass