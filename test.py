from exception.execptions import MLProjectException

def divide_numbers(a, b):
    try:
        return a / b
    except Exception as e:
        raise MLProjectException("Division failed", e)

if __name__ == "__main__":
    try:
        print(divide_numbers(10, 0))  # intentional error
    except MLProjectException as e:
        print(e)
