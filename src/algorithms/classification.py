class Classification:
    def __init__(self, value: float, name: str):
        self.value = value
        self.name = name

    def __eq__(self, other):
        return other.value == self.value

    def __ne__(self, other):
        return other.value != self.value

    def __hash__(self):
        return self.value.__hash__()

    @staticmethod
    def get_true_false():
        return [Classification(0.0, "False"), Classification(1.0, "True")]
