class Deque:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def append(self, item):
        self.items.append(item)

    def popleft(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            raise IndexError("pop")

    def __len__(self):
        return len(self.items)

