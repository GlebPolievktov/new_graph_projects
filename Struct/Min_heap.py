class Min_Heap(object):
    def __init__(self):
        self.mass = []
    def is_heap_emty(self)->bool:
        return len(self.mass) == 0
    def high(self):
        if self.is_heap_emty():
            return None
        return self.mass[0]
    def swap(self,index1,index2):
        self.mass[index1],self.mass[index2] = self.mass[index2],self.mass[index1]
    def shift_up(self,index):
        # index куда мы хотим переместить элемент на нов место он находит родитель
        parent_index = (index - 1) // 2
        while parent_index > 0 and self.mass[index][0] < self.mass[parent_index][0]:
            self.swap(index1=index,index2=parent_index)
            self.shift_up(parent_index)
    def shift_down(self,index):
        left_child_index = 2 * index + 1
        right_child_index = 2 * index + 2
        small_index = index
        if left_child_index < len(self.mass) and self.mass[left_child_index][0] < self.mass[small_index][0]:
            small_index = left_child_index
        if right_child_index < len(self.mass) and self.mass[right_child_index][0] < self.mass[small_index][0]:
            small_index = right_child_index

        if small_index != index:
            self.swap(index, small_index)
            self.shift_down(small_index)

    def push(self,el,priority):
        self.mass.append((priority,el))
        self.shift_up(len(self.mass)-1)

    def pop(self):
        if self.is_heap_emty():
            raise IndexError("pop from empty heap")
        self.swap(0,len(self.mass)-1)
        item = self.mass.pop()
        self.shift_down(0)
        return item



