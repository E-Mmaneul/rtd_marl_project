"""
延迟队列实现
"""
from collections import deque

class DelayQueue:
    def __init__(self, max_size=10):
        self.queue = deque(maxlen=max_size)

    def add(self, item):
        self.queue.append(item)

    def pop(self):
        if self.queue:
            return self.queue.popleft()
        return None
