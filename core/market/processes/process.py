from abc import ABC, abstractmethod


class ParameterProcess(ABC):
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    @abstractmethod
    def get(self):
        pass
