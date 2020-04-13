import torch as t


class Device:
    device = t.device('cpu')

    @staticmethod
    def set_device(key):
        Device.device = t.device(key)

    @staticmethod
    def get_device():
        return Device.device
