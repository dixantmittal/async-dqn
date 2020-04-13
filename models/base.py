import torch as t
import torch.nn as nn


class BaseModel(nn.Module):
    def save(self, file):
        if file is None or file == '':
            return

        t.save(self.state_dict(), file)

    def load(self, file):
        if file is None or file == '':
            return

        self.load_state_dict(t.load(file, map_location='cpu'))
