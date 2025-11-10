from torchlite.autograd.tensor import Tensor


class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self.training = True

    def __setattr__(self, name, value):
        # This method is called whenever you do:
        # self.weight = Tensor(...)
        # self.layer1 = Linear(...)
        # self.my_regular_var = 10

        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

        # this is to avoid infinite loop
        # calls original __setattr__ from object class
        # this sets the attribute normally without calling the override again
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        # makes module callabe (e.g., model(x))
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # subclasses must override this function
        raise NotImplementedError("Subclasses must implement the forward() method.")

    def parameters(self):
        parameters_list = list(self._parameters.values())
        for module in self._modules.values():
            parameters_list.extend(module.parameters())
        return parameters_list

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def summary(self):
        raise NotImplementedError()