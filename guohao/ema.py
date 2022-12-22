import math
import torch

class EMA:
    def __init__(self,
                 momentum=0.99,
                 interval=1,
                 momentum_fun=None,
                 verbose=False,
                 ):
        self.iter = 0

        assert 0 <= momentum <= 1
        self.momentum = momentum
        self.interval = interval
        self.momentum_fun = momentum_fun
        self.verbose = verbose
        return
    
    def pprint(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
        return

    def initialize_teacher_from_student(self, teacher, student):
        self.pprint('initialize_teacher_from_student')
        if isinstance(student, list) and isinstance(teacher, list):
            for s_module, t_module in zip(student, teacher):
                t_module.load_state_dict(s_module.state_dict())
                t_module.eval()
        else:
            teacher.load_state_dict(student.state_dict()) ### state dict is deep copy
            teacher.eval()
        return

    def get_momentum(self):
        return self.momentum_fun(self.iter) if self.momentum_fun else \
                        self.momentum

    @torch.no_grad()
    def after_train_iter(self, teacher, student):
        """Update ema parameter every self.interval iterations."""
        
        self.pprint('after_train_iter')
        if (self.iter + 1) % self.interval != 0:
            return
        
        momentum = self.get_momentum()
        
        if isinstance(student, list) and isinstance(teacher, list):
            for s_module, t_module in zip(student, teacher):
                for s_param, t_param in zip(s_module.parameters(), t_module.parameters()):
                    t_param.data = momentum*t_param.data + (1-momentum)*s_param.data
                t_module.eval()
        else:
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data = momentum*t_param.data + (1-momentum)*s_param.data
            teacher.eval()
        
        self.iter += 1
        return