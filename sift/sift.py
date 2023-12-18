import torch
import torch.nn as nn
import numpy as np

class SIFT():
    def __init__(self, model, sparse_rate, sparse_module, exception=[], grad_acc=1, gradient_checkpointing=False) -> None:
        self.model = model
        self.total_num = 0
        self.gradient_checkpointing = gradient_checkpointing
        
        self.sparse_rate = sparse_rate
        
        ## Parameters need to be trained sparsely
        self.sparse_module = sparse_module
        ## Parameters need to be trained normally 
        self.exception = exception
        
        ## Mapping: Parameter --> Sparse Parameter
        self.sparse_mapping = dict()
        ## For convenience, we record the gradient accumulation step for each parameter.
        self.grad_acc_count = dict()
        self.grad_acc = grad_acc
        
        self.if_get_idx = dict()
        
        ## Record all the trainable parameters(the initial parameter that need be updated sparsely).
        self.named_trainable_parameters_list = list()
        ## Record all the parameters that need be cacualated in optimizer.
        self.named_parameters_in_optimizer_list = list()
        
        self.register_sparse_param()
    
    def register_sparse_param(self):
        """Register a sparse param for each param that need be updated sparsely
            and get the sparse grad by using backward hook
        """
        for n, p in self.model.named_parameters():
            ## select the parameters needed to be trained sparsely
            self.total_num +=p.numel()
            if any([m in n for m in self.sparse_module]):
                
                p.requires_grad = True
                ## set the number of trainable components of the parameter according to the sparse rate
                train_num = min(int(self.sparse_rate * p.numel()) + 1, p.numel())
                sparse_param =  nn.Parameter(p.new_zeros(train_num), requires_grad=True)
                sparse_param.grad = sparse_param.new_zeros(train_num)
                sparse_param.train_num = train_num
                ## pick the components that have top-k maximun absolute values 
                sparse_idx = torch.flatten(abs(p.data)).topk(train_num).indices.cpu().numpy()
                ## Random pick
                # sparse_idx = np.array(random.sample(list(range(p.numel())), p.train_num), dtype=int) 
                sparse_param.idx = np.stack(np.unravel_index(sparse_idx, p.shape))
                ## help the initial parameter to find the sparse parameter 
                self.sparse_mapping[n]=sparse_param
                self.grad_acc_count[n]=0
                self.if_get_idx[n]=False
                
                # ## register a backward hook to get the 'sparse' grad
                p.register_hook(self.get_sparse_grad())
                
                ## register it in the model so the framework can recognized the sparse param as a 'normal' param 
                setattr(self.model, n.replace('.', '_')+'_sparse', sparse_param)
                ## (name, p)
                self.named_trainable_parameters_list.append((n, p))
                ## (named_sparse, sparse p)
                self.named_parameters_in_optimizer_list.append((n+'_sparse', sparse_param)) 
                
            elif self.exception and any([item in n for item in self.exception]):
                p.requires_grad = True
                self.named_trainable_parameters_list.append((n, p))
                self.named_parameters_in_optimizer_list.append((n, p))
            elif self.gradient_checkpointing and n==next(self.model.named_parameters())[0]:
                p.requires_grad = True
            else:
                p.requires_grad = False
            
            # ## gradient caculate after backward hook, we use following codes to ensure the first sparse module can get the sparse grad as we expect.
            # ## the first parameter in the model, the last parameter in backward propagation
            # m = list(self.model.modules())[1]
            # m.register_full_backward_hook(self.get_sparse_grad())
            
    
    ## keep consistent with model.named_parameters()
    def named_trainable_parameters(self):
        return iter(self.named_trainable_parameters_list)
    
    def trainable_parameters(self):
        return iter(p for _, p in self.named_trainable_parameters_list)
    
    def named_parameters_in_optimizer(self):
        return iter(self.named_parameters_in_optimizer_list)
    
    def parameters_in_optimizer(self):
        return iter(p for _, p in self.named_parameters_in_optimizer_list)
    
    def get_trainable_num(self):
        return sum(p.numel() for p in self.parameters_in_optimizer())
    
    def print_trainable_parameters(self):
        print(
            f"trainable params: {self.get_trainable_num():,d} || all params: {self.total_num:,d} || trainable%: {100 * self.get_trainable_num() / self.total_num}"
        )
    
    def set_trainer(self, trainer):
        self.trainer = trainer
        self.grad_acc = trainer.args.gradient_accumulation_steps
        
    def get_sparse_grad(self):
        """use closure function to access the param in the backward hook
        """
        def hook(x):
            with torch.no_grad():
                for n, p in self.named_trainable_parameters():
                    if n in self.sparse_mapping.keys():
                        if p.grad is not None :
                            # print(n)
                            sparse_param = self.sparse_mapping[n]
                            grad = p.grad.to(sparse_param)
                            ## clean the init grad
                            p.grad = None
                            # if self.trainer.state.epoch ==0.:
                            if not self.if_get_idx[n]:
                                self.if_get_idx[n]=True
                                
                                sparse_idx = torch.flatten(abs(grad)).topk(sparse_param.train_num).indices.cpu().numpy() 
                                sparse_param.idx = np.stack(np.unravel_index(sparse_idx, p.shape))

                                ## reset optimizer state
                                # for s in self.trainer.optimizer.state[p].values():
                                #     s.zero_()
                                if n==list(self.sparse_mapping.keys())[-1]:
                                    print('switch idx')
                                    # self.trainer.create_optimizer()
                                    # print(sparse_param.idx)
                                return

                            # ##if you are interested in grad proportion, uncomment following code 
                            # grad_norm = torch.norm(grad)
                            # sparse_grad_norm = torch.norm(grad[sparse_param.idx])
                            # print(f"{n} grad proportion: {sparse_grad_norm/grad_norm*100:.2f}")
                                
                            ## get the sparse grad
                            if sparse_param.grad != None:
                                sparse_param.grad += grad[sparse_param.idx]
                            else:
                                sparse_param.grad = grad[sparse_param.idx]
                                 
                            self.grad_acc_count[n] += 1
                            if self.grad_acc_count[n] == (self.grad_acc):
                                ## update the initial param sparsely
                                p.data = p.data + torch.sparse_coo_tensor(sparse_param.idx, sparse_param, p.shape).to(p)  
                                sparse_param.zero_()
                                self.grad_acc_count[n]=0
                                # print('sparse update!')
                            
                            
        return hook
                    
            