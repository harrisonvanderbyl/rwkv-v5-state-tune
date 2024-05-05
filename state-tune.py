import gc
import random
import time
import torch
import json
import os
import sys
from tokenizer import world
import torch.nn as nn

from b2sdk.v2 import B2Api, InMemoryAccountInfo
from cuda.v5chunk import RUN_CUDA_RWKV5


def download_file(filee, output):
    D2_ACCESS_KEY_ID = os.getenv('D2_ACCESS_KEY_ID')
    D2_SECRET_ACCESS_KEY = os.getenv('D2_SECRET_ACCESS_KEY')

    info = InMemoryAccountInfo()

    b2_api = B2Api(info)

    b2_api.authorize_account("production", D2_ACCESS_KEY_ID, D2_SECRET_ACCESS_KEY)
    try:

        bucket_name, key_name = filee[5:].split('/', 1)  # Remove 's3://' prefix and split bucket and key

        bucket = b2_api.get_bucket_by_name(bucket_name)

        local_file_name = output

        downloaded_file = bucket.download_file_by_name(key_name)

        downloaded_file.save_to(local_file_name)
    except Exception as e:
        print(e)
        
class TimeShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = None
        
    def forward(self, x):
        B = x.shape[0]
        
        if not self.training:
            
            next = torch.cat((self.state, x[:,:-1]), dim=1)
            self.state = x[:,-1:].detach().clone()
            
        else:
            next = torch.cat((self.state.repeat(B,1,1), x[:,:-1]), dim=1)
            
        return next
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        
        self.state = state_dict.pop(prefix+"state") if prefix+"state" in state_dict else None
        
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def _save_to_state_dict(self, destination, prefix, keep_vars):
            
        destination[prefix+"state"] = self.state
        
        return super()._save_to_state_dict(destination, prefix, keep_vars)

class RWKV_ChannelMix(nn.Module):
    
    def __init__(self, layer_id, n_layer, n_embd, dim_ffn):
        super().__init__()

        self.time_mix_k = nn.Parameter(torch.zeros(1,1,n_embd))
        self.time_mix_r = nn.Parameter(torch.zeros(1,1,n_embd))

        self.key = torch.nn.Linear(n_embd, dim_ffn, bias=False)
        self.receptance = torch.nn.Linear(n_embd, n_embd, bias=False)
        self.value = torch.nn.Linear(dim_ffn, n_embd, bias=False)
        self.lastlayer = layer_id == n_layer - 1
        
        self.shift = TimeShift()
        
    def forward(self, x):
 
    
        xx = self.shift(x)
        

        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        kv = self.value( torch.relu( self.key(xk) ) ** 2 )
        return (torch.sigmoid(self.receptance(xr)) * kv)
    
class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att):
        super().__init__()
        
        self.dim_att = dim_att
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.layer_id = layer_id

        self.n_head = n_head
        self.head_size = head_size
        self.head_size_divisor = 8


        self.time_mix_k = nn.Parameter(torch.zeros(1,1,n_embd))
        self.time_mix_v = nn.Parameter(torch.zeros(1,1,n_embd))
        self.time_mix_r = nn.Parameter(torch.zeros(1,1,n_embd))
        self.time_mix_g = nn.Parameter(torch.zeros(1,1,n_embd))

          
        self.receptance = nn.Linear(n_embd, dim_att, bias=False)
        self.key = nn.Linear(n_embd, dim_att, bias=False)
        

        self.value = nn.Linear(n_embd, dim_att, bias=False)
        self.output = nn.Linear(dim_att, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, dim_att, bias=False)
        self.ln_x = nn.GroupNorm(n_head, dim_att, eps=64e-5)

        
        self.register_buffer("time_decay", torch.zeros(n_head, self.head_size))
        self.register_buffer("time_faaaa", torch.zeros(n_head, self.head_size))
        
        self.silu = nn.SiLU()
        self.shift = TimeShift()
        
        self.wkvstate = None
        
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        
        if prefix+"time_decay" in state_dict:
            state_dict[prefix+"time_decay"] = state_dict[prefix+"time_decay"].double().view(self.n_head,-1).float().contiguous()
            state_dict[prefix+"time_faaaa"] = state_dict[prefix+"time_faaaa"].view(self.n_head,self.head_size)
        
        self.wkvstate = state_dict.pop(prefix+"wkvstate") if prefix+"wkvstate" in state_dict else None
        
        a = super()._load_from_state_dict(state_dict, prefix, local_metadata, False, missing_keys, unexpected_keys, error_msgs)
        
        return a
    
    

    def forward(self, x):
        
        # Get the x sizing
        B, T, C = x.shape
        H = self.time_decay.shape[0]
        if self.training:
            last_state_wkv = self.wkvstate.repeat(B,1,1)
        else:
            last_state_wkv = self.wkvstate
        K = last_state_wkv.shape[-2]
        
        # Perform the tokenshift
        xx = self.shift(x)
            

        # Get the xk, xv, xr, xg, and rkvg
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        
        g = self.silu(self.gate(xg))
        
        
            
            
        
        if g.device.type == "cuda":
            
            # torch.zeros(B, T, H, V, dtype=torch.bfloat16, device=x.device)
            r:torch.Tensor = self.receptance(xr)
            k:torch.Tensor = self.key(xk)    
            v:torch.Tensor = self.value(xv) 
            
            # calculate time decay
            wfor = self.time_decay.exp().neg().exp().view(1,1,-1).pow(torch.arange(0,T,device=r.device).reshape(1,-1,1)).repeat(B,1,1)
            # wfor[:,0] = 0
            
            #calculate the effects state has on the current activation
            rwfor = r.mul(wfor).reshape(B,T,H,K).transpose(1,2).reshape(B*H,T,K)
            
            
            xr = torch.bmm(rwfor,last_state_wkv.view(-1,K,K)).view(B,H,T,K).transpose(1,2).reshape(B,T,H*K)
            
            if not self.training:
                with torch.no_grad():
                    wback = self.time_decay.exp().neg().exp().unsqueeze(1).pow(torch.arange(T-1,-1,-1,device=r.device).reshape(1,-1,1)).repeat(B,1,1)
                    
                    self.wkvstate = last_state_wkv.detach().clone().mul(self.time_decay.unsqueeze(1).exp().neg().exp().pow(T).repeat(B,1,1))
                    
                    #calculate the effects k,v have on the state
                    torch.baddbmm(self.wkvstate,k.transpose(0,1).reshape(T,B*H,K).transpose(0,1).mul(wback).transpose(1,2),v.transpose(0,1).reshape(T,B*H,K).transpose(0,1),out=self.wkvstate)
            
            xr += RUN_CUDA_RWKV5(B,T,C,H,r,k,v,self.time_decay,self.time_faaaa)

            
        else:    
                       
            # torch.zeros(B, T, H, V, dtype=torch.bfloat16, device=x.device)
            r:torch.Tensor = self.receptance(xr).transpose(0,1).reshape(T,B*H,K).transpose(0,1)
            k:torch.Tensor = self.key(xk).transpose(0,1).reshape(T,B*H,K).transpose(0,1)    
            v:torch.Tensor = self.value(xv).transpose(0,1).reshape(T,B*H,K).transpose(0,1) 
            
            # calculate time decay
            wfor = self.time_decay.exp().neg().exp().unsqueeze(1).pow(torch.arange(0,T,device=r.device).reshape(1,-1,1)).repeat(B,1,1)
            
            
            #calculate the effects state has on the current activation
            rwfor = r.mul(wfor)
            
            u:torch.Tensor = self.time_faaaa.view(H,1,K).repeat(B,1,1)
            
            xr = torch.bmm(rwfor,last_state_wkv.view(-1,K,K))
            
            xr += (k*(u*r)).sum(-1,True).mul(v).reshape(-1,T,K)
            
            xrr = xr
            
            wback = wfor[:,torch.arange(T-1,-1,-1)].transpose(1,2)        
            k = k.transpose(1,2).mul(wback)
            
            if not self.training:
                with torch.no_grad():
                    self.wkvstate = last_state_wkv.detach().clone().mul(self.time_decay.unsqueeze(1).exp().neg().exp().pow(T).repeat(B,1,1))
                    
                    #calculate the effects k,v have on the state
                    torch.baddbmm(self.wkvstate,k,v,out=self.wkvstate)
                
            
            # calculate the effects kvr have on the future activations
            r = r[:,1:]
            k = k[:,:,:-1]
            v = v[:,:-1]
            xrr = xr[:,1:]
            
            
                    
            # 64,T,T
            splits = 64
            for i in range(0,T, splits):
                for j in range(0,T, splits):
                    mkk = torch.bmm(r[:,i:i+splits],k[:,:,j:j+splits]).tril(i-j)
                
                    mrr = xrr[:,i:i+splits]
                    mrr[:] = torch.baddbmm(mrr,mkk,v[:,j:j+splits])
                    
        
            
                
        # Reshape and normalize the logits
        x_logits = self.ln_x(xr.bfloat16().transpose(0,1).reshape(T,B,H,1,K).transpose(0,1).reshape(-1,C)).view(B, T, C)#.transpose(0,1)
        x_logits = self.output(x_logits * g)

        # Return the logits and the state
        return x_logits



class Block(nn.Module):

    def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, dim_ffn):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = RWKV_TimeMix(layer_id, n_layer, n_embd, n_head, head_size, dim_att)
        self.ffn = RWKV_ChannelMix(layer_id, n_layer, n_embd, dim_ffn)
        
    
        # Setup droupout at block level

    def forward(self, x):

        att_out = self.att(
                self.ln1(x)
            )

        
        x = x + att_out
        
        ffn_out = self.ffn(
            self.ln2(x)
        )
        
        x = x + ffn_out
        
        return x

    
class RWKV(nn.Module):

    def __init__(self,
                 load_model: str,
                 ):

 
        # Setup the parent class
        super().__init__()
           
        try:
            self.batches = micro_bsz
        except:
            self.batches = 1
            micro_bsz = 1

        try:
            grad_cp
        except:
            grad_cp = 0

        try:
            ctx_len
        except:
            ctx_len = 1024

        try:
            modelpath = load_model

        except:
            modelpath = None
        
        if modelpath:
            file = torch.load(modelpath)
            keys = list(file.keys())
            # remove _orig_mod from keys for compatibility with torch.compile
            newObj = {}
            for key in keys:
                if "_orig_mod." in key:
                    newKey = key.replace("_orig_mod.", "")
                    newObj[newKey] = file[key]
                else:
                    newObj[key] = file[key]
            file = newObj
            keys = list(file.keys())

            # detect model details
            vocab_size, n_embd = file["emb.weight"].shape
            n_embd = n_embd
            vocab_size = vocab_size
            n_layer = 0
            for key in keys:
                if key.startswith("blocks."):
                    layer = int(key.split(".")[1])
                    if layer > n_layer:
                        n_layer = layer
            n_layer = n_layer + 1
            # try:
            dim_ffn = file[f"blocks.0.ffn.value.weight"].shape[1]
            # except:
            #     dim_ffn = 2 * n_embd
            # model layers are model.2.x.yyy: find highest x
            
            try:
                n_head = file[f"blocks.0.att.time_decay"].shape[0]
            except:
                n_head = 64
           
        else:
            file = None

        try:
            dim_ffn = dim_ffn
        except:
            dim_ffn = int(3.5 * n_embd)
            
        
        self.n_embd = n_embd
        
        self.n_layer = n_layer
        
        self.n_head = n_head
        
        self.head_size = n_embd // n_head
        
        self.dim_ffn = dim_ffn
        
        self.emb = nn.Embedding(vocab_size, n_embd)
        
        self.blocks = nn.Sequential(*[
        
            Block(i, n_layer, n_embd, n_head, self.head_size, n_embd, dim_ffn) for i in range(n_layer)
        ])
        
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
        file["ln_in.weight"] = file.pop("blocks.0.ln0.weight")
        file["ln_in.bias"] = file.pop("blocks.0.ln0.bias")
        
        self.ln_in = nn.LayerNorm(n_embd)
        self.ln_out = nn.LayerNorm(n_embd)
        
        self.load_state_dict(file)
        
        
            

    def forward(self, idx: torch.Tensor):
        x = self.emb(idx)
        x = self.ln_in(x)
        x = self.blocks(x)
        x = self.ln_out(x)
        x = self.head(x)
        return x



class v5tune( torch.nn.Module):
    def __init__(self, model_path):
        self.model_name = 'v5-simple'
        
        super(v5tune, self).__init__()
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
      
        self.model = RWKV(load_model=model_path)
        self.model = self.model.to(self.dtype)
        self.model = self.model.to(self.device)
        
        self.layers = self.model.n_layer
        self.hidden = self.model.n_embd
        self.head_size = self.model.head_size
        self.heads = self.model.n_head
        
        self.model.requires_grad_(False)
        
        
        

    def forward(self, idx ,  **kwargs):

        if isinstance(idx, list):
            idx = torch.tensor(idx, device=self.device, dtype=torch.int64)
        # if idx is int, make tensor
        if isinstance(idx, int):
            idx = torch.tensor([idx], device=self.device, dtype=torch.int64)
        
        # if idx is not 3 dim tensor, make it 3 dim
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)
            idx = idx.repeat(1, 1)
            
    
            
        output = self.model.forward(idx)
        
        return output
    
        
        
    def new_state(self, B=1, rand=False, offset=0, scale=1):
        
        func = torch.randn if rand else torch.zeros
        return {
                **{
                    f"blocks.{i}.att.shift.state":func(B, 1, self.hidden, dtype=self.dtype, device=self.device).mul(scale).add(offset).requires_grad_(True) for i in range(self.layers)
                },
                **{
                    f"blocks.{i}.ffn.shift.state": func(B, 1, self.hidden, dtype=self.dtype, device=self.device).mul(scale).add(offset).requires_grad_(True) for i in range(self.layers)
                },
                **{
                    f"blocks.{i}.att.wkvstate": func(B* self.heads, self.head_size, self.head_size, dtype=self.dtype, device=self.device).mul(scale).add(offset).requires_grad_(True) for i in range(self.layers)
                }
        }
        
    def load_state(self, state):
        self.model.load_state_dict(state, strict=False)
        return self
        

    def new_softembedding(self, B):
        return torch.zeros(B, 1,self.hidden, dtype=self.dtype, device=self.device).requires_grad_(True)
    
    def cpu(self):
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        return self
    
    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self

    
        
# Load from either environment variables or command line arguments
args ={
    
    **{
        key[2:] : eval(value) if value.replace(".","").isnumeric() else value.replace("\\n","\n")  for key, value in os.environ.items() if key.startswith("--")
    },
    **{
        key : eval(value) if value.replace(".","").isnumeric() else value.replace("\\n","\n") for key, value in [(keys.split(" ")[0]," ".join(keys.split(" ")[1:]).rstrip()) for keys in " ".join(sys.argv[1:]).split("--") if len(keys.split(" "))]
    },
}

print(args)


def train_model(
    learningrate = 0.01,
    batchsize = 4,
    exit_loss = 1.5,
    max_epochs = 10,
    dataset_walk = "shuffled",# "random", "sequential", "shuffled"
    model_url = None,
    data_url = None,
    model_location = "model.pth",
    data_path = "data.jsonl",
    save_filename = "state.pth",
    prompt_cutoff = -1, # set to -1 to disable
    completion_cutoff = -1, # set to -1 to disable
    max_time = 60*60, # one hour
    huggingface_dataset: str = "lonestar108/naughty-chat",
    prompt_formatter = "user: {input}\n\nassistant:",
    response_formatter = " {output}",
    **kwargs
):

    # download model
    

    # download model
    if not os.path.exists(model_location):
        if model_url is None:
            raise Exception("File does not exist, and Model URL is not provided")
        
        if not "b2://" in model_url:
            os.system(f"wget {model_url} -O {model_location}")
        else:
            download_file(model_url, model_location)

    # download data
    if not os.path.exists(data_path):
        if data_url is None and huggingface_dataset is None:
            raise Exception("File does not exist, and Data URL is not provided")
        
        if huggingface_dataset is not None:
            from datasets import load_dataset
            dataset = load_dataset(huggingface_dataset)
            tojsonl = dataset["train"]
            with open(data_path, "w") as f:
                for line in tojsonl:
                    f.write(f"{json.dumps(line)}\n")
        else:
            if not "b2://" in data_url:
                os.system(f"wget {data_url} -O {data_path}")
            else:
                download_file(data_url, data_path)

    # open training data
    openfile = open(data_path, "r")

    # load training data
    jsonl = [json.loads(x) for x in openfile.readlines()]

    # encode prompt data
    tasks = [
        world.encode(prompt_formatter.format_map(x)[-prompt_cutoff - 1:]) for x in jsonl
    ]

    # print formatted prompt
    print("Masked input:\n",world.decode(tasks[0]))
    
    # encode completion data
    completions = [
        world.encode(response_formatter.format_map(x)[:completion_cutoff]) for x in jsonl
    ]
    
     # print formatted prompt
    print("Completion:\n",world.decode(completions[0]))
    

    # get longest length for padding
    longest = max([len(x) + len(y) for x,y in zip(tasks,completions)])

    # pad data
    rightpadded = [x + y + [0]*(longest - len(x+y)) for x,y in zip(tasks,completions)]

    # create mask
    mask = torch.tensor([[False] * len(x) + [True] * len(y) + [False]*(longest - len(x+y)) for x,y in zip(tasks,completions)])

    # get mask for inputs
    maskinputs = mask[:,1:]

    # make tensor
    batch = torch.tensor(rightpadded)

    # load model
    model = v5tune(model_location).cuda()
    
    # initialize state
    loss = 4.0
    emptystate = model.new_state(1, True)
    model.load_state(emptystate)    
    # softembedding = model.new_softembedding(1)

    # generate some shuffles
    shuffled = [torch.randperm(batch.shape[0]) for i in range(max_epochs)]
    shuffled = torch.cat(shuffled, 0)
    lossfunc = torch.nn.CrossEntropyLoss()

    # start training, set exit conditions
    starttime = time.time()
    count = 0
    running_avg = 4.0
    
    optimizer = torch.optim.Adam(list(emptystate.values()), lr=learningrate)

    losses = []
    try:
        # train
        while loss > exit_loss and count < max_epochs*(batch.__len__()//batchsize) and time.time() - starttime < max_time:
            
            torch.cuda.empty_cache()
            gc.collect()
            
            # set dataset walk
            batcrange = torch.randperm(batch.shape[0])[0:batchsize]                         if dataset_walk == "random" else \
                        torch.arange(count*batchsize, (count+1)*batchsize) % batch.shape[0] if dataset_walk == "sequential" else \
                        shuffled[count*batchsize:(count+1)*batchsize]                       if dataset_walk == "shuffled" else \
                        torch.randperm(batch.shape[0])[0:batchsize]
            
            
            # get batchitems and mask
            batchsub = batch[batcrange]
            submask = maskinputs[batcrange]
            
            # remove extra padding
            for i in range(batchsub.shape[1]):
                if submask[:,-i].sum() != 0:
                    batchsub = batchsub[:,:-i].clone()
                    submask = submask[:,:-i].clone()
                    break
                
            # move to cuda
            batchsub = batchsub.cuda()
            
            # zero gradients
            optimizer.zero_grad()
            
            # forward pass
            logits = model.forward(batchsub)
            
            # calculate loss
            logitsize = logits.shape[-1]
            logits = logits[:,:-1].reshape(-1,logitsize)[submask.reshape(-1)]
            batchinp = batchsub[:,1:].reshape(-1)[submask.reshape(-1)]
            loss = lossfunc(logits, batchinp)
        
            if(loss.isnan()):
                print("Loss is NaN")
                loss = 4.0
                continue
            # backward pass
            loss.backward()
            
            optimizer.step()
            
            running_avg = running_avg * 0.95 + loss.cpu().item() * 0.05
            losses += [loss.cpu().item()]
            
            # print loss, 5 decimal places
            mmr = running_avg.__format__(".5f")
            print(mmr, end="", flush=True)
            

            count += 1
                
            print("\n", end="", flush=True)
    # on ctrl-c save state
    except KeyboardInterrupt:
        print("\nKeyboard interrupt, saving state")
    except Exception as e:
        print("\nError, saving state to backup.pth")
        torch.save(emptystate, "backup.pth")
        raise e
    finally:
        print("Saving state")
        # save state
        torch.save(emptystate, save_filename)
        
        # show loss curve
        import matplotlib.pyplot as plt
        plt.plot(losses, label='ln = '+str(learningrate))
        plt.legend()
        plt.show()
        
        

    
if args.get("prompt", False):
    promptin = args["prompt"]
    model = v5tune(args["model_location"]).to(args.get("device", "cuda")).train(False)
    model.load_state(model.new_state())
    state = torch.load(args["save_filename"])
    
    print("Base model:\n")
    prompt = world.encode(promptin)
    promptlength = len(prompt)
    for i in range(50):
        prompt = [model.forward(prompt)[0,-1].argmax().cpu().item()]
        try:
            toshow = world.decode(prompt)
            if toshow == "\n\n":
                break
            promptlength += 1
            print(toshow, end="", flush=True)
        except:
            continue
        
    model.load_state(state)
    
    print("\n\nTuned model:\n")
    prompt = world.encode(promptin)
    promptlength = len(prompt)
    for i in range(50):
        prompt = [model.forward(prompt)[0,-1].argmax().cpu().item()]
        try:
            toshow = world.decode(prompt)
            if toshow == "\n\n":
                break
            promptlength += 1
            print(toshow, end="", flush=True)
        except:
            continue
        
    
    print("\n\n")
else:
    train_model(**args)


