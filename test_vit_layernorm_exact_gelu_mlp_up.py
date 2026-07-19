import os
import torch
from torch import nn
from module import configure_vit_layernorm_exact_gelu_mlp_up

ARTIFACT="/home/mehran/.local/share/lewm-kernels/vit-layernorm-exact-gelu-mlp-up/v1/build/torch212-cxx11-cu133-x86_64-linux"

class Attention(nn.Module):
    def forward(self,x,attention_mask=None,**kwargs): return x*0.25,None
class ViTMLP(nn.Module):
    def __init__(self): super().__init__();self.fc1=nn.Linear(192,768);self.fc2=nn.Linear(768,192);self.activation_fn=nn.GELU(approximate="none")
    def forward(self,x): return self.fc2(self.activation_fn(self.fc1(x)))
class ViTLayer(nn.Module):
    def __init__(self): super().__init__();self.attention=Attention();self.layernorm_before=nn.LayerNorm(192,eps=1e-12);self.layernorm_after=nn.LayerNorm(192,eps=1e-12);self.mlp=ViTMLP();self.dropout=nn.Dropout(0)
    def forward(self,x,attention_mask=None,**kwargs):
        r=x;x=self.attention(self.layernorm_before(x))[0]+r;r=x;return self.mlp(self.layernorm_after(x))+r

def test_modes_state_dict_compile_and_gradients():
    os.environ["LOCAL_KERNELS"]=f"mlengineer-ai/vit-layernorm-exact-gelu-mlp-up={ARTIFACT}"
    eager=ViTLayer().cuda().to(torch.bfloat16);fused=ViTLayer().cuda().to(torch.bfloat16);fused.load_state_dict(eager.state_dict())
    keys=set(fused.state_dict());configure_vit_layernorm_exact_gelu_mlp_up(fused,"fused");assert set(fused.state_dict())==keys
    x=torch.randn(4,257,192,device="cuda",dtype=torch.bfloat16,requires_grad=True);xe=x.detach().clone().requires_grad_(True)
    ye=eager(xe);yf=fused(x);torch.testing.assert_close(yf,ye,atol=.25,rtol=.02)
    ge=torch.autograd.grad(ye.square().mean(),(xe,*eager.parameters()));gf=torch.autograd.grad(yf.square().mean(),(x,*fused.parameters()))
    for a,b in zip(gf,ge):torch.testing.assert_close(a,b,atol=.25,rtol=.03)
    compiled=torch.compile(fused,fullgraph=True);torch.testing.assert_close(compiled(x.detach()),yf.detach(),atol=.25,rtol=.02)

def test_validate_records_all_layers():
    os.environ["LOCAL_KERNELS"]=f"mlengineer-ai/vit-layernorm-exact-gelu-mlp-up={ARTIFACT}"
    model=nn.Sequential(*[ViTLayer().cuda().to(torch.bfloat16) for _ in range(12)]);layers=configure_vit_layernorm_exact_gelu_mlp_up(model,"validate")
    model(torch.randn(2,257,192,device="cuda",dtype=torch.bfloat16));assert len(layers)==12 and all(len(x._lewm_mlp_up_records)==1 for x in layers)
