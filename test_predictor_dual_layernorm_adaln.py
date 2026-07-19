import os
import torch
from module import ARPredictor, load_predictor_kernel

ARTIFACT = "/home/mehran/.local/share/lewm-kernels/predictor-dual-layernorm-adaln/v1/build/torch212-cxx11-cu133-x86_64-linux"

def make(mode):
    return ARPredictor(num_frames=3, depth=6, heads=16, mlp_dim=2048,
        input_dim=192, hidden_dim=192, output_dim=192, dim_head=64,
        dropout=0, emb_dropout=0, dual_layernorm_adaln_implementation=mode).cuda().to(torch.bfloat16)

def test_six_block_predictor_forward_backward_and_compile():
    os.environ["LOCAL_KERNELS"] = f"mlengineer-ai/predictor-dual-layernorm-adaln={ARTIFACT}"
    load_predictor_kernel()
    eager, fused = make("eager"), make("fused"); fused.load_state_dict(eager.state_dict())
    x=torch.randn(4,3,192,device="cuda",dtype=torch.bfloat16,requires_grad=True); c=torch.randn_like(x,requires_grad=True)
    xe=x.detach().clone().requires_grad_(); ce=c.detach().clone().requires_grad_()
    ye=eager(xe,ce); yf=fused(x,c); torch.testing.assert_close(yf,ye,atol=.125,rtol=.02)
    ge=torch.autograd.grad(ye.square().mean(),(xe,ce)); gf=torch.autograd.grad(yf.square().mean(),(x,c))
    for a,b in zip(gf,ge): torch.testing.assert_close(a,b,atol=.08,rtol=.03)
    compiled=torch.compile(fused,fullgraph=True); torch.testing.assert_close(compiled(x,c),yf,atol=.125,rtol=.02)
