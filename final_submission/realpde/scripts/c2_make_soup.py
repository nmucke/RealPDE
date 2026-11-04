"""c2_soup: uniform weight soup of K fine-tuned checkpoints (same arch, same ctx_const) -> one checkpoint whose
payload is member 0's with the averaged state dict (so every downstream script -- prior / ke fits, packaging --
treats it as an ordinary fine-tuned backbone).  usage: c2_make_soup.py --out <soup.pth> <m0.pth> <m1.pth> ..."""
import argparse
import hashlib

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("members", nargs="+")
    a = ap.parse_args()
    ps = [torch.load(m, map_location="cpu", weights_only=False) for m in a.members]
    p0 = ps[0]
    for p in ps[1:]:
        assert p["model"] == p0["model"]
        assert p["ctx_const"] == p0["ctx_const"], "ctx_const differs"
        assert p["model_state_dict"].keys() == p0["model_state_dict"].keys()
    sd = {}
    for k, v in p0["model_state_dict"].items():
        if v.is_floating_point():
            sd[k] = (sum(p["model_state_dict"][k].double() for p in ps) / len(ps)).to(v.dtype)
        else:
            assert all(torch.equal(p["model_state_dict"][k], v) for p in ps), k
            sd[k] = v
    out = dict(p0)
    out["model_state_dict"] = sd
    out["history"] = []
    out["soup_members"] = [(m, hashlib.sha256(open(m, "rb").read()).hexdigest()) for m in a.members]
    out["note"] = f"c2_make_soup.py: uniform soup of {len(ps)} fine-tunes (payload = member 0's, averaged weights)"
    torch.save(out, a.out)
    print("soup ->", a.out, "members", len(ps))
    for m, h in out["soup_members"]:
        print(f"  member {h}  {m}")
    print(f"  soup   {hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}  {a.out}")


if __name__ == "__main__":
    main()
