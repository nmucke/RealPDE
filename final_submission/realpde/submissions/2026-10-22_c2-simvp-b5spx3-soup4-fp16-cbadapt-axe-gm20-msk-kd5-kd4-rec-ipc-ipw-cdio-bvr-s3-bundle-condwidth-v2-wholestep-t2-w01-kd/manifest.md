# 2026-10-22_c2-simvp-b5spx3-soup4-fp16-cbadapt-axe-gm20-msk-kd5-kd4-rec-ipc-ipw-cdio-bvr-s3-bundle-condwidth-v2-wholestep-t2-w01-kd

c2 twin of **FZ-EIKC** (main 2026-10-21_simvp-…-ipw-cdio-…, 3090 zip eef3cfe4) + **BVR** (experiments/c2_d3, c2_v27; review experiments/c2_review/fr_c2_bvr.md with R1 fix): the CDIO source + CONFIG "ipc_byvalue": True + the by-value output wrapper / _ByValueTensor (identical block to the 2026-10-21_c2-…-ipw-bvr twin). CDIO covers CUDA-input harnesses, BVR covers CPU-input harnesses that move our outputs with torch fd sharing; they never act on the same outputs.
Chain: experiments/c2_d3/0002-retrain_all-b5spx3-BVR-on-CDIO.patch (BVR=1 with CDIO=1 IPW=1 AXE=1 [KE3=1]). Status: NOT READY until main review + V.
