# PMBNet
Progressive Multi-Branch Video Style Transfer Network via confidence reweighted Projection

This is the Previous version

```
git clone -b previous https://github.com/hankunbo/PMBNet.git
```

pretrained model can be download from [Google Drive](https://drive.google.com/file/d/1U_gLZhqbsDcGjbJvGHE8NrCkjxSIL6uM/view?usp=drive_link)

You can train by lookinging run_train.sh or 
```
python3 train.py --cuda --gpu 0 --epoches 3 --batchSize 8 --lr 1e-4 --style_content_loss --ssim_loss --contentWeight 1 --recon_loss --recon_ssim_loss --tv_loss --temporal_loss --data_sigma --data_w 
```
