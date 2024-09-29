# PMBNet
Progressive Multi-Branch Video Style Transfer Network via confidence reweighted Projection

Our site is live at https://hankunbo.github.io/PMBNet-Display-page/

You can download the pretrained model from [Here](https://drive.google.com/file/d/1PXcU0sKl4AWmRo4BACr8VCYmy0MEh2Zm/view?usp=drive_link)

```
python train.py --cuda --gpu 0 --style_content_loss --ssim_loss --recon_loss --recon_ssim_loss --tv_loss --temporal_loss --with_IRT --data_sigma --data_w --content_data "D:/coco2014/train2014" --style_data "D:/Wiki_style/train"
```
