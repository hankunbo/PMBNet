# PMBNet Test Version
Progressive Multi-Branch Video Style Transfer Network via confidence reweighted Projection

Our site is live at https://hankunbo.github.io/PMBNet-Display-page/

## Create the environment
```
conda env create -f environment.yaml
conda activate PMBNet
```

## Testing
Put your trained model and put under file Model

fix the option page under options\test_options.py 

```
python test.py --cuda --use_Global
python test.py --cuda
```
