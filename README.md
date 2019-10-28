# preprocess of cwru dataset for domain adaptation task

# the differences from original repo:
1. This preproces supports two kinds of domain adaptation tasks, different loading conditions between drive end, different sensor location between drive end and fan end

2. the split between training set and test set is according to specific sample numbers, not according to percentage. 

The training dataset are sampled using augmentation.
The test dataset are not augmented.
The validation dataset is not in use.

3. When preprocessing the drive end data, use --defe=DE to fetch the matdata from 'DE_time'.
When preprocessing the fan end data, use --defe=FE to fetch the matdata from 'FE_time'.

# How to use
python preprocess-cwru.py --dataroot=data/deonly --trainnumber=660 --testnumber=25 --framesize=2048 --outputpath=output/deonly

python preprocess-cwru.py --dataroot=data/defe/de --defe=DE --trainnumber=660 --testnumber=25 --framesize=2048 --outputpath=output/defe
python preprocess-cwru.py --dataroot=data/defe/fe --defe=FE --trainnumber=660 --testnumber=25 --framesize=2048 --outputpath=output/defe
