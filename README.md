# Base-quantization
## accomplished
- 2022.11.2
  - Base QAT, PTQ
  - Per_tensor, Per_channel
  - Minmax, EmaMinmax, Histogram, omse, adaround, bias_correction
  - dorefa, lsq 

## Requirements
```
Python > 3.6 + Pytorch >= 1.6
```
## Usage
### base quazation
#### such as QAT per_layer
```
python main.py --type QAT --level L
```
 You can change --type and --level to choose different quazation method
#### adaround
```
python main.py --type PTQ --adaround --level L
```
#### bias_correction
```
python main.py --type PTQ --bias_correction --level L
```
#### Histogram
```
python main.py --type PTQ --Histogram --level L
```
#### omse
```
python main.py --type PTQ --omse --level L
```
#### dorefa
```
python main.py --type QAT --dorefa --level L
```
#### lsq 
```
python main.py --type QAT --lsq --level L
```

## Example
| Models | VGG_S| 
| ------ | ------|
| QAT_8| 99.4  |
|dorefa_4_32|99.46|
|dorefa_6_32|99.56|
|LSQ_8|99.48|
|PTQ_bias_correction|99.43|
|PTQ_adaround|99.41|

## Note
- LSQ need pretrained model to inital scale
- Because MNIST is small, different quanzation methods all can compress the model without accuracy drop
- The code structure is simple and easy to expand other sota quanzation methods



