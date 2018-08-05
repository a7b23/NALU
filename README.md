# NALU

## This is an implementation of the paper "Neural Arithmetic Logic Units", Trask et al., ArXiV, 2018


To start training, run -
```
python NAC.py
```

To test the trained model run - 
```
python NAC_test.py
```
To run the baseline GRU, run - 
```
python GRU.py
```
To test the baseline model run - 
```
python GRU_test.py
```


The mean absolute error at different extrapolation ranges can be obtained by changing the sequence_length value.

The results(mean absolute error) have been given below :-


|  Seq_length|   GRU    | NAC      |
|------------|----------|----------|
|     10     |  2.79    |   1.60   |
|     100    |  371.18  |   6.65   |
|     1000   |  4370.00 |   24.96  |


