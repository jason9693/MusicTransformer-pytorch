# Music Transformer: Generating Music with Long-Term Structure

- 2019 ICLR, Cheng-Zhi Anna Huang, Google Brain
- Re-producer : Yang-Kichang
- [paper link](https://arxiv.org/abs/1809.04281) 
- [paper review](https://github.com/SSUHan/PaparReviews/issues/13)



## Abstract

1. This Repository is perfectly cometible with **pytorch**



## Contribution

* Domain: Dramatically reduces the memory footprint, allowing it to scale to musical sequences on the order of minutes.
* Algorithm: Reduced space complexity of Transformer from O(N^2D) to O(ND).



## Preprocessing

* In this repository using single track method (2nd method in paper.).

* If you want to get implementation of method 1, see [here](https://github.com/COMP6248-Reproducability-Challenge/music-transformer-comp6248) .

* ~~I refered preprocess code from [performaceRNN re-built repository.](https://github.com/djosix/Performance-RNN-PyTorch).~~

* Preprocess implementation repository is [here](https://github.com/jason9693/midi-neural-processor).

  
  ![](https://user-images.githubusercontent.com/11185336/51083282-cddfc300-175a-11e9-9341-4a9042b17c19.png)



## Simple Start ( Repository Setting )

```bash
$ git clone https://github.com/jason9693/MusicTransformer-pytorch.git
$ cd MusicTransformer-pytorch
$ git clone https://github.com/jason9693/midi-neural-processor.git
$ mv midi-neural-processor midi_processor
```



## Midi Download	

```bash
$ sh dataset/script/{ecomp_piano_downloader, midi_world_downloader, ...}.sh
```

* These shell files are from [performaceRNN re-built repository](https://github.com/djosix/Performance-RNN-PyTorch) implemented by [djosix](https://github.com/djosix)



## Prepare Dataset	

```bash
$ python preprocess.py {midi_load_dir} {dataset_save_dir}
```



## Trainig

```bash
$ python train.py -c {config yml file 1} {config yml file 2} ... -m {model_dir}
```



## Hyper Parameter

* learning rate : 0.0001
* head size : 4
* number of layers : 6
* seqence length : 2048
* embedding dim : 256 (dh = 256 / 4 = 64)
* batch size : 2



## Result

-  Baseline Transformer ( Green, Gray Color ) vs Music Transformer ( Blue, Red Color )

* Loss

  ![loss](readme_src/loss.svg)

* Accuracy

  ![accuracy](readme_src/accuracy.svg)



## Generate Music

```bash
$ python generate.py -c {config yml file 1} {config yml file 2} -m {model_dir}
```




## Generated Samples ( Youtube Link )

* click the image.

  [<img src="readme_src/sample_meta.jpeg" width="400"/>](https://www.youtube.com/watch?v=n6pi7QJ6nvk&list=PLVopZAnUrGWrbIkLGB3bz5nitWThIueS2)
