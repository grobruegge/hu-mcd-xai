# Towards Human-Understandable Multi-Dimensional Concept Discovery (HU-MCD)

This is the PyTorch implementation of my master thesis "Towards Human-Understandable Multi-Dimensional Concept Discovery (HU-MCD)". HU-MCD is an extension of [Multi-dimensional concept discovery (MCD): A unifying framework with completeness guarantees](https://arxiv.org/pdf/2301.11911.pdf) and automatically extracts human-understandable concepts from pre-trained Convolutional Neural Networks (CNNs). Using these concepts, HU-MCD enables to explain the predictions of CNNs both locally and globally and includes a completeness relation that quantifies to what extend the concepts are sufficient in explaining the CNNs decision.   

The following illustration showcases an example explanations given an image for which the a ResNet50 model pretrained on ImageNet predicts the class `airliner`. Locally, each region within the image can be associated with a concept which can be attributed a local relevance score, highlighting its contribution to the final model prediction. Additionally, locally discovered concept can be associated with global concept candidates to further enhance interpretability. 

![Example illustration showcasing the local and global concept assignments for an image which the model classified as airliner.](images/local_example_airliner.png)

## Getting Started

### Install Requirements

#### General Requirements
Create a new python environent (here shown with `virtualenv` but you might also use for example `anaconda`)

```shell
# create new virtual environment
$ python3.9 -m venv <env_name>
# activate environment
$ source <env_name>/bin/activate
# install packages in requirements.txt
$ pip install -r requirements.txt
```

#### PyTorch

Note that you should **install PyTorch sperately** depending on whether you have a GPU available or not (see for https://pytorch.org/get-started/locally/). At the time of developing this framework the command would look as follows (for GPU installation)

```sh
# install PyTorch with GPU support
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Segment Anything
Additionally, you are required to **install the Segment Anything** libary using their [GitHub repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints). There, you can also download the [model checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) for the *ViT-H SAM Model* and place it within the main folder of the repository.

### Downloading images

Before running the code, you are required to download images for the classes which you wish to explain and store:
1. Create a folder called "*sourceDir*" within the main folder of this repository (you might also name the folder differently and pass this name as argument to the python script *-- source_dir [some name]*). 

2. For each class, create a folder within "*sourceDir*" named **exactly** like the classes. For each class at least 400 images should be available (you might also use a different number of class images using the argument *--n_cls_imgs [some number]*)

3. To represent concepts using prototypical images, additional create a folder named "*val_imgs*" within "*sourceDir*" that contains for each class a folder named "*[class name]_val*" containing validation images which where not used for training the explainer.

3. If you additionally want to run ACE for comparison, create an additional folder called "*random*" within "*sourceDir*" that contains random images of different classes (which are used as negative examples to train the Concept Activation Vectors)

For example, if you want to explain the class *airliner* for the ImageNet1k dataset, your folder *"sourceDir*" should look as follows:

```
hu-mcd
├───sourceDir
│   ├──airliner
│   │   ├─airliner_img_01.png
│   │   ├─airliner_img_02.png
│   │   └─...
│   │
│   ├──val_imgs
│   │   └─airliner_val
│   │       ├─airliner_val_img_01.png
│   │       ├─airliner_val_img_02.png
│   │       └─...
│   │
│   └──random
│       ├─rdm_img_01.png
│       ├─rdm_img_02.png
│       └─...
```

If you do not have any images at hand, this repository provides you with a helper script to download ImageNet1K images.

> **Information**: Note that the script uses multiprocessing and thus does not save exactly *--images_per_class* images but usually slightly more (because some processes still need to terminate). If this behavior for whatever reason is not acceptable, use *--number_of_processes 1*!

> **Warning**: Depending on the operating system, the script will not work as expected. I tested it in *Linux* and *Windows*, and for *Linux* everything worked fine, but for *Windows* the script does not stop executing. The problem is caused by the variable `class_img_counter` which is shared accross processes and keeps track of how many images have been downloaded for the class. In Unix-based systems like Linux, fork() is used to create child processes, which inherit the memory space of the parent process. This allows the shared variable class_img_counter to work as expected. However, Windows uses a different approach called spawn to create new processes. In this method, each new process starts with a fresh memory space, and thus, the shared variable doesn't behave as intended because each process gets its own copy of this variable. Long story short, use *Linux*... 

#### Downloading specific class images

To download images for one or several **specified classes** proceed as follows. The important thing is to pass the argument *--class_list* with the classes to download images for.

```shell
# run the script
$  python imagenet1k_downloader.py --save_dir sourceDir --class_list "<class_name_1>, <class_name_1>, ..." --images_per_class <args.num_target_class_imgs>
```

#### Downloading random images

To download random images, proceed as follows. Here, do not pass *--class_list* (only if you specify every class you want to have random images from) but instead pass the argument *--number_of_classes*. It then randomly selectes *--number_of_classes* classes with at least *--images_per_class* images. It is recommended to choose as many different classes as possible. The results are stored in `*--save_dir*/random`.

```sh
# run the script
$ python imagenet1k_downloader.py --save_dir sourceDir --number_of_classes <num_of_classes> --images_per_class <images_per_class> --number_of_processes 1
```

It is recommended to set *--number_of_processes* to 1 because otherwise you will end up with way to many images (see problem described above). Contrary to downloading images for one specific class, this will take rather long. Maybe I will fix this in a future implementation...

### Enable Layer Masking

To enable Input Masking for the ResNet architecture (loaded using the *timm* libary), replace `timm/models/resnet.py` with `input_masking/resnet.py`. Also, place `input_masking/sal_layers.py` in `timm/models`. To enable Input Masking for other model architectures refer to the [original implementation](https://github.com/SriramB-98/layer_masking/tree/main) of the [Input Masking procedure](https://arxiv.org/pdf/2211.14646.pdf).

## Running the code

### HU-MCD

The code for HU-MCD can be excuted by running the following command.

```sh
$ python run_humcd.py
```

This will globally explain the prediction of a ResNet50 model for 10 *ImageNet1k* classes using the activation after Global Average Pooling (GAP) of the final convolutional layer. The generated concepts will be saved within the folder `concept_examples/humcd/`. Different settings can be tested by modifying the arguments.

### MCD

For comparison, this repository also offers the possability to run [Multi-Dimensional Concept Discovery (MCD)](https://arxiv.org/pdf/2301.11911.pdf) framework based on their [publically available implementation](https://github.com/jvielhaben/MCD-XAI/tree/main).

```sh
$ python run_mcd.py
```

### MCD

For comparison, this repository also offers the possability to run [Multi-Dimensional Concept Discovery (MCD)](https://arxiv.org/pdf/2301.11911.pdf) framework based on their [publically available implementation](https://github.com/jvielhaben/MCD-XAI/tree/main).

```sh
$ python run_mcd.py
```




