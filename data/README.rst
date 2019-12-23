Datasets
********

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

This folder contains data loaders for common datasets. Note, the code in this folder is a derivative of the dataloaders developed in `this <https://github.com/chrhenning/ann_implementations/tree/master/src/data>`__ repo. For examples on how to use these data loaders with Tensorflow checkout the `original code <https://github.com/chrhenning/ann_implementations>`__.

All dataloaders are derived from the abstract base class :class:`data.dataset.Dataset` to provide a common API to the user.

Preparation of datasets
=======================

**Datasets not mentioned in this section will be automatically downloaded and processed.**

Here you can find instructions about how to prepare some of the datasets for automatic processing.

Large-scale CelebFaces Attributes (CelebA) Dataset
--------------------------------------------------

`CelebA <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ is a dataset with over 200K celebrity images. It can be downloaded from `here <https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8>`__.

Google Drive will split the dataset into multiple zip-files. In the following, we explain, how you can extract these files on Linux. To decompress the sharded zip files, simply open a terminal, move to the downloaded zip-files and enter:

.. code-block:: console

    $ unzip '*.zip'

This will create a local folder named ``CelebA``.

Afterwards move into the ``Img`` subfolder:

.. code-block:: console

    $ cd ./CelebA/Img/

You can now decide, whether you want to use the JPG or PNG encoded images.

For the jpeg images, you have to enter:

.. code-block:: console

    $ unzip img_align_celeba.zip

This will create a folder ``img_align_celeba``, containing all images in jpeg format.

To save space on your local machine, you may delete the zip file via ``rm img_align_celeba.zip``.

The same images are also available in png format. To extract these, you have to move in the corresponding subdirectory via ``cd img_align_celeba_png.7z``. You can now extract the sharded 7z files by entering:

.. code-block:: console

    $ 7z e img_align_celeba_png.7z.001

Again, you may now delete the archives to save space via ``rm img_align_celeba_png.7z.0*``.

You can proceed similarly if you want to work with the original images located in the folder ``img_celeba.7z``.

FYI, there are scripts available (e.g., `here <https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py>`__), that can be used to download the dataset.

Imagenet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)
-------------------------------------------------------------------

The ILSVRC2012 dataset is a subset of the ImageNet dataset and contains over 1.2 Mio. training images depicting natural image scenes of 1,000 object classes. The dataset can be downloaded here `here <http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads>`__.

For our program to be able to use the dataset, it has to be prepared as described `here <https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset>`__.

In the following, we recapitulate the required steps (which are executed from the directory in which the dataset has been loaded to).

1. Download the training and validation images as well as the `development kit for task 1 & 2 <http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_devkit_t12.tar.gz>`__.

2. Extract the training data.

   .. code-block:: console
   
        mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
        tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
        find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
        cd ..

   **Note, this step deletes the the downloaded tar-file. If this behavior is not desired replace the command** ``rm -f ILSVRC2012_img_train.tar`` **with** ``mv ILSVRC2012_img_train.tar ..``.

3. Extract the validation data and move images to subfolders.
  
   .. code-block:: console
   
      mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
      wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
      cd ..

   This step ensures that the validation samples are grouped in the same folder structure as the training samples, i.e., validation images are stored under their corresponding WordNet ID (*WNID*).

4. Extract the meta data:
  
   .. code-block:: console
   
      mkdir meta && mv ILSVRC2012_devkit_t12.tar.gz meta/ && cd meta && tar -xvzf ILSVRC2012_devkit_t12.tar.gz --strip 1
      cd ..
