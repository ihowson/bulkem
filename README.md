# bulkem

R package to perform multicore and CUDA-accelerated fitting of finite mixture models.

## Use cases

* You want to fit mixture models composed of inverse Gaussian components
* You have a lot of datasets to fit (thousands) and it's taking too long
* The datasets are difficult to fit, so you want to try a lot of initialisation parameters, and this is taking a long time
* You have a lot of difficult-to-fit datasets and it's taking days to run
* You have large datasets (over 10000 observations) to fit

## Installation on Ubuntu 14.04 LTS (Trusty 64-bit) from a bare install

We need a newer version of R, so edit `/etc/apt/sources.list` and add the line

    deb http://cran.rstudio.com/bin/linux/ubuntu trusty/

Then at the command line,

    $ gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
    $ gpg -a --export E084DAB9 | sudo apt-key add -
    $ sudo apt-get update
    $ sudo apt-get install libssl-dev r-base r-cran-rcurl libxml2-dev libcurl4-openssl-dev nvidia-current git-core

Download the CUDA Toolkit:

    wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run

Install the CUDA Toolkit. `bulkem` assumes that you use the default installation location of `/usr/local/cuda-6.5`. We only need the toolkit; the graphics driver has already been installed.

    $ chmod +x cuda_6.5.14_linux_64.run
    $ sudo ./cuda_6.5.14_linux_64.run -toolkit -silent

Start R:

    $ R

Install the `devtools` package:

    > install.packages('devtools')

Finally, we can install `bulkem`. This is manual at the moment:

    $ git clone https://github.com/ihowson/bulkem
    $ cd bulkem/src/cuda
    $ make
    $ cd ../..
    $ LD_LIBRARY_PATH=/usr/local/cuda/lib64 R
    > devtools::build()
    > devtools::install()
