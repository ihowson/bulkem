# bulkem

R package to perform multicore and CUDA-accelerated fitting of finite mixture models.

## Use cases

* You want to fit mixture models composed of inverse Gaussian components
* You have a lot of datasets to fit (thousands) and it's taking too long
* The datasets are difficult to fit, so you want to try a lot of initialisation parameters, and this is taking a long time
* You have a lot of difficult-to-fit datasets and it's taking days to run

## Installation

Not ideal right now:

* Clone the repo
* In the `src/cuda` directory, run `make`
* In the top-level directory, run:

    $ R
    > devtools::load_all()


