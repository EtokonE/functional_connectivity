# Functional Connectivity
Analysis the functionally integrated relationship between spatially separated brain regions.

## Clone the Functional Connectivity repository.

```bash
git clone https://github.com/EtokonE/functional_connectivity.git
cd functional_connectivity
```

## Docker
```bash
$ docker pull etokone/functional_connectivity:latest

$ docker run -it --rm -p 8888:8888 \
-v /path/to/functional_connectivity/:/home/neuro/functional_connectivity/ \
etokone/functional_connectivity jupyter notebook
```

## CometML
[Here](https://www.comet.ml/etokone/fmri-la5c-study/view/new/panels) you can find more information about experiments
