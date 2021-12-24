#!/bin/bash

set -e

# Generate Dockerfile
generate_docker() {
  docker run --rm kaczmarj/neurodocker:master generate docker \
           --base neurodebian:stretch-non-free \
           --pkg-manager apt \
           --install convert3d ants fsl gcc g++ graphviz tree \
                     git-annex-standalone vim emacs-nox nano less ncdu \
                     tig git-annex-remote-rclone octave netbase \
           --add-to-entrypoint "source /etc/fsl/fsl.sh" \
           --spm12 version=r7219 \
           --user=neuro \
           --workdir /home/neuro \
           --miniconda \
             conda_install="python=3.7 pytest jupyter jupyterlab jupyter_contrib_nbextensions
                            traits pandas matplotlib scikit-learn scikit-image seaborn nbformat nb_conda" \
             pip_install="https://github.com/nipy/nipype/tarball/master
                          https://github.com/INCF/pybids/tarball/0.7.1
                          nilearn datalad[full] nipy duecredit nbval niflow-nipype1-workflows comet_ml python-dotenv" \
             create_env="neuro" \
             activate=True \
           --env LD_LIBRARY_PATH="/opt/miniconda-latest/envs/neuro:$LD_LIBRARY_PATH" \
           --run-bash "source activate neuro && jupyter nbextension enable exercise2/main && jupyter nbextension enable spellchecker/main" \
           --user=root \
           --run 'mkdir /data && chmod 777 /data && chmod a+s /data' \
           --run 'mkdir /output && chmod 777 /output && chmod a+s /output' \
           --user=neuro \
           --run 'printf "[user]\n\tname = Maksim Kalinin\n\temail = maksim.kalinin.21@mail.ru\n" > ~/.gitconfig' \
           --run-bash 'source activate neuro' \
           --copy . "/home/neuro/functional_connectivity" \
           --user=root \
           --run 'chown -R neuro /home/neuro/functional_connectivity' \
           --run 'rm -rf /opt/conda/pkgs/*' \
           --user=neuro \
           --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
           --workdir /home/neuro/functional_connectivity \
           --cmd jupyter-notebook
}

generate_docker > Dockerfile
