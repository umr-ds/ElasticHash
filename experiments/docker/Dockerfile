FROM python:3.7.3-slim
RUN apt update && apt install -y wget libpq-dev python-dev build-essential gcc
RUN pip install --upgrade pip
RUN pip3 install \
    jupyter \
    jupyterlab \
    jupyter_contrib_nbextensions \
    ipywidgets \
    jupytext \
    matplotlib \
    seaborn \
    bitstring \
    sklearn \
    progressbar \
    psycopg2 \
    numpy

RUN rm -rf ~/.cache ~/.local

EXPOSE 8888

CMD ["jupyter", "lab", "--allow-root", "--ip",  "0.0.0.0"]