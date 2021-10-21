FROM python:3.7.3-slim
RUN apt update && apt install -y wget
RUN python -m pip install --upgrade --force-reinstall pip
COPY app/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN rm requirements.txt
RUN rm -rf ~/.cache ~/.local
# COPY ./tools /tools
COPY ./app /app
WORKDIR /app
EXPOSE 8000
CMD ["bash", "-c", "exec gunicorn main:app --bind 0.0.0.0:8000 --workers ${GUNICORN_WORKERS}"]
