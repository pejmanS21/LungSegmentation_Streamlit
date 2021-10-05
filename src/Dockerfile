FROM python:3
RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt
EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]
CMD [ "main.py" ]
