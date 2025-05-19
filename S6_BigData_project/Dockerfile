FROM python:3.12
WORKDIR /project

COPY project_app/df_flask.py .
COPY project_app/df_code.py .
COPY project_app/df_metrics.py .

RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install psutil
RUN pip install flask
RUN pip install python-dotenv
RUN pip install prometheus-client

CMD ["python", "df_flask.py"]

