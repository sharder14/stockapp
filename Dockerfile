FROM tiangolo/uwsgi-nginx-flask:python3.8
ENV STATIC_URL /app/static
ENV STATIC_PATH /app/static
ENV SCRIPT_NAME=/stockapp
COPY requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt
