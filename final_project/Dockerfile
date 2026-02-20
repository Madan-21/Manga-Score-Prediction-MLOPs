FROM apache/airflow:2.3.2

# Switch to root user to install dependencies
USER root

# Install build-essential for packages that need to be compiled
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libmariadb-dev

# Switch back to the airflow user
USER airflow

# Copy the requirements file
COPY requirements.txt /

# Install the python dependencies
RUN pip install --no-cache-dir -r /requirements.txt