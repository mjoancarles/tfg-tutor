# Use official Python base image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app code into the container
COPY . /app/

# Expose ports
EXPOSE 8501
EXPOSE 22

# Command to run the Streamlit app
CMD ["streamlit", "run", "Welcome.py", "--server.port=8501"]