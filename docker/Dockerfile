# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -e .

# Install development dependencies if needed (uncomment if required)
# RUN pip install --no-cache-dir -e .[dev]

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV NAME Instaffo_Matching

# Run app.py when the container launches
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
