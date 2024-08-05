FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -e .

# Install development dependencies (Uncomment if needed)
# RUN pip install --no-cache-dir -e .[dev]

# Make sure scripts are executable
RUN chmod +x scripts/*.py

# Set environment variables for default paths
ENV DATA_PATH=/app/data/data.json
ENV MODEL_SAVE_PATH=/app/models_artifacts/
ENV MODEL_PATH=/app/models_artifacts/model_03_08_2024.joblib

# Create a shell script to run all commands
RUN echo '#!/bin/bash\n\
echo "Training the model..."\n\
python scripts/train_model.py --data_path $DATA_PATH --model_save_path $MODEL_SAVE_PATH\n\
\n\
echo "Evaluating the model..."\n\
python scripts/evaluate_model.py --data_path $DATA_PATH --model_path $MODEL_PATH\n\
\n\
echo "Running search example..."\n\
python scripts/search_example.py --data_path $DATA_PATH --model_path $MODEL_PATH\n\
\n\
exec "$@"' > /app/run_all.sh && chmod +x /app/run_all.sh

# Set the default command to run the shell script
CMD ["/app/run_all.sh"]


# From the root of the project, build the docker image and run it
# docker build -f docker/Dockerfile -t instaffo-matching-test .
# docker run instaffo-matching-test

# To run only the training script
# docker run instaffo-matching-test python scripts/train_model.py --data_path $DATA_PATH --model_save_path $MODEL_SAVE_PATH

# # To run only the evaluation script
# docker run instaffo-matching-test python scripts/evaluate_model.py --data_path $DATA_PATH --model_path $MODEL_PATH

# # To run only the search example script
# docker run instaffo-matching-test python scripts/search_example.py --data_path $DATA_PATH --model_path $MODEL_PATH