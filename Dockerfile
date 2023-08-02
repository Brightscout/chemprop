# Set the base image to use, which is "mambaorg/micromamba:0.23.0"
FROM mambaorg/micromamba:0.23.0

# Create a named volume at "/opt/chemprop/data" in the container to persist data across container runs
VOLUME /opt/chemprop/data

# Switch to the root user to perform some installation tasks
USER root

# Update the package list and install "git" package, then remove unnecessary cached files to reduce image size
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/{apt,dpkg,cache,log}

# Switch back to the non-root user specified in the "MAMBA_USER" environment variable
USER $MAMBA_USER

# Copy the "environment.yml" file from the host to the container's "/tmp/" directory
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Install the dependencies specified in "environment.yml" using micromamba and create a new conda environment named "base"
# Then, clean up the micromamba cache and other unnecessary files
RUN micromamba install -y -n base -f /tmp/environment.yml && \
    micromamba clean --all --yes

# Copy the current directory from the host to the container's "/opt/chemprop" directory
COPY --chown=$MAMBA_USER:$MAMBA_USER . /opt/chemprop

# Set the working directory to "/opt/chemprop"
WORKDIR /opt/chemprop

# Install the Python package in editable mode within the conda environment "base" previously created
RUN /opt/conda/bin/python -m pip install -e .

# Expose port 5000 for the container to listen on
EXPOSE 5000

# Change the working directory to "/opt/chemprop/chemprop/web"
WORKDIR /opt/chemprop/chemprop/web

# Switch back to the root user to execute the final command
USER root

# The default command to run when the container starts.
# It uses Gunicorn to serve the application on 0.0.0.0:5000 and initializes the database with the root folder specified
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:build_app(init_db=True, root_folder='/opt/chemprop/data')"]
