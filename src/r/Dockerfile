FROM rocker/r-ver:4.4.2

WORKDIR /app

COPY src/r/install_packages.R ./

RUN Rscript install_packages.R

COPY src/r/ ./

CMD ["Rscript", "main.R"]