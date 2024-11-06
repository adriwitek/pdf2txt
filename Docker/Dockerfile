FROM ubuntu:22.04


WORKDIR /app



# Setup Script
## DELTE THIS IF FINALLY I DONT USE THE .SH
RUN apt-get update && apt-get install -y git git-lfs wget


#RUN git clone --depth 1 --branch main https://github.com/adriwitek/pdf2txt
RUN git clone --depth 1 --branch main https://github.com/nextprocurement/pdf2txt.git
WORKDIR /app/pdf2txt/Docker
RUN chmod +x setup_docker.sh && bash setup_docker.sh





ENV TESSDATA_PREFIX /usr/share/tesseract-ocr/4.00/tessdata/

ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH




# APP on container
WORKDIR /app/pdf2txt
#CMD python pipeline/pipeline.py
CMD /bin/bash