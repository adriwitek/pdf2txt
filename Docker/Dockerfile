FROM ubuntu:22.04


WORKDIR /app



# Setup Script
RUN git clone --depth 1 --branch main https://github.com/adriwitek/pdf2txt
WORKDIR /app/pdf2txt/Docker
RUN bash setup_docker.sh


ENV TESSDATA_PREFIX /usr/share/tesseract-ocr/4.00/tessdata/

ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH




# APP on container
WORKDIR /app/pdf2txt
#CMD python pipeline/pipeline.py
CMD /bin/bash