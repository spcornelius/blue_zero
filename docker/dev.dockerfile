FROM mmdanziger/blue_zero

ARG GH_TOKEN
RUN bash entrypoint.sh
ADD trained_models.tar.gz .
RUN conda run -n blue_zero pip install pandas seaborn

ENTRYPOINT [  ]
CMD [ "jupyter-lab", "--no-browser", "--ip", "0.0.0.0" ]