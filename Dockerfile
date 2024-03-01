FROM pytorch/pytorch
ENV DEBIAN_FRONTEND=noninteractive

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN mkdir -p /work
RUN chown -R user:user /work # /yolo

# Allow password-less 'root' login with 'su'
RUN passwd -d root
RUN sed -i 's/nullok_secure/nullok/' /etc/pam.d/common-auth

USER user

RUN pip install -U transformers scipy scikit-image opencv-python-headless

WORKDIR /work

CMD /bin/bash
