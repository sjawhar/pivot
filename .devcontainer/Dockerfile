ARG NODE_VERSION=22.21.1
ARG PYTHON_VERSION=3.13.9
ARG UV_VERSION=0.9.18
ARG YARN_VERSION=1.22.22

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM node:${NODE_VERSION}-bookworm AS node

FROM python:${PYTHON_VERSION}-bookworm AS python
ARG UV_PROJECT_ENVIRONMENT=/opt/python
ENV PATH=${UV_PROJECT_ENVIRONMENT}/bin:$PATH

FROM python AS builder
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_INSTALLER_METADATA=1
ENV UV_LINK_MODE=copy

WORKDIR /source
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --all-extras \
        --all-groups \
        --locked \
        --no-install-project

FROM python
RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update \
 && apt-get install -y --no-install-recommends \
        bash-completion \
        curl \
        git \
        jq \
        less \
        locales \
        nano \
        rsync \
        unzip \
 && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen en_US.UTF-8

COPY --from=node /usr/local/bin /usr/local/bin
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=uv /uv /uvx /usr/local/bin/

ARG YARN_VERSION=1.22.22
RUN --mount=type=cache,target=/root/.npm \
    npm install --global --force yarn@${YARN_VERSION}

RUN echo 'eval "$(uv generate-shell-completion bash)"' >> /etc/bash_completion.d/uv

ARG APP_USER=pivot
ARG APP_UID=1000
ARG APP_GID=1000
ARG APP_DIR=/home/${APP_USER}/app

RUN groupadd -g ${APP_GID} ${APP_USER} \
    && useradd -m -u ${APP_UID} -g ${APP_USER} -s /bin/bash ${APP_USER} \
    && mkdir -p ${APP_DIR} \
    && chown -R ${APP_UID}:${APP_GID} ${APP_DIR}

COPY --from=builder --chown=${APP_UID}:${APP_GID} ${UV_PROJECT_ENVIRONMENT} ${UV_PROJECT_ENVIRONMENT}

WORKDIR ${APP_DIR}
COPY --chown=${APP_UID}:${APP_GID} . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --all-extras \
        --all-groups \
        --locked \
 && find ${UV_PROJECT_ENVIRONMENT} -user root -exec chown ${APP_UID}:${APP_GID} {} +

USER ${APP_USER}
