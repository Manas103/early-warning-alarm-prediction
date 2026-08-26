# Containerized edge scorer. Builds the C++ scorer from source, then runs it
# with no network access at all, mirroring Honeywell's edge-AI framing of
# running "completely isolated, in a container" with "no internet
# connectivity". The final image has no Python, no PyTorch, no network
# client of any kind, only the compiled scorer binary and its weights.
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY cpp/ ./cpp/
RUN cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=build /src/build/edge_scorer /app/edge_scorer
COPY weights/weights.bin weights/norm.txt /app/weights/
# no EXPOSE: this binary never opens a socket. It reads a local run file and
# a local weights file and writes a local JSON file, nothing else.
ENTRYPOINT ["/app/edge_scorer"]
# example invocation (mount a run file and an output directory as volumes):
#   docker run --rm --network none \
#     -v "$(pwd)/data/eval_runs:/data:ro" -v "$(pwd)/out:/out" \
#     early-warning-edge-scorer \
#     /app/weights/weights.bin /app/weights/norm.txt /data/upset_000_seed900000.bin \
#     180 0.90 3 /out/pred.json docker_smoke_run
# --network none proves no cloud dependency: the container cannot reach the
# network even if the binary tried to.
