from __future__ import annotations

import uvicorn

from mapanything_service import build_argparser, create_app, load_service_config


def main() -> None:
    args = build_argparser().parse_args()
    config = load_service_config(args.config)
    if args.host is not None:
        config["service"]["host"] = args.host
    if args.port is not None:
        config["service"]["port"] = int(args.port)
    if args.model_dir is not None:
        config["model"]["model_dir"] = args.model_dir

    app = create_app(config)
    uvicorn.run(
        app,
        host=config["service"]["host"],
        port=int(config["service"]["port"]),
        log_level=str(config["service"]["log_level"]),
    )


if __name__ == "__main__":
    main()
