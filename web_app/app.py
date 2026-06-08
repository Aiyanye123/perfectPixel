from __future__ import annotations

import base64
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_BATCH_BYTES = 300 * 1024 * 1024
MAX_BATCH_FILES = 200
MAX_GRID_AXIS = 4096
MAX_OUTPUT_PIXELS = 1024 * 1024
ALLOWED_SAMPLE_METHODS = {"adaptive", "center", "median", "majority"}
ALLOWED_BACKENDS = {"auto", "opencv", "numpy"}


def _module_root() -> Path:
    return Path(__file__).resolve().parent


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(_module_root() / "templates"),
        static_folder=str(_module_root() / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_BATCH_BYTES

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "opencv": cv2.__version__})

    @app.post("/api/process")
    def process_image():
        uploaded = request.files.get("image")
        if uploaded is None or not uploaded.filename:
            return _error("请选择一张图片。")

        try:
            options = _parse_options(request.form)
            result = _process_upload(uploaded, options, include_overlay=True)
            return jsonify(
                {
                    "result": _encode_png(result["output"]),
                    "overlay": _encode_png(result["overlay"]),
                    "diagnostics": result["diagnostics"],
                }
            )
        except ValueError as exc:
            return _error(str(exc))
        except Exception:
            app.logger.exception("Image processing failed")
            return _error("处理图片时发生错误，请尝试其他参数或图片。", 500)

    @app.post("/api/process-batch")
    def process_batch():
        uploaded_files = [file for file in request.files.getlist("images") if file.filename]
        if not uploaded_files:
            return _error("请选择至少一张图片。")
        if len(uploaded_files) > MAX_BATCH_FILES:
            return _error(f"单次最多处理 {MAX_BATCH_FILES} 张图片。")

        try:
            options = _parse_options(request.form)
            export_scale = _bounded_int(request.form, "export_scale", 1, 16, 1)
            started = time.perf_counter()
            report: list[dict[str, Any]] = []
            archive_buffer = io.BytesIO()

            with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for uploaded in uploaded_files:
                    safe_path = _safe_archive_path(uploaded.filename)
                    try:
                        result = _process_upload(uploaded, options, include_overlay=False)
                        output = result["output"]
                        if export_scale > 1:
                            output = cv2.resize(
                                output,
                                (output.shape[1] * export_scale, output.shape[0] * export_scale),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        archive.writestr(_result_archive_path(safe_path), _png_bytes(output))
                        report.append(
                            {
                                "file": safe_path,
                                "status": "success",
                                "output_width": result["diagnostics"]["output_width"],
                                "output_height": result["diagnostics"]["output_height"],
                            }
                        )
                    except Exception as exc:
                        report.append({"file": safe_path, "status": "failed", "error": str(exc)})

                summary = {
                    "total": len(report),
                    "success": sum(item["status"] == "success" for item in report),
                    "failed": sum(item["status"] == "failed" for item in report),
                    "export_scale": export_scale,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "files": report,
                }
                archive.writestr(
                    "perfect-pixel-report.json",
                    json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"),
                )

            archive_buffer.seek(0)
            response = send_file(
                archive_buffer,
                mimetype="application/zip",
                as_attachment=True,
                download_name="perfect-pixel-batch.zip",
            )
            response.headers["X-Batch-Total"] = str(summary["total"])
            response.headers["X-Batch-Success"] = str(summary["success"])
            response.headers["X-Batch-Failed"] = str(summary["failed"])
            response.headers["X-Batch-Elapsed-Ms"] = str(summary["elapsed_ms"])
            return response
        except ValueError as exc:
            return _error(str(exc))
        except Exception:
            app.logger.exception("Batch image processing failed")
            return _error("批量处理时发生错误，请减少图片数量或调整参数。", 500)

    @app.errorhandler(413)
    def too_large(_error):
        return _error_response("上传内容过大：单张图片不能超过 20 MB，整批不能超过 300 MB。", 413)

    return app


def _parse_options(form: Any) -> dict[str, Any]:
    sample_method = form.get("sample_method", "adaptive")
    backend = form.get("backend", "auto")
    if sample_method not in ALLOWED_SAMPLE_METHODS:
        raise ValueError("无效的采样方式。")
    if backend not in ALLOWED_BACKENDS:
        raise ValueError("无效的处理后端。")

    auto_grid = form.get("auto_grid", "true").lower() == "true"
    grid_size = None
    if not auto_grid:
        grid_size = (
            _bounded_int(form, "grid_width", 2, MAX_GRID_AXIS),
            _bounded_int(form, "grid_height", 2, MAX_GRID_AXIS),
        )

    return {
        "sample_method": sample_method,
        "backend": backend,
        "grid_size": grid_size,
        "min_size": _bounded_float(form, "min_size", 1, 32, 4),
        "peak_width": _bounded_int(form, "peak_width", 2, 24, 6),
        "refine_intensity": _bounded_float(form, "refine_intensity", 0, 0.5, 0.25),
        "fix_square": form.get("fix_square", "true").lower() == "true",
    }


def _bounded_int(form: Any, name: str, minimum: int, maximum: int, default: int | None = None) -> int:
    raw = form.get(name)
    if (raw is None or raw == "") and default is not None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是整数。") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间。")
    return value


def _bounded_float(
    form: Any, name: str, minimum: float, maximum: float, default: float
) -> float:
    try:
        value = float(form.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数字。") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间。")
    return value


def _load_backend(preference: str):
    if preference == "numpy":
        from perfect_pixel import perfect_pixel_noCV2 as backend

        return "NumPy 轻量后端", backend

    if preference in {"auto", "opencv"}:
        from perfect_pixel import perfect_pixel as backend

        return "OpenCV 高性能后端", backend

    raise ValueError("无效的处理后端。")


def _process_upload(uploaded: Any, options: dict[str, Any], include_overlay: bool) -> dict[str, Any]:
    started = time.perf_counter()
    raw = uploaded.read()
    if not raw:
        raise ValueError("图片内容为空。")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("单张图片不能超过 20 MB。")

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("无法识别图片格式，请使用 PNG、JPEG、WEBP 或 BMP。")

    rgb, alpha = _decode_color_and_alpha(image)
    analysis_rgb = _composite_for_analysis(rgb, alpha)
    if max(rgb.shape[:2]) > 4096:
        raise ValueError("图片边长不能超过 4096 像素。")

    backend_name, backend = _load_backend(options["backend"])
    grid_size = options["grid_size"]
    ranking = None
    if grid_size is None:
        ranking = backend.detect_grid_candidates(
            analysis_rgb,
            peak_width=options["peak_width"],
            max_ratio=1.5,
            min_size=options["min_size"],
        )
        best_candidate = ranking["best"]
        if best_candidate is None:
            raise ValueError("未能自动识别网格。请尝试手动指定网格尺寸。")
        grid_size = (
            best_candidate["grid_width"],
            best_candidate["grid_height"],
        )

    _validate_grid_size(grid_size, rgb.shape[1], rgb.shape[0])
    x_coords, y_coords = backend.refine_grids(
        analysis_rgb, grid_size[0], grid_size[1], options["refine_intensity"]
    )
    output_rgb = _sample_grid(backend, rgb, x_coords, y_coords, options["sample_method"])
    output = output_rgb
    if alpha is not None:
        output_alpha = _sample_grid(backend, alpha, x_coords, y_coords, "adaptive")
        output = np.dstack((output_rgb, output_alpha))
    height, width = output.shape[:2]

    overlay = None
    if include_overlay:
        overlay = _draw_grid_overlay(
            np.dstack((rgb, alpha)) if alpha is not None else rgb,
            x_coords,
            y_coords,
        )

    return {
        "output": output,
        "overlay": overlay,
        "diagnostics": {
            "input_width": int(rgb.shape[1]),
            "input_height": int(rgb.shape[0]),
            "detected_grid_width": int(grid_size[0]),
            "detected_grid_height": int(grid_size[1]),
            "output_width": int(width),
            "output_height": int(height),
            "cell_width": round(rgb.shape[1] / grid_size[0], 2),
            "cell_height": round(rgb.shape[0] / grid_size[1], 2),
            "backend": backend_name,
            "sample_method": options["sample_method"],
            "has_alpha": alpha is not None,
            "grid_confidence": ranking["confidence"] if ranking else None,
            "grid_candidates": ranking["alternatives"] if ranking else [],
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        },
    }


def _validate_grid_size(grid_size: tuple[int, int], image_width: int, image_height: int) -> None:
    grid_width, grid_height = grid_size
    if grid_width > image_width or grid_height > image_height:
        raise ValueError(
            f"网格尺寸 {grid_width} × {grid_height} 不能超过原图尺寸 "
            f"{image_width} × {image_height}。"
        )
    if grid_width * grid_height > MAX_OUTPUT_PIXELS:
        raise ValueError(
            f"输出网格最多包含 {MAX_OUTPUT_PIXELS:,} 个像素，请降低网格宽或高。"
        )


def _decode_color_and_alpha(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), None
    if image.shape[2] == 2:
        rgb = cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB)
        return rgb, image[..., 1]
    if image.shape[2] == 4:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        return rgba[..., :3], rgba[..., 3]
    if image.shape[2] == 1:
        return cv2.cvtColor(image[..., 0], cv2.COLOR_GRAY2RGB), None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), None


def _composite_for_analysis(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return rgb
    opacity = alpha.astype(np.float32)[..., None] / 255.0
    composite = rgb.astype(np.float32) * opacity + 255.0 * (1.0 - opacity)
    return np.clip(np.rint(composite), 0, 255).astype(np.uint8)


def _sample_grid(backend: Any, image: np.ndarray, x_coords, y_coords, method: str) -> np.ndarray:
    if method == "adaptive":
        return backend.sample_adaptive(image, x_coords, y_coords)
    if method == "majority":
        return backend.sample_majority(image, x_coords, y_coords)
    if method == "median":
        sampled = backend.sample_median(image, x_coords, y_coords)
        return sampled[..., 0] if image.ndim == 2 and sampled.ndim == 3 else sampled
    return backend.sample_center(image, x_coords, y_coords)


def _safe_archive_path(filename: str) -> str:
    parts = []
    for part in filename.replace("\\", "/").split("/"):
        clean = re.sub(r'[<>:"|?*\x00-\x1f]', "_", part).strip(" .")
        if clean and clean not in {".", ".."}:
            parts.append(clean)
    return "/".join(parts[-12:]) or "image"


def _result_archive_path(source_path: str) -> str:
    path = Path(source_path)
    filename = f"{path.stem}_perfect.png"
    parent = path.parent.as_posix()
    return f"results/{parent}/{filename}" if parent != "." else f"results/{filename}"


def _draw_grid_overlay(
    image: np.ndarray, x_coords: list[int], y_coords: list[int]
) -> np.ndarray:
    overlay = image.copy()
    color = (43, 226, 181, 255) if image.ndim == 3 and image.shape[2] == 4 else (43, 226, 181)
    for x in x_coords:
        cv2.line(overlay, (int(x), 0), (int(x), overlay.shape[0] - 1), color, 1)
    for y in y_coords:
        cv2.line(overlay, (0, int(y)), (overlay.shape[1] - 1, int(y)), color, 1)
    return cv2.addWeighted(image, 0.72, overlay, 0.28, 0)


def _encode_png(image: np.ndarray) -> str:
    payload = base64.b64encode(_png_bytes(image)).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _png_bytes(image: np.ndarray) -> bytes:
    if image.ndim == 3 and image.shape[2] == 4:
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    elif image.ndim == 3:
        encoded_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        encoded_image = image
    success, encoded = cv2.imencode(".png", encoded_image)
    if not success:
        raise RuntimeError("PNG encoding failed")
    return encoded.tobytes()


def _error(message: str, status: int = 400):
    return _error_response(message, status)


def _error_response(message: str, status: int):
    return jsonify({"error": message}), status


app = create_app()
